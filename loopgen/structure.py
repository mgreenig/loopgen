"""
Structure classes for storing sequences and atomic coordinates (N, CA, C, and CB).
"""

from __future__ import annotations

from typing import Union, Sequence, Optional, Any, Iterable, Dict, Tuple, List
from pathlib import Path
from enum import Enum

import torch
import h5py
import numpy as np

from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure as BioPDBStructure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

from einops import einsum
from torch_geometric.typing import PairTensor
from torch_scatter import scatter_mean
from e3nn.o3 import axis_angle_to_matrix

from .utils import (
    get_unit_normals,
    get_covalent_bonds,
    get_dihedral_angles,
    combine_coords,
    standardise_batch_and_ptr,
    expand_batch_tensor,
)


# stores indices under the 3-letter code of each AA
AminoAcid3 = Enum(
    "AminoAcid3",
    [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ],
    start=0,
)

# stores 1 letter codes but maintains index ordering of 3 letter codes
AminoAcid1 = Enum(
    "AminoAcid1", [protein_letters_3to1[aa] for aa in AminoAcid3.__members__], start=0
)

# the PDB codes for all backbone atoms
BackboneAtoms = Enum("BackboneAtoms", ["CA", "C", "O", "N"])

# minimum and maximum covalent bond lengths
MIN_COV_BOND_LEN: float = 0.4
MAX_COV_BOND_LEN: float = 1.9


class BondLengths(Enum):
    """
    Enum for storing backbone bond lengths, taken from:

    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA = 1.459
    CA_C = 1.525
    C_N = 1.336
    C_O = 1.229


class BondLengthStdDevs(Enum):
    """
    Enum for storing standard deviations in backbone bond lengths, taken from:

    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA = 0.020
    CA_C = 0.026
    C_N = 0.023


class BondAngles(Enum):
    """
    Enum for storing backbone bond angles, taken from:

    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA_C = 1.937
    CA_C_N = 2.046
    C_N_CA = 2.124


class BondAngleStdDevs(Enum):
    """
    Enum for storing backbone bond angle standard deviations, taken from:

    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA_C = 0.047
    CA_C_N = 0.038
    C_N_CA = 0.044


class BondAngleCosineStdDevs(Enum):
    """
    Standard deviation of the cosine of each bond angle, estimated
    with a simulation.
    """

    N_CA_C = 0.044
    CA_C_N = 0.034
    C_N_CA = 0.037


class AtomVanDerWaalRadii(Enum):
    """
    Enum for storing atomic van der waal radii, taken from table 1:

    Batsanov, S.S. Van der Waals Radii of Elements.
    Inorganic Materials 37, 871–885 (2001). https://doi.org/10.1023/A:1011625728803
    """

    C = 1.77
    N = 1.64
    O = 1.58
    S = 1.81


class AtomMasses(Enum):
    """
    Enum for storing atomic masses.
    """

    C = 12.011
    N = 14.007
    O = 15.999
    S = 32.06


class OrientationFrames:
    """
    Class for storing orientation frames for a set of residues. Specifically a single orientation frame
    consists of a (3, 3) rotation matrix specifying the orientation of the residue and a (3,) translation vector
    specifying the location of the alpha carbon.

    The class can store more than one rotation/translation,
    i.e. it can store a tensor of rotations of shape (N, 3, 3) and a tensor of translations of shape (N, 3).
    Note that the first dimension of the two tensors must be the same size.
    """

    __slots__ = ("rotations", "translations", "batch", "ptr")

    def __init__(
        self,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
    ):
        if rotations.shape[0] != translations.shape[0]:
            raise ValueError("Rotations and translations shapes do not match.")

        if len(rotations.shape) < 3:
            if len(rotations.shape) == 2:
                rotations = rotations.unsqueeze(0)
            else:
                raise ValueError(
                    "Rotations tensor must be at least rank 2, got rank {}.".format(
                        len(rotations.shape)
                    )
                )

        if len(translations.shape) < 2:
            if len(translations.shape) == 1:
                translations = translations.unsqueeze(0)
            else:
                raise ValueError(
                    "Translations tensor must be at least rank 1, got rank {}.".format(
                        len(translations.shape)
                    )
                )

        self.rotations = rotations
        self.translations = translations

        batch, ptr = standardise_batch_and_ptr(batch, ptr)

        self.batch = batch
        self.ptr = ptr

    def __repr__(self) -> str:
        return "{}({}={})".format(
            self.__class__.__name__, "num_frames", self.num_residues
        )

    @property
    def has_batch(self) -> bool:
        """
        Boolean property indicating whether the OrientationFrames have a batch assignment vector attribute.
        """
        return self.batch is not None

    @classmethod
    def from_three_points(
        cls,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
    ) -> OrientationFrames:
        """
        Get rotations and translations from three sets of 3-D points via Gram-Schmidt process.
        In proteins, these three points are typically N, CA, and C coordinates.

        :param x1: Tensor of shape (..., 3)
        :param x2: Tensor of shape (..., 3)
        :param x3: Tensor of shape (..., 3)
        :param batch: Batch assignment vector of length N.
        :param ptr: Batch pointer vector of length equal to the number of batches plus 1.
        :return: OrientationFrames object containing the relevant rotations/translations.
        """

        v1 = x3 - x2
        v2 = x1 - x2

        return cls.from_two_vectors(v1, v2, x2, batch, ptr)

    @classmethod
    def from_two_vectors(
        cls,
        v1: torch.Tensor,
        v2: torch.Tensor,
        translations: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
    ) -> OrientationFrames:
        """
        Get rotations and translations from three two 3-D vectors via Gram-Schmidt process.
        Vectors in `v1` are taken as the first component of the orthogonal basis, then the component of `v2`
        orthogonal to `v1`, and finally the cross product of `v1` and the orthogonalised `v2`. In
        this case the translations must be provided as well.

        In proteins, these two vectors are typically N-CA, and C-CA bond vectors,
        and the translations are the CA coordinates.

        :param v1: Tensor of shape (N, 3)
        :param v2: Tensor of shape (N, 3)
        :param translations: Tensor of translations of shape (N, 3).
        :param batch: Batch assignment vector of length N.
        :param ptr: Batch pointer vector of length equal to the number of batches plus 1.
        :return: OrientationFrames object containing the relevant rotations/translations.
        """

        e1 = v1 / torch.linalg.norm(v1, dim=-1).unsqueeze(-1)
        u2 = v2 - e1 * (torch.sum(e1 * v2, dim=-1).unsqueeze(-1))
        e2 = u2 / torch.linalg.norm(u2, dim=-1).unsqueeze(-1)
        e3 = torch.cross(e1, e2, dim=-1)

        rotations = torch.stack([e1, e2, e3], dim=-2).transpose(-2, -1)
        rotations = torch.nan_to_num(rotations)

        return cls(rotations, translations, batch, ptr)

    def to(self, device: torch.device) -> OrientationFrames:
        """
        Returns a new OrientationFrames instance moved to the specified device.
        """

        rotations = self.rotations.to(device)
        translations = self.translations.to(device)

        if self.has_batch:
            batch = self.batch.to(device)
            ptr = self.ptr.to(device)
        else:
            batch = None
            ptr = None

        return self.__class__(rotations, translations, batch, ptr)

    def inverse(self):
        """
        Gets the inverse of the frame(s). For a frame (R, t),
        the inverse is (R^T, -R^T t) (assuming R is a rotation).
        """
        new_translations = -einsum(
            self.rotations, self.translations, "... j i, ... j -> ... i"
        )
        return OrientationFrames(
            self.rotations.transpose(-2, -1),
            new_translations,
        )

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """
        Applies the rotations/translations to a set of N 3-dimensional vectors `points`.

        :param points: Tensor of shape (N, 3).
        :returns: Tensor of shape (N, 3) containing the transformed points.
        """

        rotated = einsum(self.rotations, points, "... i j, ... j -> ... i")

        if len(rotated.shape) > 2:
            transformed = rotated + self.translations.unsqueeze(-2)
            transformed = transformed.squeeze(-2)
        else:
            transformed = rotated + self.translations

        return transformed

    def apply_inverse(self, points: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse rotations/translations to a set of N 3-dimensional vectors `points`. Points
        should have shape (N, 3).

        :param points: Tensor of shape (N, 3).
        :returns: Tensor of shape (N, 3) containing the transformed points.
        """

        inv = self.inverse()
        return inv.apply(points)

    def compose(self, *frames: OrientationFrames) -> OrientationFrames:
        """
        Composes a sequence of orientation frames to the current instance of frames
        by applying rotations and translations sequentially.
        """

        t = self.translations.clone()
        R = self.rotations.clone()

        for frame in frames:
            R = torch.matmul(frame.rotations, R)
            t = (
                einsum(
                    frame.rotations,
                    t,
                    "... i j, ... j -> ... i",
                )
                + frame.translations
            )

        return OrientationFrames(R, t)

    @property
    def num_residues(self) -> int:
        """
        Returns the number of residues in the orientation frame(s),
        defined as the size of the third-to-last dimension.
        """
        return self.rotations.shape[-3]

    def __len__(self) -> int:
        return self.rotations.shape[0]

    def __getitem__(self, idx: Union[int, slice, Sequence]) -> OrientationFrames:
        """
        Indexes elements in the set of orientation frames, i.e. along the -3 dimension for rotations
        and the -2 dimension for translations.
        """

        return OrientationFrames(
            self.rotations[..., idx, :, :].reshape(
                (-1, self.rotations.shape[-2], self.rotations.shape[-1])
            ),
            self.translations[..., idx, :].reshape((-1, self.translations.shape[-1])),
        )

    def __setitem__(self, idx: Union[int, slice, Sequence], values: PairTensor) -> None:
        """
        Sets an item via an index and a tuple of (rotation, translation) to be set at the provided index.

        Usage: `self[idx] = (rotation, translation)`
        """

        new_rotation, new_translation = values
        self.rotations[..., idx, :, :] = new_rotation
        self.translations[..., idx, :] = new_translation

    def center(
        self, return_centre: bool = True
    ) -> Union[OrientationFrames, Tuple[OrientationFrames, torch.Tensor]]:
        """
        Centers the OrientationFrames at the origin, returning a new OrientationFrames object
        with centered translations. If `return_centre` is True, also returns the center of mass,
        which will have first dimension equal to the number of batches (one centre of mass per batch).
        """
        # expand the epitope batch tensor to accommodate the individual atom coords
        if not self.has_batch:
            batch = torch.zeros(
                len(self), device=self.translations.device, dtype=torch.int64
            )
        else:
            batch = self.batch

        centre_of_mass = scatter_mean(self.translations, batch, dim=0)
        centred_frames = self.translate(-centre_of_mass[batch])

        if not return_centre:
            return centred_frames

        return centred_frames, centre_of_mass

    def translate(
        self,
        translation: torch.Tensor,
    ) -> OrientationFrames:
        """
        Translates the OrientationFrames by applying a translation to the translations.
        The input translation must be broadcastable with the current translations.
        """
        return OrientationFrames(
            self.rotations,
            self.translations + translation,
            batch=self.batch,
            ptr=self.ptr,
        )

    def to_backbone_coords(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts the rotations and translations for each residue into
        backbone atomic coordinates for each residue. Returns a 3-tuple
        of tensors (N coords, CA coords, C coords).

        Assumes `self.translations` are the alpha carbon coordinates,
        the first column vector of each rotation in `self.rotations` is the direction
        of the C-CA bond, the next column vector is the component of the N-CA bond
        orthogonal to the C-CA bond, and the third column vector is the cross product.
        """
        C_coords = (self.rotations[..., 0] * BondLengths.CA_C.value) + self.translations
        CA_coords = self.translations

        # get the N-CA bond by rotating the second column vector to get the desired
        # bond angle
        N_bond_rotation = axis_angle_to_matrix(
            self.rotations[..., -1],
            torch.as_tensor(
                BondAngles.N_CA_C.value - (np.pi / 2),
                device=self.rotations.device,
                dtype=self.rotations.dtype,
            ),
        )
        N_bonds = torch.matmul(
            N_bond_rotation,
            self.rotations[..., 1:2] * BondLengths.N_CA.value,
        ).squeeze(-1)
        N_coords = N_bonds + self.translations

        return N_coords, CA_coords, C_coords

    @classmethod
    def combine(
        cls,
        orientation_frames_sequence: Sequence[OrientationFrames],
        add_batch: bool = True,
    ) -> OrientationFrames:
        """
        Concatenates the rotations/translations from a sequence of orientation frames along the first dimension,
        returning a new instance of OrientationFrames.
        """

        concat_rotations = torch.cat(
            [frames.rotations for frames in orientation_frames_sequence], dim=0
        )
        concat_translations = torch.cat(
            [frames.translations for frames in orientation_frames_sequence], dim=0
        )

        # if batch vectors are present, concatenate them
        if all([frames.has_batch for frames in orientation_frames_sequence]):
            batch = torch.cat(
                [frames.batch for frames in orientation_frames_sequence], dim=-1
            )
            ptr = torch.cat(
                [frames.ptr for frames in orientation_frames_sequence], dim=-1
            )
        elif add_batch is True:
            struct_lengths = torch.as_tensor(
                [len(frames) for frames in orientation_frames_sequence],
                device=concat_rotations.device,
            )
            batch = torch.repeat_interleave(
                torch.arange(
                    len(orientation_frames_sequence), device=concat_rotations.device
                ),
                repeats=struct_lengths,
            )
            ptr = torch.nn.functional.pad(torch.cumsum(struct_lengths, dim=-1), (1, 0))
        else:
            batch = None
            ptr = None

        return cls(concat_rotations, concat_translations, batch, ptr)

    def split(self) -> List[OrientationFrames]:
        """
        :returns: If the OrientationFrames is batched, a list of OrientationFrames objects, one for each
            batch element. If not, returns a list containing only the OrientationFrames itself.
        """

        if not self.has_batch:
            return [self.clone()]

        num_per_batch = torch.diff(self.ptr)
        indices = torch.split(
            torch.arange(len(self), device=self.batch.device),
            num_per_batch.tolist(),
        )
        frames = [self[idx] for idx in indices]
        for i in range(len(frames)):
            frames[i].batch = None
            frames[i].ptr = None

        return frames

    def clone(self) -> OrientationFrames:
        """
        Clones the frame's underlying tensors and returns a new OrientationFrames object.
        """

        return self.__class__(
            self.rotations.clone(),
            self.translations.clone(),
            batch=self.batch.clone() if self.batch is not None else None,
            ptr=self.ptr.clone() if self.ptr is not None else None,
        )

    def detach(self) -> OrientationFrames:
        """
        Detaches the frame's underlying tensors from the computational graph,
        returning a new OrientationFrames object.
        """

        return self.__class__(
            self.rotations.detach(),
            self.translations.detach(),
            batch=self.batch.detach() if self.batch is not None else None,
            ptr=self.ptr.detach() if self.ptr is not None else None,
        )

    def repeat(self, n: int) -> OrientationFrames:
        """
        Repeats the frames n times. Behaviour is similar to numpy.tile()
        and torch.repeat(), where the entire frames' rotations/translations are copied and stacked
        below the existing frames. This adds new batch assignment and pointer tensors
        regardless of whether the structure was batched or not.

        :param n: Number of times to repeat the frames.
        :returns: OrientationFrames with repeated coordinates. The n = 1 case returns the original frames,
            while n > 1 returns a new frames object.
        """
        if n <= 1:
            if n == 1:
                return self
            else:
                raise ValueError(f"n must be greater than or equal to 1, got {n}.")

        rotations_repeated = (
            self.rotations.unsqueeze(0)
            .expand(n, -1, -1, -1)
            .flatten(start_dim=0, end_dim=1)
        )
        translations_repeated = (
            self.translations.unsqueeze(0)
            .expand(n, -1, -1)
            .flatten(start_dim=0, end_dim=1)
        )

        if self.has_batch:
            n_per_struct = torch.diff(self.ptr)
        else:
            n_per_struct = torch.as_tensor([len(self)], device=self.rotations.device)

        n_per_struct_repeated = (
            n_per_struct.unsqueeze(0).expand(n, -1).flatten(start_dim=0, end_dim=1)
        )
        new_ptr = torch.nn.functional.pad(
            torch.cumsum(n_per_struct_repeated, dim=-1), (1, 0)
        )
        new_batch, _ = standardise_batch_and_ptr(None, new_ptr)

        return self.__class__(
            rotations_repeated,
            translations_repeated,
            batch=new_batch,
            ptr=new_ptr,
        )


def impute_CB_coords(
    N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    """
    Imputes beta carbon coordinates by assuming tetrahedral geometry around each CA atom,
    implemented as in the Geometric Vector Perceptron (https://arxiv.org/pdf/2009.01411.pdf).
    """
    N_CA = N - CA
    C_CA = C - CA
    cross_prods = torch.nn.functional.normalize(torch.cross(N_CA, C_CA, dim=-1), dim=-1)
    bond_bisectors = torch.nn.functional.normalize(N_CA + C_CA, dim=-1)
    CB_bonds = np.sqrt(1 / 3) * cross_prods - np.sqrt(2 / 3) * bond_bisectors
    CB_coords = CA + CB_bonds
    return CB_coords


def impute_O_coords(
    N: torch.Tensor,
    CA: torch.Tensor,
    C: torch.Tensor,
    ptr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Imputes carboxyl O coordinates using the negative bisector of the CA-C and C-N bonds
    (if the C atom is covalently bonded to an included N atom) or a rotation of the last residue's
    CA-C bond (if the C atom is not covalently bonded to an included N atom).

    :param N: Tensor of shape (N, 3) containing N coordinates.
    :param CA: Tensor of shape (N, 3) containing CA coordinates.
    :param C: Tensor of shape (N, 3) containing C coordinates.
    :param ptr: Batch pointer vector of length equal to the number of batches plus 1.
    :returns: Tensor of shape (N, 3) containing O coordinates.
    """
    if not N.shape == CA.shape == C.shape:
        raise ValueError(
            "N, CA, and C tensors must have the same shape, got {} {} {}".format(
                N.shape, CA.shape, C.shape
            )
        )

    if ptr is None:
        ptr = torch.as_tensor([0, len(N)], device=N.device)

    N_expanded, pad_mask = expand_batch_tensor(
        N, torch.diff(ptr), return_padding_mask=True
    )

    # Bonds for covalently bonded C atoms are the negative bisector of the CA-C and C-N bonds
    N_shifted = torch.roll(N_expanded, -1, dims=1)[~pad_mask]
    CA_C_bonds = CA - C
    N_C_bonds = N_shifted - C
    bond_bisectors = torch.nn.functional.normalize(CA_C_bonds + N_C_bonds, dim=-1)
    O_bonds = -bond_bisectors * BondLengths["C_O"].value

    # For the C terminus of the structure we just assume O lies in the same plane as the N-CA-C atoms
    c_terminus_indices = ptr[1:] - 1
    last_residue_CA_C_bonds = CA_C_bonds[c_terminus_indices]
    last_residue_CA_N_bonds = (N - CA)[c_terminus_indices]

    rot_axis = get_unit_normals(last_residue_CA_C_bonds, last_residue_CA_N_bonds)
    rot_matrix = axis_angle_to_matrix(
        rot_axis, torch.as_tensor(np.pi / 3, device=rot_axis.device)
    )

    last_O_C_bonds = (
        torch.nn.functional.normalize(
            einsum(rot_matrix, -last_residue_CA_C_bonds, "... i j, ... j -> ... i")
        )
        * BondLengths["C_O"].value
    )
    O_bonds[c_terminus_indices] = last_O_C_bonds

    return C + O_bonds


class Structure:
    """
    Class for storing simplified atom coordinate and amino acid sequence information for a protein structure.
    Specifically, this class stores N, CA, C, and CB coordinates for each residue, as well as the amino acid
    identity. Coordinate tensor attributes are expected to be of shape (N, 3) where N is the number of residues.

    All slots added to this class should be tensors where the size of the first dimension corresponds to
    the number of residues in the structure.
    """

    __slots__ = (
        "N_coords",
        "CA_coords",
        "C_coords",
        "CB_coords",
        "_sequence",
        "orientation_frames",
        "batch",
        "ptr",
    )

    def __init__(
        self,
        N_coords: torch.Tensor,
        CA_coords: torch.Tensor,
        C_coords: torch.Tensor,
        CB_coords: torch.Tensor,
        sequence: torch.Tensor,
        orientation_frames: Optional[OrientationFrames] = None,
        batch: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
    ):
        # ensure all coordinate tensors are rank 2 at least and sequence is rank 1
        if len(N_coords.shape) < 2:
            N_coords = N_coords.unsqueeze(0)
        if len(CA_coords.shape) < 2:
            CA_coords = CA_coords.unsqueeze(0)
        if len(C_coords.shape) < 2:
            C_coords = C_coords.unsqueeze(0)
        if len(CB_coords.shape) < 2:
            CB_coords = CB_coords.unsqueeze(0)
        if len(sequence.shape) < 1:
            sequence = sequence.unsqueeze(0)

        self.N_coords = N_coords
        self.CA_coords = CA_coords
        self.C_coords = C_coords
        self.CB_coords = CB_coords
        self.sequence = sequence

        batch, ptr = standardise_batch_and_ptr(batch, ptr)

        self.batch = batch
        self.ptr = ptr

        if orientation_frames is None and not self.missing_atoms:
            orientation_frames = OrientationFrames.from_three_points(
                self.N_coords, self.CA_coords, self.C_coords, self.batch, self.ptr
            )

        self.orientation_frames = orientation_frames

    @property
    def sequence(self) -> torch.Tensor:
        """The amino acid sequence of the structure."""
        return self._sequence

    @sequence.setter
    def sequence(self, value: torch.Tensor) -> None:
        """
        Sets the amino acid sequence of the structure,
        setting the CB coords of glycine residues to their CA coords.
        """
        self._sequence = value

        # set CB coordinates for glycine residues to be same as CA
        if not self.missing_atoms:
            gly_value = AminoAcid3["GLY"].value
            self.CB_coords[self._sequence == gly_value] = self.CA_coords[
                self._sequence == gly_value
            ]

    def write_to_hdf5(self, h5py_group: h5py.Group):
        """
        Writes structure information (coordinates + sequence) to a h5py group.
        """

        h5py_group.create_dataset(
            "sequence", dtype=np.int8, data=self.sequence.cpu().numpy()
        )
        h5py_group.create_dataset("CA_coords", data=self.CA_coords.cpu().numpy())
        h5py_group.create_dataset("C_coords", data=self.C_coords.cpu().numpy())
        h5py_group.create_dataset("N_coords", data=self.N_coords.cpu().numpy())
        h5py_group.create_dataset("CB_coords", data=self.CB_coords.cpu().numpy())

    def write_to_pdb(
        self,
        filepath: str,
        pdb_id: Optional[str] = None,
        residue_chains: Optional[Union[str, Sequence[str]]] = None,
        residue_numbers: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Writes structure information (coordinates + sequence) to a PDB file.
        Users can provide chain and residue number information to be written to the PDB file;
        if these are None, the chain is set to A and the residue numbers are set to integers
        starting from 1. If this is called on a batched structure, the different structures will
        be saved as different models in the PDB file.

        :param filepath: Path to the PDB file to be written.
        :param pdb_id: Optional PDB ID for the structure. If this is not present, the stem
            of the filepath is used.
        :param residue_chains: Optional chain information to be written to the PDB file. This can
            either be a single string (which assumes all residues are on that chain)
            or a sequence of strings, the same length as the number of residues.
        :param residue_numbers: Optional residue number information to be written to the PDB file.
            This should be a sequence of strings, the same length as the number of residues.
        """

        if self.missing_atoms:
            raise ValueError("Cannot write PDB file for structure with missing atoms.")

        if pdb_id is None:
            pdb_id = Path(filepath).stem

        if residue_chains is None:
            residue_chains = ["A"] * len(self)
        elif isinstance(residue_chains, str):
            residue_chains = [residue_chains] * len(self)

        unique_chains = set(residue_chains)

        if not self.has_batch:
            ptr = torch.as_tensor([0, len(self)], device=self.N_coords.device)
        else:
            ptr = self.ptr

        if residue_numbers is None:
            residue_numbers = []
            for model in range(len(ptr) - 1):
                start = ptr[model]
                end = ptr[model + 1]
                chain_residue_counters = {chain: 1 for chain in unique_chains}
                for chain in residue_chains[start:end]:
                    res_num = chain_residue_counters[chain]
                    residue_numbers.append(res_num)
                    chain_residue_counters[chain] += 1

        if len(residue_chains) != len(self) or len(residue_numbers) != len(self):
            raise ValueError(
                "Chains and residue numbers must be the same length as the number of residues."
            )

        aa3_index_to_name = {v.value: k for k, v in AminoAcid3.__members__.items()}
        structure = BioPDBStructure(pdb_id)
        O_coords = impute_O_coords(
            self.N_coords, self.CA_coords, self.C_coords, self.ptr
        )
        for i, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            model = Model(i)
            model_chains = {
                chain_id: Chain(chain_id) for chain_id in sorted(unique_chains)
            }
            for j in range(start, end):
                chain = residue_chains[j]
                res_num = residue_numbers[j]
                res_chain = model_chains[chain]
                res = Residue(
                    (" ", res_num, " "),
                    aa3_index_to_name[self.sequence[j].item()],
                    "",
                )

                res_N = Atom(
                    "N", self.N_coords[j].cpu().numpy(), 0.0, 1.0, " ", "N", 0, "N"
                )
                res_CA = Atom(
                    "CA", self.CA_coords[j].cpu().numpy(), 0.0, 1.0, " ", "CA", 0, "C"
                )
                res_C = Atom(
                    "C", self.C_coords[j].cpu().numpy(), 0.0, 1.0, " ", "C", 0, "C"
                )

                res.add(res_N)
                res.add(res_CA)
                res.add(res_C)

                # add O atom if the structure is a linear chain
                if isinstance(self, LinearStructure):
                    res_O = Atom(
                        "O", O_coords[j].cpu().numpy(), 0.0, 1.0, " ", "O", 0, "O"
                    )
                    res.add(res_O)

                res_CB = Atom(
                    "CB", self.CB_coords[j].cpu().numpy(), 0.0, 1.0, " ", "CB", 0, "C"
                )
                res.add(res_CB)
                res_chain.add(res)

            for chain in model_chains.values():
                model.add(chain)

            structure.add(model)

        io = PDBIO()
        io.set_structure(structure)
        io.save(filepath)

    @classmethod
    def from_pdb_residues(
        cls,
        residues: Iterable[Residue],
        residue_atoms: Optional[Iterable[Dict[str, Atom]]] = None,
    ) -> Structure:
        """
        Converts an iterable of Bio.PDB.Residue objects into a Structure.

        :param residues: Iterable of residue objects.
        :param residue_atoms: Optional iterable of dictionaries mapping atom names
            to Atom objects for each residue.
        :return: Structure object.
        """

        # get sidechain atom coordinates and atom types
        N_coords = []
        CA_coords = []
        C_coords = []
        CB_coords = []
        sequence = []

        if residue_atoms is None:
            residue_atoms = [{a.get_name(): a for a in r.get_atoms()} for r in residues]

        for i, (res, res_atoms) in enumerate(zip(residues, residue_atoms)):
            res_name = res.get_resname()

            if res_name in AminoAcid3.__members__:
                sequence.append(AminoAcid3[res_name].value)

            if "N" in res_atoms:
                N_coords.append(res_atoms["N"].get_coord())
            if "CA" in res_atoms:
                CA_coords.append(res_atoms["CA"].get_coord())
            if "C" in res_atoms:
                C_coords.append(res_atoms["C"].get_coord())

            if res_name != "GLY":
                if "CB" in res_atoms:
                    CB_coords.append(res_atoms["CB"].get_coord())
            elif "CA" in res_atoms:
                CB_coords.append(res_atoms["CA"].get_coord())

        N_coords = torch.as_tensor(np.array(N_coords))
        CA_coords = torch.as_tensor(np.array(CA_coords))
        C_coords = torch.as_tensor(np.array(C_coords))
        CB_coords = torch.as_tensor(np.array(CB_coords))
        sequence = torch.as_tensor(np.array(sequence))

        return cls(N_coords, CA_coords, C_coords, CB_coords, sequence)

    @classmethod
    def combine(cls, structures: Sequence[Structure], add_batch: bool = True) -> Any:
        """
        Combines a sequence of structures into a single structure object.

        :param structures: Sequence of structures.
        :param add_batch: If True, adds a batch index and batch pointer to the combined structure.
        :returns: Combined structure, containing atom coordinates of all the individual structures.
        """

        N_coords = torch.cat([struct.N_coords for struct in structures], dim=-2)
        CA_coords = torch.cat([struct.CA_coords for struct in structures], dim=-2)
        C_coords = torch.cat([struct.C_coords for struct in structures], dim=-2)
        CB_coords = torch.cat([struct.CB_coords for struct in structures], dim=-2)
        sequence = torch.cat([struct.sequence for struct in structures], dim=-1)

        if all([struct.has_orientation_frames for struct in structures]):
            orientations = OrientationFrames.combine(
                [struct.orientation_frames for struct in structures]
            )
        else:
            orientations = None

        # if batch vectors are present, concatenate them
        if all([struct.has_batch for struct in structures]) and not add_batch:
            batch = torch.cat([struct.batch for struct in structures], dim=-1)
            ptr = torch.cat([struct.ptr for struct in structures], dim=-1)
        elif add_batch:
            struct_lengths = torch.as_tensor(
                [len(struct) for struct in structures], device=N_coords.device
            )
            batch = torch.repeat_interleave(
                torch.arange(len(structures), device=N_coords.device),
                repeats=struct_lengths,
            )
            ptr = torch.nn.functional.pad(torch.cumsum(struct_lengths, dim=-1), (1, 0))
        else:
            batch = None
            ptr = None

        concatenated_structure = cls(
            N_coords=N_coords,
            CA_coords=CA_coords,
            C_coords=C_coords,
            CB_coords=CB_coords,
            sequence=sequence,
            orientation_frames=orientations,
            batch=batch,
            ptr=ptr,
        )

        return concatenated_structure

    @property
    def has_orientation_frames(self) -> bool:
        """
        Boolean property indicating whether the structure has a backbone orientation frames attribute.
        """

        return self.orientation_frames is not None

    @property
    def has_batch(self) -> bool:
        """
        Boolean property indicating whether the structure has a batch assignment vector attribute.
        """

        return self.batch is not None

    def __repr__(self):
        return "{}({}={})".format(
            self.__class__.__name__, "num_residues", self.num_residues
        )

    # indexing returns another structure with a subset of coordinates
    def __getitem__(self, idx: Union[int, slice, torch.Tensor]):
        """
        Index function that returns another structure with a subset of coordinates. Indexing is performed on
        the second-to-last dimension - this is typically the "residue" dimension of the coordinate tensors.
        """

        N_coords = self.N_coords[idx]
        CA_coords = self.CA_coords[idx]
        C_coords = self.C_coords[idx]
        CB_coords = self.CB_coords[idx]
        sequence = self.sequence[idx]

        if self.has_orientation_frames:
            orientation_frames = self.orientation_frames[idx]
        else:
            orientation_frames = None

        if self.has_batch:
            batch = self.batch[idx]
        else:
            batch = None

        return self.__class__(
            N_coords=N_coords,
            CA_coords=CA_coords,
            C_coords=C_coords,
            CB_coords=CB_coords,
            sequence=sequence,
            orientation_frames=orientation_frames,
            batch=batch,
        )

    def __len__(self):
        if self.CA_coords.shape[-1] == 0:
            return 0
        return self.CA_coords.shape[-2]

    @property
    def num_residues(self):
        """The number of underlying residues, equal to len(self)."""
        return len(self)

    @property
    def missing_atoms(self) -> bool:
        """
        Boolean property indicating whether the structure is missing any atoms (N, CA, C, or CB),
        based on whether all the coordinate shapes match and whether they have more than 0 elements.
        """

        same_num_atoms = (
            self.N_coords.shape[-2]
            == self.CA_coords.shape[-2]
            == self.C_coords.shape[-2]
            == self.CB_coords.shape[-2]
            == self.sequence.shape[-1]
        )
        more_than_one_coord = self.N_coords.numel() > 0

        return not (same_num_atoms and more_than_one_coord)

    def to(self, device: torch.device) -> Structure:
        """
        Moves an entire structure to a specified device.

        :param device: torch.device object.
        :return: Structure on the new device.
        """

        N_coords = self.N_coords.to(device)
        CA_coords = self.CA_coords.to(device)
        C_coords = self.C_coords.to(device)
        CB_coords = self.CB_coords.to(device)
        sequence = self.sequence.to(device)

        if self.has_orientation_frames:
            orientation_frames = self.orientation_frames.to(device)
        else:
            orientation_frames = None

        if self.has_batch:
            batch = self.batch.to(device)
        else:
            batch = None

        return self.__class__(
            N_coords=N_coords,
            CA_coords=CA_coords,
            C_coords=C_coords,
            CB_coords=CB_coords,
            sequence=sequence,
            orientation_frames=orientation_frames,
            batch=batch,
        )

    def clone(self):
        """
        Clones the Structure, returning a new Structure object with the same data
        but with the underlying tensors cloned.
        """
        N_coords = self.N_coords.clone()
        CA_coords = self.CA_coords.clone()
        C_coords = self.C_coords.clone()
        CB_coords = self.CB_coords.clone()
        sequence = self.sequence.clone()

        if self.has_orientation_frames:
            orientation_frames = self.orientation_frames.clone()
        else:
            orientation_frames = None

        if self.has_batch:
            batch = self.batch.clone()
            ptr = self.ptr.clone()
        else:
            batch = None
            ptr = None

        return self.__class__(
            N_coords=N_coords,
            CA_coords=CA_coords,
            C_coords=C_coords,
            CB_coords=CB_coords,
            sequence=sequence,
            orientation_frames=orientation_frames,
            batch=batch,
            ptr=ptr,
        )

    def detach(self) -> Structure:
        """
        Detaches the structure's underlying tensors from the computational graph,
        returning a new Structure object.
        """
        N_coords = self.N_coords.detach()
        CA_coords = self.CA_coords.detach()
        C_coords = self.C_coords.detach()
        CB_coords = self.CB_coords.detach()
        sequence = self.sequence.detach()

        if self.has_orientation_frames:
            orientation_frames = self.orientation_frames.detach()
        else:
            orientation_frames = None

        if self.has_batch:
            batch = self.batch.detach()
            ptr = self.ptr.detach()
        else:
            batch = None
            ptr = None

        return self.__class__(
            N_coords=N_coords,
            CA_coords=CA_coords,
            C_coords=C_coords,
            CB_coords=CB_coords,
            sequence=sequence,
            orientation_frames=orientation_frames,
            batch=batch,
            ptr=ptr,
        )

    def split(self) -> List[Structure]:
        """
        :returns: If the structure is batched, a list of Structure objects, one for each
            batch element. If not, returns a list containing only the structure itself.
        """

        if not self.has_batch:
            return [self.clone()]

        num_per_batch = torch.diff(self.ptr)
        indices = torch.split(
            torch.arange(len(self), device=self.batch.device),
            num_per_batch.tolist(),
        )
        structures = [self[idx] for idx in indices]
        for i in range(len(structures)):
            structures[i].batch = None
            structures[i].ptr = None

        return structures

    def translate(
        self,
        translation: torch.Tensor,
    ) -> Structure:
        """
        Translates the entire structure by an input translation.
        The input translation should be broadcastable to the shape of
        each of the structure's coordinates.

        :param translation: Tensor of shape (..., 3) representing one or more translation vectors.
        :returns: Structure with translated coordinates.
        """

        translated_N = self.N_coords + translation
        translated_CA = self.CA_coords + translation
        translated_C = self.C_coords + translation
        translated_CB = self.CB_coords + translation
        if self.has_orientation_frames:
            translated_frames = OrientationFrames(
                self.orientation_frames.rotations,
                self.orientation_frames.translations + translation,
            )
        else:
            translated_frames = None

        return self.__class__(
            N_coords=translated_N,
            CA_coords=translated_CA,
            C_coords=translated_C,
            CB_coords=translated_CB,
            sequence=self.sequence,
            orientation_frames=translated_frames,
            batch=self.batch,
            ptr=self.ptr,
        )

    def rotate(
        self,
        rotation: torch.Tensor,
    ) -> Structure:
        """
        Rotates the entire structure by an input rotation matrix or matrices via a
        left matrix multiplication. The input rotation matrix or matrices
        should be broadcastable to the shape of each of the structure's coordinates.

        :param rotation: Tensor of shape (..., 3, 3) representing one or more rotation matrices
            (determinant of 1, orthogonal).
        :returns: Structure with rotated coordinates.
        """

        rotated_N = einsum(
            rotation,
            self.N_coords,
            "... i j, ... j -> ... i",
        )
        rotated_CA = einsum(
            rotation,
            self.CA_coords,
            "... i j, ... j -> ... i",
        )
        rotated_C = einsum(
            rotation,
            self.C_coords,
            "... i j, ... j -> ... i",
        )
        rotated_CB = einsum(
            rotation,
            self.CB_coords,
            "... i j, ... j -> ... i",
        )

        if self.has_orientation_frames:
            new_rotations = torch.matmul(rotation, self.orientation_frames.rotations)
            new_translations = einsum(
                rotation,
                self.orientation_frames.translations,
                "... i j, ... j -> ... i",
            )
            rotated_frames = OrientationFrames(new_rotations, new_translations)
        else:
            rotated_frames = None

        return self.__class__(
            N_coords=rotated_N,
            CA_coords=rotated_CA,
            C_coords=rotated_C,
            CB_coords=rotated_CB,
            sequence=self.sequence,
            orientation_frames=rotated_frames,
            batch=self.batch,
            ptr=self.ptr,
        )

    def center(
        self, return_centre: bool = True
    ) -> Union[Structure, Tuple[Structure, torch.Tensor]]:
        """
        Centers the Structure at the origin, returning a new Structure object with centered coordinates.
        If `return_centre` is True, also returns the previous center of mass, which will have first dimension
        equal to the number of batches (one centre of mass per batch).

        :param return_centre: If True, returns the previous center of mass.
        :returns: Centered Structure, or tuple of (centered Structure, center of mass).
        """
        if self.missing_atoms:
            raise ValueError(f"Structure has missing atoms, cannot be centered.")
        all_coords = combine_coords(
            self.N_coords,
            self.CA_coords,
            self.C_coords,
            self.CB_coords,
        )
        # expand the batch tensor to accommodate the individual atom coords
        if not self.has_batch:
            batch = torch.zeros(len(self), device=all_coords.device, dtype=torch.int64)
        else:
            batch = self.batch

        batch_expanded = torch.repeat_interleave(batch, 4, dim=0)
        centre_of_mass = scatter_mean(all_coords, batch_expanded, dim=0)
        centred_structure = self.translate(-centre_of_mass[batch])

        if not return_centre:
            return centred_structure

        return centred_structure, centre_of_mass

    def add_gaussian_noise(
        self, std: float, random_state: Optional[int] = None
    ) -> Structure:
        """
        Adds gaussian noise to all the atomic coordinates, returning a new Structure.

        :param std: Standard deviation of the gaussian noise.
        :param random_state: Random seed for reproducibility.
        :returns: A new Structure with added noise.
        """

        if random_state is not None:
            torch.manual_seed(random_state)

        return Structure(
            N_coords=self.N_coords + torch.randn_like(self.N_coords) * std,
            CA_coords=self.CA_coords + torch.randn_like(self.CA_coords) * std,
            C_coords=self.C_coords + torch.randn_like(self.C_coords) * std,
            CB_coords=self.CB_coords + torch.randn_like(self.CB_coords) * std,
            sequence=self.sequence,
            batch=self.batch,
            ptr=self.ptr,
        )

    def scramble_sequence(self):
        """Randomly scrambles the Structure's sequence (i.e. permuting the amino acid identities)."""
        scramble_indices = torch.randperm(
            len(self.sequence),
            device=self.sequence.device,
        )
        scrambled_sequence = self.sequence[scramble_indices]

        old_glycine_mask = (scrambled_sequence != AminoAcid3["GLY"].value) & (
            self.sequence == AminoAcid3["GLY"].value
        )

        scrambled_structure = self.clone()
        scrambled_structure.sequence = scrambled_sequence
        scrambled_structure.CB_coords[old_glycine_mask] = impute_CB_coords(
            scrambled_structure.N_coords[old_glycine_mask],
            scrambled_structure.CA_coords[old_glycine_mask],
            scrambled_structure.C_coords[old_glycine_mask],
        )

        return scrambled_structure

    def repeat(self, n: int):
        """
        Repeats the structure n times. Behaviour is similar to numpy.tile()
        and torch.repeat(), where the entire structure's coordinates are copied and stacked
        below the existing coordinates. This adds new batch assignment and pointer tensors
        regardless of whether the structure was batched or not.

        :param n: Number of times to repeat each structure.
        :returns: New structure with repeated coordinates. The n = 1 case returns the original structure,
            while n > 1 returns a new structure.
        """
        if self.missing_atoms:
            raise NotImplementedError("Cannot repeat structure with missing atoms.")

        if n <= 1:
            if n == 1:
                return self
            else:
                raise ValueError(f"n must be greater than or equal to 1, got {n}.")

        N_coords_repeated = (
            self.N_coords.unsqueeze(0).expand(n, -1, -1).flatten(start_dim=0, end_dim=1)
        )
        CA_coords_repeated = (
            self.CA_coords.unsqueeze(0)
            .expand(n, -1, -1)
            .flatten(start_dim=0, end_dim=1)
        )
        C_coords_repeated = (
            self.C_coords.unsqueeze(0).expand(n, -1, -1).flatten(start_dim=0, end_dim=1)
        )
        CB_coords_repeated = (
            self.CB_coords.unsqueeze(0)
            .expand(n, -1, -1)
            .flatten(start_dim=0, end_dim=1)
        )
        sequence_repeated = (
            self.sequence.unsqueeze(0).expand(n, -1).flatten(start_dim=0, end_dim=1)
        )

        if self.has_orientation_frames:
            frames_repeated = self.orientation_frames.repeat(n)
        else:
            frames_repeated = None

        if self.has_batch:
            n_per_struct = torch.diff(self.ptr)
        else:
            n_per_struct = torch.as_tensor([len(self)], device=self.CA_coords.device)

        n_per_struct_repeated = (
            n_per_struct.unsqueeze(0).expand(n, -1).flatten(start_dim=0, end_dim=1)
        )
        new_ptr = torch.nn.functional.pad(
            torch.cumsum(n_per_struct_repeated, dim=-1), (1, 0)
        )
        new_batch, _ = standardise_batch_and_ptr(None, new_ptr)

        return self.__class__(
            N_coords=N_coords_repeated,
            CA_coords=CA_coords_repeated,
            C_coords=C_coords_repeated,
            CB_coords=CB_coords_repeated,
            orientation_frames=frames_repeated,
            sequence=sequence_repeated,
            batch=new_batch,
            ptr=new_ptr,
        )

    @classmethod
    def from_frames(cls, frames: OrientationFrames):
        """
        Creates a structure from an OrientationFrames object, imputing CB coordinates using
        tetrahedral geometry around the CA atom and setting the sequence to all zeros (alanines).
        """
        N_coords, CA_coords, C_coords = frames.to_backbone_coords()
        CB_coords = impute_CB_coords(N_coords, CA_coords, C_coords)
        sequence = torch.zeros(
            (len(CA_coords)), dtype=torch.long, device=CA_coords.device
        )
        return cls(
            N_coords,
            CA_coords,
            C_coords,
            CB_coords,
            sequence=sequence,
            batch=frames.batch,
            ptr=frames.ptr,
        )


class LinearStructure(Structure):
    """
    Extends Structure to specifically work with linear peptide chains,
    adding functions for calculating chain-based structural features like dihedral angles.

    It is assumed that the coordinates within the coordinate tensor attributes
    appear in N-to-C order along the primary sequence of the protein.
    """

    def get_backbone_dihedrals(self) -> torch.Tensor:
        """
        Calculates backbone dihedral angles from atomic coordinates.
        Returns an (N, 3) tensor containing the three dihedral angles -
        phi, omega, and psi (in that order) - for each residue in the structure.
        """

        if self.missing_atoms:
            raise RuntimeError(
                "Structure is missing atoms, dihedral angle calculation failed"
            )

        # backbone dihedral calculation not possible for fewer than two residues
        if self.num_residues < 2:
            backbone_dihedrals_by_residue = torch.zeros(
                (self.CA_coords.shape[:-1] + (3,)), device=self.CA_coords.device
            )

        else:
            # get coordinates of backbone atoms
            backbone_coords = combine_coords(
                self.N_coords, self.CA_coords, self.C_coords
            )
            # get a tensor of all dihedral angles
            all_backbone_dihedrals = get_dihedral_angles(backbone_coords)

            psi, omega, phi = torch.split(
                all_backbone_dihedrals.reshape((-1, 3)), 1, dim=-1
            )

            backbone_dihedrals_by_residue = torch.cat(
                [
                    torch.nn.functional.pad(phi, (0, 0, 1, 0)),
                    torch.nn.functional.pad(omega, (0, 0, 1, 0)),
                    torch.nn.functional.pad(psi, (0, 0, 0, 1)),
                ],
                dim=-1,
            )

            # if batch vector is present, zero dihedrals representing the start/end of chains
            if self.has_batch:
                backbone_dihedrals_by_residue[self.ptr[:-1], :-1] = 0.0
                backbone_dihedrals_by_residue[self.ptr[1:] - 1, -1] = 0.0

        return backbone_dihedrals_by_residue
