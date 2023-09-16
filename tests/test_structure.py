"""
Tests the structure module, which contains the two classes Structure and LinearStructure
for storing geometric and sequence data of protein structures.
"""

from typing import Tuple

import pytest
import torch
from torch_scatter import scatter_mean

from sabdl.structure import (
    BondAngles,
    BondLengths,
    OrientationFrames,
    Structure,
    AminoAcid3,
)
from sabdl.utils import combine_coords


# this should be an even number
NUM_RESIDUES: int = 10


@pytest.fixture
def structure_data() -> (
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
):
    """
    Returns the data needed to create a structure, including:
        1. N coordinates
        2. CA coordinates
        3. C coordinates
        4. CB coordinates
        5. Sequence
        6. Batch
    """
    torch.manual_seed(123)
    N = torch.randn(NUM_RESIDUES, 3)
    CA = torch.randn(NUM_RESIDUES, 3)
    C = torch.randn(NUM_RESIDUES, 3)
    CB = torch.randn(NUM_RESIDUES, 3)
    sequence = torch.arange(NUM_RESIDUES)
    batch = torch.cat(
        [torch.zeros(NUM_RESIDUES // 2), torch.ones(NUM_RESIDUES // 2)]
    ).long()

    return N, CA, C, CB, sequence, batch


class TestOrientationFrames:
    """
    Tests the OrientationFrames class, which stores a set of orientation frames,
    i.e. a rotation and a translation for each frame.
    """

    def test_from_three_points(self, structure_data):
        """
        Tests the from_three_points() method, which creates orientation frames
        from three points.
        """
        N, CA, C, CB, sequence, batch = structure_data
        frames = OrientationFrames.from_three_points(N, CA, C)

        assert frames.rotations.shape == (
            CA.shape[0],
            3,
            3,
        ), f"Rotations should be a {CA.shape[0]} x 3 x 3 tensor"
        assert frames.translations.shape == (
            CA.shape[0],
            3,
        ), f"Translations should be a {CA.shape[0]} x 3 tensor"

        # check first column vector of rotation matrices is the unit-length vector from C to CA
        assert torch.allclose(
            frames.rotations[:, :, 0], torch.nn.functional.normalize(C - CA, dim=-1)
        ), "First column vector of rotation matrices should be the unit-length vector from C to CA"

        # check orthonormality of rotation matrices
        assert torch.allclose(
            torch.matmul(frames.rotations, frames.rotations.transpose(-2, -1)),
            torch.eye(3).unsqueeze(0).expand(frames.rotations.shape[0], -1, -1),
            atol=1e-5,
        ), "Rotation matrices should be orthonormal"

    def test_apply(self, structure_data):
        """
        Tests the apply() function, which applies the orientation frames as
        a transformation to a tensor of 3D points.
        """
        N, CA, C, CB, sequence, batch = structure_data
        frames = OrientationFrames.from_three_points(N, CA, C)

        # apply orientation frames to N and CA coordinates
        N_transformed = frames.apply(N)
        CA_transformed = frames.apply(CA)

        # check that N and CA distances remain the same after transformation
        assert torch.allclose(
            torch.linalg.norm(N_transformed - CA_transformed, dim=-1),
            torch.linalg.norm(N - CA, dim=-1),
            atol=1e-5,
        ), (
            "N and CA distances should remain the same after applying "
            "orientation frames (since the transformation is affine)"
        )

    def test_inverse(self, structure_data):
        """
        Tests the inverse() function, which returns the inverse transformation of each of
        the orientation frames, creating a new frames object whose apply() function undoes
        the apply() of the original frames.
        """
        N, CA, C, CB, sequence, batch = structure_data
        frames = OrientationFrames.from_three_points(N, CA, C)
        inv_frames = frames.inverse()

        assert isinstance(
            inv_frames, OrientationFrames
        ), "Inverse of orientation frames should be an OrientationFrames object"

        # apply orientation frames to CA coordinates
        CA_transformed = frames.apply(CA)

        # check that inverse of orientation frames returns original coordinates
        assert torch.allclose(
            inv_frames.apply(CA_transformed), CA, atol=1e-5
        ), "Inverse of orientation frames should return original coordinates"

    def test_compose(self, structure_data):
        """
        Tests the compose() function, which composes multiple sets of orientation frames
        to return a single new orientation frames object, whose apply() function is equivalent
        to applying the original frames in sequence.
        """
        N, CA, C, CB, sequence, batch = structure_data
        frames = OrientationFrames.from_three_points(N, CA, C)
        composed_frames = frames.compose(frames, frames)

        assert isinstance(
            composed_frames, OrientationFrames
        ), "Composed frames should be an OrientationFrames object"

        assert (
            composed_frames.rotations.shape == frames.rotations.shape
        ), "Composed frames should have the same number of rotations as the original frames"

        assert (
            composed_frames.translations.shape == frames.translations.shape
        ), "Composed frames should have the same number of translations as the original frames"

        # apply orientation frames three times to CA coordinates
        CA_transformed = frames.apply(frames.apply(frames.apply(CA)))

        # check that composed frames return the same coordinates as applying the original frames twice
        assert torch.allclose(
            composed_frames.apply(CA), CA_transformed, atol=1e-5
        ), "Composition of 2x frames should return the same coordinates as applying the original frames twice"

    def test_to_backbone_coords(self, structure_data):
        """
        Tests the to_backbone_coords() function, which converts a set of orientation frames
        into a set of backbone coordinates.
        """
        N, CA, C, CB, sequence, batch = structure_data
        frames = OrientationFrames.from_three_points(N, CA, C)
        bb_N, bb_CA, bb_C = frames.to_backbone_coords()

        assert (
            bb_N.shape == N.shape
        ), f"Output backbone N coordinates should match the input N coordinate shape"
        assert (
            bb_CA.shape == CA.shape
        ), f"Output backbone CA coordinates should match the input CA coordinate shape"
        assert (
            bb_C.shape == C.shape
        ), f"Output backbone C coordinates should match the input C coordinate shape"

        # check CA are the same as translations
        assert torch.allclose(
            bb_CA, frames.translations, atol=1e-5
        ), "CA coordinates should be the same as the translations"

        # check that backbone bond lengths and angles match literature values
        N_CA_lengths = torch.linalg.norm(bb_N - bb_CA, dim=-1)
        C_CA_lengths = torch.linalg.norm(bb_C - bb_CA, dim=-1)
        N_CA_C_angles = torch.acos(
            torch.sum((bb_N - bb_CA) * (bb_C - bb_CA), dim=-1)
            / (N_CA_lengths * C_CA_lengths)
        )

        assert torch.allclose(
            N_CA_lengths, torch.as_tensor(BondLengths["N_CA"].value)
        ), "N-CA bond lengths should match literature values"
        assert torch.allclose(
            C_CA_lengths, torch.as_tensor(BondLengths["CA_C"].value)
        ), "C-CA bond lengths should match literature values"
        assert torch.allclose(
            N_CA_C_angles, torch.as_tensor(BondAngles["N_CA_C"].value)
        ), "N-CA-C bond angles should match literature values"

    def test_translate(self, structure_data):
        """Tests the translate() function, which translates the frames by an input translation."""
        N, CA, C, CB, sequence, batch = structure_data
        frames = OrientationFrames.from_three_points(N, CA, C)

        # translate by a single vector and a whole tensor
        translation_1 = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        translation_2 = torch.randn_like(frames.translations)

        # check that the translation is applied correctly
        assert torch.allclose(
            frames.translate(translation_1).translations,
            frames.translations + translation_1,
        ), "Translated coordinates should match the original coordinates plus the translation vector"

        assert torch.allclose(
            frames.translate(translation_2).translations,
            frames.translations + translation_2,
        ), "Translated coordinates should match the original coordinates plus the translation vector"

    def test_center(self, structure_data):
        """Tests the center() function, which centers the frames."""
        N, CA, C, CB, sequence, batch = structure_data
        frames = OrientationFrames.from_three_points(N, CA, C)
        frames_batched = OrientationFrames.from_three_points(N, CA, C, batch=batch)

        centered_frames, centroid = frames.center()
        centered_frames_batched, centroid_batched = frames_batched.center()

        # check that the calculated centroids are correct
        assert torch.allclose(
            centroid, torch.mean(frames.translations, dim=0, keepdim=True)
        ), "Centroid should be the mean of the translations"
        assert torch.allclose(
            centroid_batched, scatter_mean(frames_batched.translations, batch, dim=0)
        ), "Centroid should be the mean of the translations"

        # check that the centering is applied correctly
        assert torch.allclose(
            centered_frames.translations.mean(dim=0), torch.zeros(3), atol=1e-5
        ), "Centered frames should have zero mean"
        batched_centroids = scatter_mean(
            centered_frames_batched.translations, batch, dim=0
        )
        assert torch.allclose(
            batched_centroids, torch.zeros_like(batched_centroids), atol=1e-5
        ), "Centered frames should have zero mean"


class TestStructure:
    """
    Tests the Structure class, which stores atomic coordinates and sequence information,
    with the option to store batch information.
    """

    def test_init(self, structure_data):
        N, CA, C, CB, sequence, batch = structure_data
        structure = Structure(N, CA, C, CB, sequence)
        missing_atoms_structure = Structure(N, CA[:-1], C, CB, sequence)
        batched_structure = Structure(N, CA, C, CB, sequence, batch=batch)

        assert torch.allclose(
            structure.CB_coords[sequence == AminoAcid3["GLY"].value],
            structure.CA_coords[sequence == AminoAcid3["GLY"].value],
        ), "CB coordinates for GLY should be the same as CA coordinates"

        assert torch.allclose(
            batched_structure.CB_coords[sequence == AminoAcid3["GLY"].value],
            batched_structure.CA_coords[sequence == AminoAcid3["GLY"].value],
        ), "CB coordinates for GLY should be the same as CA coordinates"

        assert (
            missing_atoms_structure.missing_atoms
        ), "Structure should have missing atoms property as True"
        assert (
            not missing_atoms_structure.has_orientation_frames
        ), "Structure with missing atoms should not have orientation frames"

        assert hasattr(
            structure, "N_coords"
        ), "Structure should have N_coords attribute"
        assert hasattr(
            structure, "CA_coords"
        ), "Structure should have CA_coords attribute"
        assert hasattr(
            structure, "C_coords"
        ), "Structure should have C_coords attribute"
        assert hasattr(
            structure, "CB_coords"
        ), "Structure should have CB_coords attribute"
        assert hasattr(
            structure, "sequence"
        ), "Structure should have sequence attribute"
        assert (
            structure.has_orientation_frames
        ), "Structure should have orientation frames"
        assert not structure.has_batch, "Structure should not have batch attribute"

        assert hasattr(
            batched_structure, "N_coords"
        ), "Structure should have N_coords attribute"
        assert hasattr(
            batched_structure, "CA_coords"
        ), "Structure should have CA_coords attribute"
        assert hasattr(
            batched_structure, "C_coords"
        ), "Structure should have C_coords attribute"
        assert hasattr(
            batched_structure, "CB_coords"
        ), "Structure should have CB_coords attribute"
        assert hasattr(
            batched_structure, "sequence"
        ), "Structure should have sequence attribute"
        assert hasattr(
            batched_structure, "batch"
        ), "Structure should have batch attribute"
        assert hasattr(batched_structure, "ptr"), "Structure should have ptr attribute"

        assert torch.all(
            batched_structure.ptr == torch.as_tensor([0, 5, 10])
        ).item(), "Ptr attribute should be [0, 5, 10] for the input batch tensor"

        assert (
            batched_structure.has_orientation_frames
        ), "Structure should have orientation frames"
        assert batched_structure.has_batch, "Structure should have batch"

    def test_len(self, structure_data):
        N, CA, C, CB, sequence, batch = structure_data
        structure = Structure(N, CA, C, CB, sequence, batch=batch)

        assert len(structure) == len(
            CA
        ), "Structure should have the same length as CA coordinates"

        assert (
            len(structure) == structure.num_residues
        ), "Structure should have the same length as num_residues"

    def test_getitem(self, structure_data):
        N, CA, C, CB, sequence, batch = structure_data
        structure = Structure(N, CA, C, CB, sequence, batch=batch)

        assert isinstance(
            structure[0], Structure
        ), "Structure[0] should return a Structure object"

        # test integer indexing
        for i in range(len(structure)):
            assert torch.allclose(
                structure[i].N_coords, N[i]
            ), "N coords of indexed structure should match indexed N coords"
            assert torch.allclose(
                structure[i].CA_coords, CA[i]
            ), "CA coords of indexed structure should match indexed CA coords"
            assert torch.allclose(
                structure[i].C_coords, C[i]
            ), "C coords of indexed structure should match indexed C coords"
            assert torch.allclose(
                structure[i].CB_coords, CB[i]
            ), "CB coords of indexed structure should match indexed CB coords"
            assert torch.allclose(
                structure[i].sequence, sequence[i]
            ), "Sequence of indexed structure should match indexed sequence"
            assert torch.allclose(
                structure[i].batch, batch[i]
            ), "Batch of indexed structure should match indexed batch"

        # test tensor/slice indexing
        for i in range(len(structure) - 5):
            # test slice indexing
            idx = slice(i, i + 4)
            assert torch.allclose(
                structure[idx].N_coords, N[idx]
            ), "N coords of indexed structure should match indexed N coords"
            assert torch.allclose(
                structure[idx].CA_coords, CA[idx]
            ), "CA coords of indexed structure should match indexed CA coords"
            assert torch.allclose(
                structure[idx].C_coords, C[idx]
            ), "C coords of indexed structure should match indexed C coords"
            assert torch.allclose(
                structure[idx].CB_coords, CB[idx]
            ), "CB coords of indexed structure should match indexed CB coords"
            assert torch.allclose(
                structure[idx].sequence, sequence[idx]
            ), "Sequence of indexed structure should match indexed sequence"
            assert torch.allclose(
                structure[idx].batch, batch[idx]
            ), "Batch of indexed structure should match indexed batch"

            # test tensor indexing
            idx = torch.arange(i, i + 4, dtype=torch.long)
            assert torch.allclose(
                structure[idx].N_coords, N[idx]
            ), "N coords of indexed structure should match indexed N coords"
            assert torch.allclose(
                structure[idx].CA_coords, CA[idx]
            ), "CA coords of indexed structure should match indexed CA coords"
            assert torch.allclose(
                structure[idx].C_coords, C[idx]
            ), "C coords of indexed structure should match indexed C coords"
            assert torch.allclose(
                structure[idx].CB_coords, CB[idx]
            ), "CB coords of indexed structure should match indexed CB coords"
            assert torch.allclose(
                structure[idx].sequence, sequence[idx]
            ), "Sequence of indexed structure should match indexed sequence"
            assert torch.allclose(
                structure[idx].batch, batch[idx]
            ), "Batch of indexed structure should match indexed batch"

    def test_combine(self, structure_data):
        """
        Tests the combine() function, which combines a list of structures into a single structure
        and optionally adds a batch attribute to identify the original structure indices.
        """
        N, CA, C, CB, sequence, batch = structure_data
        batch_size = 5
        structures = [Structure(N, CA, C, CB, sequence) for _ in range(batch_size)]
        batched_structure = Structure.combine(structures)
        unbatched_structure = Structure.combine(structures, add_batch=False)

        assert torch.allclose(
            batched_structure.N_coords,
            torch.cat([s.N_coords for s in structures], dim=0),
        ), "N coords of batched structure should match concatenated N coords"
        assert torch.allclose(
            batched_structure.CA_coords,
            torch.cat([s.CA_coords for s in structures], dim=0),
        ), "CA coords of batched structure should match concatenated CA coords"
        assert torch.allclose(
            batched_structure.C_coords,
            torch.cat([s.C_coords for s in structures], dim=0),
        ), "C coords of batched structure should match concatenated C coords"
        assert torch.allclose(
            batched_structure.CB_coords,
            torch.cat([s.CB_coords for s in structures], dim=0),
        ), "CB coords of batched structure should match concatenated CB coords"
        assert torch.allclose(
            batched_structure.sequence,
            torch.cat([s.sequence for s in structures], dim=0),
        ), "Sequence of batched structure should match concatenated sequence"

        assert batched_structure.has_batch, "Batched structure should have batch"
        assert hasattr(
            batched_structure, "ptr"
        ), "Batched structure should have ptr attribute"

        assert torch.allclose(
            batched_structure.batch,
            torch.arange(batch_size).repeat_interleave(CA.shape[0]).long(),
        ), "Batch of batched structure should be [0, 0, ..., 1, 1, ...]"

        assert (
            unbatched_structure.has_batch is False
        ), "Unbatched structure should not have batch"

    def test_split(self, structure_data):
        """
        Tests the split() function, which converts from a
        batched structure into a list of individual structures.
        """
        N, CA, C, CB, sequence, batch = structure_data
        batch_size = 5
        structures = [Structure(N, CA, C, CB, sequence) for _ in range(batch_size)]
        batched_structure = Structure.combine(structures)

        split_structures = batched_structure.split()

        assert (
            len(split_structures) == batch_size
        ), "Split structures should have length equal to batch size"

        for i in range(batch_size):
            assert not split_structures[
                i
            ].has_batch, "Split structure should not have batch"

            assert torch.allclose(
                split_structures[i].N_coords, structures[i].N_coords
            ), "N coords of split structure should match original N coords"
            assert torch.allclose(
                split_structures[i].CA_coords, structures[i].CA_coords
            ), "CA coords of split structure should match original CA coords"
            assert torch.allclose(
                split_structures[i].C_coords, structures[i].C_coords
            ), "C coords of split structure should match original C coords"
            assert torch.allclose(
                split_structures[i].CB_coords, structures[i].CB_coords
            ), "CB coords of split structure should match original CB coords"

    def test_translate(self, structure_data):
        """
        Tests the translate() function, which translates a structure's coordinates.
        """

        N, CA, C, CB, sequence, batch = structure_data
        structure = Structure(N, CA, C, CB, sequence, batch=batch)

        # test with both a constant translation and a variable per-residue translation
        torch.manual_seed(123)
        translation_1 = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        translation_2 = torch.randn((structure.num_residues, 3))

        structure_translated_1 = structure.translate(translation_1)

        assert torch.allclose(
            structure_translated_1.N_coords,
            structure.N_coords + translation_1,
        ), "N coords of translated structure should match original N coords + translation"
        assert torch.allclose(
            structure_translated_1.CA_coords,
            structure.CA_coords + translation_1,
        ), "CA coords of translated structure should match original CA coords + translation"
        assert torch.allclose(
            structure_translated_1.C_coords,
            structure.C_coords + translation_1,
        ), "C coords of translated structure should match original C coords + translation"
        assert torch.allclose(
            structure_translated_1.CB_coords,
            structure.CB_coords + translation_1,
        ), "CB coords of translated structure should match original CB coords + translation"
        assert torch.allclose(
            structure_translated_1.sequence,
            structure.sequence,
        ), "Sequence of translated structure should match original sequence"

        structure_translated_2 = structure.translate(translation_2)

        assert torch.allclose(
            structure_translated_2.N_coords,
            structure.N_coords + translation_2,
        ), "N coords of translated structure should match original N coords + translation"
        assert torch.allclose(
            structure_translated_2.CA_coords,
            structure.CA_coords + translation_2,
        ), "CA coords of translated structure should match original CA coords + translation"
        assert torch.allclose(
            structure_translated_2.C_coords,
            structure.C_coords + translation_2,
        ), "C coords of translated structure should match original C coords + translation"
        assert torch.allclose(
            structure_translated_2.CB_coords,
            structure.CB_coords + translation_2,
        ), "CB coords of translated structure should match original CB coords + translation"
        assert torch.allclose(
            structure_translated_2.sequence,
            structure.sequence,
        ), "Sequence of translated structure should match original sequence"

    def test_center(self, structure_data):
        """Tests the center() function, which centers a structure's coordinates."""
        N, CA, C, CB, sequence, batch = structure_data
        batch_expanded = torch.repeat_interleave(batch, 4)

        # test both with and without batch
        structure = Structure(N, CA, C, CB, sequence)
        structure_batched = Structure(N, CA, C, CB, sequence, batch=batch)

        orig_centroid = torch.mean(
            combine_coords(
                structure.N_coords,
                structure.CA_coords,
                structure.C_coords,
                structure.CB_coords,
            ),
            dim=0,
            keepdim=True,
        )
        orig_batch_centroids = scatter_mean(
            combine_coords(
                structure_batched.N_coords,
                structure_batched.CA_coords,
                structure_batched.C_coords,
                structure_batched.CB_coords,
            ),
            batch_expanded,
            dim=0,
        )

        centered_structure, centroid = structure.center()

        # test with return_centre=False
        centered_structure = structure_batched.center(return_centre=False)

        assert torch.allclose(
            centroid, orig_centroid, atol=1e-5
        ), "Centroid output should match original centroid"

        all_centered_structure_coords = combine_coords(
            centered_structure.N_coords,
            centered_structure.CA_coords,
            centered_structure.C_coords,
            centered_structure.CB_coords,
        )
        new_centroid = torch.mean(all_centered_structure_coords, dim=0)

        assert torch.allclose(
            new_centroid, torch.zeros_like(new_centroid), atol=1e-5
        ), "Centred structure should have zero mean coordinates"

        centered_structure_batched, batch_centroids = structure_batched.center()

        # test with return_centre=False
        centered_structure_batched = structure_batched.center(return_centre=False)

        assert torch.allclose(
            batch_centroids, orig_batch_centroids, atol=1e-5
        ), "Batch centroids output should match original batch centroids"

        all_centered_structure_coords_batched = combine_coords(
            centered_structure_batched.N_coords,
            centered_structure_batched.CA_coords,
            centered_structure_batched.C_coords,
            centered_structure_batched.CB_coords,
        )
        centroids = scatter_mean(
            all_centered_structure_coords_batched, batch_expanded, dim=0
        )

        assert torch.allclose(
            centroids, torch.zeros_like(centroids), atol=1e-5
        ), "Centred structure should have zero mean coordinates"

    def test_align_to_pcs(self, structure_data):
        N, CA, C, CB, sequence, batch = structure_data

        N_1 = N[batch == 0]
        CA_1 = CA[batch == 0]
        C_1 = C[batch == 0]
        CB_1 = CB[batch == 0]
        sequence_1 = sequence[batch == 0]

        N_2 = N[batch == 1]
        CA_2 = CA[batch == 1]
        C_2 = C[batch == 1]
        CB_2 = CB[batch == 1]
        sequence_2 = sequence[batch == 1]

        structure_1 = Structure(N_1, CA_1, C_1, CB_1, sequence_1)
        structure_2 = Structure(N_2, CA_2, C_2, CB_2, sequence_2)

        structure_1 = structure_1.center(return_centre=False)
        structure_2 = structure_2.center(return_centre=False)

        structure_1_aligned = structure_1.align_to_pcs(structure_2)

        structure_1_aligned_coords = combine_coords(
            structure_1_aligned.N_coords,
            structure_1_aligned.CA_coords,
            structure_1_aligned.C_coords,
            structure_1_aligned.CB_coords,
        )
        structure_2_coords = combine_coords(
            structure_2.N_coords,
            structure_2.CA_coords,
            structure_2.C_coords,
            structure_2.CB_coords,
        )

        _, _, structure_1_aligned_pcs = torch.pca_lowrank(structure_1_aligned_coords)
        _, _, structure_2_pcs = torch.pca_lowrank(structure_2_coords)

        assert torch.allclose(
            structure_1_aligned_pcs, structure_2_pcs, atol=1e-5
        ) or torch.allclose(
            -structure_1_aligned_pcs, structure_2_pcs, atol=1e-5
        ), "Aligned structure should have the same principal components as the target structure"

    def test_scramble_sequence(self, structure_data):
        """Tests the scramble_sequence() function, which scrambles the sequence of a structure."""

        N, CA, C, CB, sequence, _ = structure_data
        structure = Structure(N, CA, C, CB, sequence)

        torch.manual_seed(123)
        structure_scrambled = structure.scramble_sequence()

        assert not torch.allclose(
            structure_scrambled.sequence, structure.sequence
        ), "Scrambled sequence should be different to original sequence"

        scrambled_gly_mask = structure_scrambled.sequence == AminoAcid3["GLY"].value
        assert torch.allclose(
            structure_scrambled.CA_coords[scrambled_gly_mask],
            structure_scrambled.CB_coords[scrambled_gly_mask],
        ), "Scrambled CB coordinates for GLY should be the same as CA coordinates"

        assert not torch.allclose(
            structure_scrambled.CA_coords[~scrambled_gly_mask],
            structure_scrambled.CB_coords[~scrambled_gly_mask],
        ), "Scrambled CB coordinates for non-GLY should be different to CA coordinates"

    def test_repeat(self, structure_data):
        """Tests the repeat() function, which repeats a structure n times."""
        N, CA, C, CB, sequence, batch = structure_data
        structure = Structure(N, CA, C, CB, sequence, batch=batch)

        n = 3
        structure_repeated = structure.repeat(n)

        assert (
            len(structure_repeated) == len(structure) * n
        ), "Repeated structure should have length equal to original structure * n"

        for idx in range(0, len(structure) * n, len(structure)):
            assert torch.allclose(
                structure_repeated.N_coords[idx : (idx + len(structure))],
                structure.N_coords,
            ), "N coords of each repeat of the structure should match original N coords"
            assert torch.allclose(
                structure_repeated.CA_coords[idx : (idx + len(structure))],
                structure.CA_coords,
            ), "CA coords of each repeat of the structure should match original CA coords"
            assert torch.allclose(
                structure_repeated.C_coords[idx : (idx + len(structure))],
                structure.C_coords,
            ), "C coords of each repeat of the structure should match original C coords"
            assert torch.allclose(
                structure_repeated.CB_coords[idx : (idx + len(structure))],
                structure.CB_coords,
            ), "CB coords of each repeat of the structure should match original CB coords"
            assert torch.allclose(
                structure_repeated.sequence[idx : (idx + len(structure))],
                structure.sequence,
            ), "Sequence of each repeat of the structure should match original sequence"

        # test that the new batch and ptr attributes are correct
        num_batches = torch.unique(batch).shape[0]
        num_repeat_batches = n * num_batches
        assert (
            torch.max(structure_repeated.batch) == num_repeat_batches - 1
        ), "Max batch of repeated structure should be (n * number of original batches) - 1"

        assert torch.all(
            torch.isin(
                torch.unique(structure_repeated.batch), torch.arange(num_repeat_batches)
            )
        ).item(), (
            "Repeated structure does not have all batch "
            "values between 0 and (n * number of original batches) - 1"
        )

        n_per_struct = torch.bincount(structure_repeated.batch)
        assert torch.allclose(
            n_per_struct, torch.diff(structure_repeated.ptr)
        ), "Batch and ptr of repeated structure should match"

        orig_n_per_struct = torch.diff(structure.ptr)
        for idx in range(0, len(orig_n_per_struct) * n, len(orig_n_per_struct)):
            assert torch.allclose(
                n_per_struct[idx : (idx + len(orig_n_per_struct))],
                orig_n_per_struct,
            ), "Batch and ptr of each repeat of the structure should match the original structure"
