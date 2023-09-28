"""
Metrics for evaluating models.
"""

from typing import Optional, Tuple

import torch
import einops
from torch_scatter import scatter_sum, scatter_mean

from ..structure import (
    Structure,
    LinearStructure,
    BondLengths,
    BondAngles,
    BondAngleStdDevs,
    BondLengthStdDevs,
    BondAngleCosineStdDevs,
    AtomVanDerWaalRadii,
)

from ..utils import get_distance_matrix

# these are calculated using an analytical solution assuming a normal distribution of angles
# with the given mean and standard deviation
N_CA_C_ANGLE_COS_MEAN = torch.exp(
    torch.as_tensor(BondAngleStdDevs["N_CA_C"].value ** 2) / 2
) * torch.cos(torch.as_tensor(BondAngles["N_CA_C"].value))
CA_C_N_ANGLE_COS_MEAN = torch.exp(
    torch.as_tensor(BondAngleStdDevs["CA_C_N"].value ** 2) / 2
) * torch.cos(torch.as_tensor(BondAngles["CA_C_N"].value))
C_N_CA_ANGLE_COS_MEAN = torch.exp(
    torch.as_tensor(BondAngleStdDevs["C_N_CA"].value ** 2) / 2
) * torch.cos(torch.as_tensor(BondAngles["C_N_CA"].value))

N_CA_C_ANGLE_COS_STD = torch.as_tensor(BondAngleCosineStdDevs["N_CA_C"].value)
CA_C_N_ANGLE_COS_STD = torch.as_tensor(BondAngleCosineStdDevs["CA_C_N"].value)
C_N_CA_ANGLE_COS_STD = torch.as_tensor(BondAngleCosineStdDevs["C_N_CA"].value)


def get_clash_loss(structure: LinearStructure) -> torch.Tensor:
    """
    Calculates loss penalising steric clashes, calculated using Van Der Waals radii.
    Specifically any non-covalently bonded atoms whose Van der Waals radii overlap
    are deemed a clash. Implemented exactly the same way as in AlphaFold2
    (https://www.nature.com/articles/s41586-021-03819-2).

    :param structure: The LinearStructure with N residues for which the clash loss will be calculated.
        The structure must be linear so that the covalent bond structure can be inferred.
    :returns: Tensor of shape (N,) containing the clash loss for each residue.
    """
    N_N_lit_dist = 2.0 * AtomVanDerWaalRadii["N"].value
    C_C_lit_dist = 2.0 * AtomVanDerWaalRadii["C"].value
    C_N_lit_dist = AtomVanDerWaalRadii["C"].value + AtomVanDerWaalRadii["N"].value

    N_dist = get_distance_matrix(
        structure.N_coords, batch=structure.batch, pad_value=torch.inf
    )
    CA_dist = get_distance_matrix(
        structure.CA_coords, batch=structure.batch, pad_value=torch.inf
    )
    C_dist = get_distance_matrix(
        structure.C_coords, batch=structure.batch, pad_value=torch.inf
    )

    N_CA_dist = get_distance_matrix(
        structure.N_coords,
        structure.CA_coords,
        batch=structure.batch,
        pad_value=torch.inf,
    )
    N_C_dist = get_distance_matrix(
        structure.N_coords,
        structure.C_coords.roll(1, dims=-2),
        batch=structure.batch,
        pad_value=torch.inf,
    )
    CA_C_dist = get_distance_matrix(
        structure.CA_coords,
        structure.C_coords,
        batch=structure.batch,
        pad_value=torch.inf,
    )

    # fill diagonals so that covalently bonded atoms are not penalised for being within VDW radius
    diag_mask_matrix = (
        torch.eye(N_dist.shape[-2], N_dist.shape[-2], device=structure.CA_coords.device)
        .bool()
        .expand(N_dist.shape[:-2] + (-1, -1))
    )

    pad_mask = torch.all(CA_dist == torch.inf, dim=-1)
    N_dist = torch.where(diag_mask_matrix, torch.inf, N_dist)
    CA_dist = torch.where(diag_mask_matrix, torch.inf, CA_dist)
    C_dist = torch.where(diag_mask_matrix, torch.inf, C_dist)
    N_CA_dist = torch.where(diag_mask_matrix, torch.inf, N_CA_dist)
    N_C_dist = torch.where(diag_mask_matrix, torch.inf, N_C_dist)
    CA_C_dist = torch.where(diag_mask_matrix, torch.inf, CA_C_dist)

    N_clash_loss = torch.clamp_min(N_N_lit_dist - 1.5 - N_dist, 0.0) / 2
    CA_clash_loss = torch.clamp_min(C_C_lit_dist - 1.5 - CA_dist, 0.0) / 2
    C_clash_loss = torch.clamp_min(C_C_lit_dist - 1.5 - C_dist, 0.0) / 2
    N_CA_clash_loss = torch.clamp_min(C_N_lit_dist - 1.5 - N_CA_dist, 0.0)
    N_C_clash_loss = torch.clamp_min(C_N_lit_dist - 1.5 - N_C_dist, 0.0)
    CA_C_clash_loss = torch.clamp_min(C_C_lit_dist - 1.5 - CA_C_dist, 0.0)

    total_class_loss = torch.sum(
        N_clash_loss
        + CA_clash_loss
        + C_clash_loss
        + N_CA_clash_loss
        + N_C_clash_loss
        + CA_C_clash_loss,
        dim=-1,
    )

    total_class_loss = total_class_loss[~pad_mask]

    return total_class_loss


def get_bond_angle_loss(structure: LinearStructure, num_stds: int = 12) -> torch.Tensor:
    """
    Calculates a bond angle loss term, depending on deviations
    between predicted backbone bond angles and their literature values.
    Specifically this uses a flat-bottomed loss that only takes values >0
    if the cosine of the bond angle is outside of the mean +/- num_stds * std.
    Implemented the same way as in AlphaFold2 (https://www.nature.com/articles/s41586-021-03819-2).

    :param structure: The LinearStructure with N residues to calculate the bond angle loss for.
    :param num_stds: The number of standard deviations to use for the flat-bottomed loss. Default
        is 12, which is the value used in AlphaFold2.
    :returns: Tensor of shape (N,) containing the bond angle loss for each residue.
    """
    N_CA_vectors = torch.nn.functional.normalize(
        structure.N_coords - structure.CA_coords
    )
    C_CA_vectors = torch.nn.functional.normalize(
        structure.C_coords - structure.CA_coords
    )
    # Roll the N coords, so that the coords are lined up correctly, and then cut the last element
    C_N_vectors = torch.nn.functional.normalize(
        structure.C_coords - structure.N_coords.roll(-1, dims=-2)
    )

    cos_N_CA_C_bond_angles = torch.sum(N_CA_vectors * C_CA_vectors, dim=-1)
    cos_CA_C_N_bond_angles = torch.sum(C_CA_vectors * C_N_vectors, dim=-1)
    cos_C_N_CA_bond_angles = torch.sum(
        C_N_vectors * -N_CA_vectors.roll(-1, dims=-2), dim=-1
    )

    if structure.has_batch:
        cos_CA_C_N_bond_angles = cos_CA_C_N_bond_angles.index_fill(
            0, structure.ptr[1:] - 1, CA_C_N_ANGLE_COS_MEAN
        )
        cos_C_N_CA_bond_angles = cos_C_N_CA_bond_angles.index_fill(
            0, structure.ptr[1:] - 1, C_N_CA_ANGLE_COS_MEAN
        )
    else:
        cos_CA_C_N_bond_angles[-1] = CA_C_N_ANGLE_COS_MEAN
        cos_C_N_CA_bond_angles[-1] = C_N_CA_ANGLE_COS_MEAN

    N_CA_C_angle_loss = (
        torch.clamp_min(
            torch.abs(cos_N_CA_C_bond_angles - N_CA_C_ANGLE_COS_MEAN)
            - num_stds * N_CA_C_ANGLE_COS_STD,
            0.0,
        )
        / cos_N_CA_C_bond_angles.shape[-1]
    )

    CA_C_N_angle_loss = (
        torch.clamp_min(
            torch.abs(cos_CA_C_N_bond_angles - CA_C_N_ANGLE_COS_MEAN)
            - num_stds * CA_C_N_ANGLE_COS_STD,
            0.0,
        )
        / cos_CA_C_N_bond_angles.shape[-1]
    )

    C_N_CA_angle_loss = (
        torch.clamp_min(
            torch.abs(cos_C_N_CA_bond_angles - C_N_CA_ANGLE_COS_MEAN)
            - num_stds * C_N_CA_ANGLE_COS_STD,
            0.0,
        )
        / cos_C_N_CA_bond_angles.shape[-1]
    )

    return N_CA_C_angle_loss + CA_C_N_angle_loss + C_N_CA_angle_loss


def get_bond_length_loss(
    structure: LinearStructure, num_stds: int = 12
) -> torch.Tensor:
    """
    Calculates a bond length loss term, depending on deviations
    between predicted backbone bond lengths and their literature values.
    Specifically this uses a flat-bottomed loss that only takes values >0
    if the bond length is outside of the mean +/- num_stds * std.
    Implemented the same way as in AlphaFold2 (https://www.nature.com/articles/s41586-021-03819-2).

    :param structure: The LinearStructure with N residues to calculate the bond angle loss for.
    :param num_stds: The number of standard deviations to use for the flat-bottomed loss. Default
        is 12, which is the value used in AlphaFold2.
    :returns: Tensor of shape (N,) containing the bond angle loss for each residue.
    """
    N_CA_bond_lengths = torch.linalg.norm(
        structure.N_coords - structure.CA_coords, dim=-1
    )
    CA_C_bond_lengths = torch.linalg.norm(
        structure.CA_coords - structure.C_coords, dim=-1
    )
    # Roll the N coords, so that the coords are lined up correctly, and then cut the last element
    C_N_bond_lengths = torch.linalg.norm(
        structure.C_coords - structure.N_coords.roll(-1, dims=-2),
        dim=-1,
    )

    if structure.has_batch:
        C_N_bond_lengths = C_N_bond_lengths.index_fill(
            0, structure.ptr[1:] - 1, BondLengths["C_N"].value
        )
    else:
        C_N_bond_lengths[-1] = BondLengths["C_N"].value

    N_CA_length_loss = (
        torch.clamp_min(
            torch.abs(N_CA_bond_lengths - BondLengths["N_CA"].value)
            - num_stds * BondLengthStdDevs["N_CA"].value,
            0.0,
        )
        / N_CA_bond_lengths.shape[0]
    )

    CA_C_length_loss = (
        torch.clamp_min(
            torch.abs(CA_C_bond_lengths - BondLengths["CA_C"].value)
            - num_stds * BondLengthStdDevs["CA_C"].value,
            0.0,
        )
        / CA_C_bond_lengths.shape[0]
    )

    C_N_length_loss = (
        torch.clamp_min(
            torch.abs(C_N_bond_lengths - BondLengths["C_N"].value)
            - num_stds * BondLengthStdDevs["C_N"].value,
            0.0,
        )
        / C_N_bond_lengths.shape[0]
    )

    return N_CA_length_loss + CA_C_length_loss + C_N_length_loss


def get_violations(
    structure: LinearStructure,
) -> torch.Tensor:
    """
    Identifies whether each structure in the batch has any structural violation,
    i.e. non-zero values for the bond length, bond angle, and clash loss terms.
    Returns a binary float tensor with a 1 for each structure with a violation,
    and a 0 for each structure without a violation.

    :param structure: A LinearStructure object of N residues containing the predicted coordinates.
    :returns: Tensor of shape (N,) containing a 1 for each structure with a violation, and a 0 for each
        structure without a violation.
    """
    if structure.has_batch:
        batch = structure.batch
    else:
        batch = torch.zeros(
            len(structure), device=structure.CA_coords.device, dtype=torch.long
        )

    bond_len_loss = get_bond_length_loss(structure)
    bond_ang_loss = get_bond_angle_loss(structure)
    cl_loss = get_clash_loss(structure)

    loss_per_structure = scatter_sum(
        bond_len_loss + bond_ang_loss + cl_loss,
        batch,
        dim=0,
    )

    violations = (loss_per_structure > 0).float()

    return violations


def get_epitope_cdr_clashes(
    cdr: Structure, epitope: Structure, threshold: float = 3.5
) -> torch.Tensor:
    """
    Returns a binary tensor indicating whether each CDR residue clashes with
    any epitope residue with the same batch assignment.
    """
    if not cdr.has_batch == epitope.has_batch:
        raise ValueError(
            "CDR and epitope must either both have batches or neither have batches."
        )

    cdr_epitope_distances = get_distance_matrix(
        cdr.CA_coords,
        epitope.CA_coords,
        batch=cdr.batch,
        other_batch=epitope.batch,
        pad_value=torch.inf,
    )

    clashing = torch.any(
        cdr_epitope_distances.flatten(end_dim=1) < threshold, dim=-1
    ).float()

    return clashing


def get_rmsd(
    coords_1: torch.Tensor,
    coords_2: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    dim: int = 0,
) -> torch.Tensor:
    """
    Calculates root mean squared deviation for two tensors of coordinates. Note
    that this does not perform the typical centering or Kabsch rotation to align
    the two coordinates before calculating RMSD.

    :param coords_1: One tensor of coordinates.
    :param coords_2: Another tensor of coordinates.
    :param batch: Optional batch tensor specifying the batch to which each coordinate belongs.
        If this is provided and reduce is set to True, the returned Tensor will have multiple
        elements, i.e. one RMSD for each batch element (default: None).
    :param dim: Dimension over which the mean squared deviation is calculated (default: 0).
    :return: Float tensor containing the RMSD value(s).
    """
    sq_distances = torch.sum((coords_1 - coords_2) ** 2, dim=-1)

    if batch is not None:
        mean_sq_distance = scatter_mean(sq_distances, batch, dim=dim)
    else:
        mean_sq_distance = torch.mean(sq_distances, dim=dim)

    root_mean_sq_dev = torch.sqrt(mean_sq_distance)

    return root_mean_sq_dev


def mean_pairwise_rmsd(
    pred_coords: list[torch.Tensor], other_coords: list[torch.Tensor]
):
    """
    Gets the mean of all pairwise RMSDs between two sets of
    predicted coordinates. They must be the same length.
    """

    coords_stacked = torch.stack(pred_coords)
    coords_stacked -= torch.mean(coords_stacked, dim=1, keepdim=True)
    other_coords_stacked = torch.stack(other_coords)
    other_coords_stacked -= torch.mean(other_coords_stacked, dim=1, keepdim=True)

    coords_1 = einops.rearrange(coords_stacked, "b r d -> 1 b r d")
    coords_2 = einops.rearrange(other_coords_stacked, "b r d -> b 1 r d")
    rmsds = torch.sum((coords_1 - coords_2) ** 2, dim=-1).mean(dim=-1).sqrt()

    # if the two sets of coordinates are the same, distance matrix will contain repeats so
    # take upper triangular
    if torch.allclose(coords_stacked, other_coords_stacked):
        idx1, idx2 = torch.triu_indices(rmsds.shape[0], rmsds.shape[1], offset=1)
        rmsds = rmsds[idx1, idx2]

    return torch.mean(rmsds)


def pca(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a PCA on a set of coordinates. Returns the principal components
    and their associated eigenvalues.

    :param coords: Tensor of shape (N, 3) containing the coordinates to perform PCA on.
    :returns: Tuple of tensors containing the principal components and their associated eigenvalues.
    """
    U, S, V = torch.pca_lowrank(coords)
    return V, torch.diag(S)
