""" Generic loss functions that can be used across model. """

from typing import Tuple, Union
import torch
from torch import nn
from torch_scatter import scatter_sum

from ..structure import (
    LinearStructure,
    BondLengths,
    BondAngles,
    BondAngleStdDevs,
    BondLengthStdDevs,
    BondAngleCosineStdDevs,
    AtomVanDerWaalRadii,
)

from ..utils import get_distance_matrix, combine_coords


class StructureLoss(nn.Module):
    """
    Loss function between two structures, consisting of an MSE loss between coordinates,
    violation loss terms for deviations from idealised bond lengths and angles, and a
    clash term for steric clashes. The violation and clash losses are taken from AlphaFold2
    (https://www.nature.com/articles/s41586-021-03819-2).

    Note this should always be used with a model with SE(3)-equivariant coordinate updates,
    since the MSE metric is not invariant to roto-translations of the underlying structure.
    """

    def __init__(self, num_bond_angle_stds: int = 12, num_bond_len_stds: int = 12):
        super().__init__()

        self._num_bond_angle_stds = num_bond_angle_stds
        self._num_bond_len_stds = num_bond_len_stds

        N_CA_C_angle_cos_mean = torch.exp(
            torch.as_tensor(BondAngleStdDevs["N_CA_C"].value ** 2) / 2
        ) * torch.cos(torch.as_tensor(BondAngles["N_CA_C"].value))
        CA_C_N_angle_cos_mean = torch.exp(
            torch.as_tensor(BondAngleStdDevs["CA_C_N"].value ** 2) / 2
        ) * torch.cos(torch.as_tensor(BondAngles["CA_C_N"].value))
        C_N_CA_angle_cos_mean = torch.exp(
            torch.as_tensor(BondAngleStdDevs["C_N_CA"].value ** 2) / 2
        ) * torch.cos(torch.as_tensor(BondAngles["C_N_CA"].value))

        self.register_buffer("N_CA_C_angle_cos", N_CA_C_angle_cos_mean)
        self.register_buffer("CA_C_N_angle_cos", CA_C_N_angle_cos_mean)
        self.register_buffer("C_N_CA_angle_cos", C_N_CA_angle_cos_mean)

        self.register_buffer(
            "N_CA_C_angle_cos_std",
            torch.as_tensor(BondAngleCosineStdDevs["N_CA_C"].value),
        )
        self.register_buffer(
            "CA_C_N_angle_cos_std",
            torch.as_tensor(BondAngleCosineStdDevs["CA_C_N"].value),
        )
        self.register_buffer(
            "C_N_CA_angle_cos_std",
            torch.as_tensor(BondAngleCosineStdDevs["C_N_CA"].value),
        )

    @staticmethod
    def get_clash_loss(structure: LinearStructure) -> torch.Tensor:
        """
        Calculates loss penalising steric clashes, calculated using Van Der Waals radii.
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
            torch.eye(
                N_dist.shape[-2], N_dist.shape[-2], device=structure.CA_coords.device
            )
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

    def get_bond_angle_loss(self, structure: LinearStructure) -> torch.Tensor:
        """
        Calculates a bond angle loss term, depending on deviations
        between predicted backbone bond angles and their literature values.
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
                0, structure.ptr[:-1], self.CA_C_N_angle_cos
            )
            cos_C_N_CA_bond_angles = cos_C_N_CA_bond_angles.index_fill(
                0, structure.ptr[1:] - 1, self.C_N_CA_angle_cos
            )
        else:
            cos_CA_C_N_bond_angles[-1] = self.CA_C_N_angle_cos
            cos_C_N_CA_bond_angles[-1] = self.C_N_CA_angle_cos

        N_CA_C_angle_loss = (
            torch.clamp_min(
                torch.abs(cos_N_CA_C_bond_angles - self.N_CA_C_angle_cos)
                - self._num_bond_angle_stds * self.N_CA_C_angle_cos_std,
                0.0,
            )
            / cos_N_CA_C_bond_angles.shape[-1]
        )

        CA_C_N_angle_loss = (
            torch.clamp_min(
                torch.abs(cos_CA_C_N_bond_angles - self.CA_C_N_angle_cos)
                - self._num_bond_angle_stds * self.CA_C_N_angle_cos_std,
                0.0,
            )
            / cos_CA_C_N_bond_angles.shape[-1]
        )

        C_N_CA_angle_loss = (
            torch.clamp_min(
                torch.abs(cos_C_N_CA_bond_angles - self.C_N_CA_angle_cos)
                - self._num_bond_angle_stds * self.C_N_CA_angle_cos_std,
                0.0,
            )
            / cos_C_N_CA_bond_angles.shape[-1]
        )

        return N_CA_C_angle_loss + CA_C_N_angle_loss + C_N_CA_angle_loss

    def get_bond_length_loss(self, structure: LinearStructure) -> torch.Tensor:
        """
        Calculates a bond length loss term, depending on deviations
        between predicted backbone bond lengths and their literature values.
        """
        N_CA_bondlengths = torch.linalg.norm(
            structure.N_coords - structure.CA_coords, dim=-1
        )
        CA_C_bondlengths = torch.linalg.norm(
            structure.CA_coords - structure.C_coords, dim=-1
        )
        # Roll the N coords, so that the coords are lined up correctly, and then cut the last element
        C_N_bondlengths = torch.linalg.norm(
            structure.C_coords - structure.N_coords.roll(-1, dims=-2),
            dim=-1,
        )

        if structure.has_batch:
            C_N_bondlengths = C_N_bondlengths.index_fill(
                0, structure.ptr[1:] - 1, BondLengths["C_N"].value
            )
        else:
            C_N_bondlengths[-1] = BondLengths["C_N"].value

        N_CA_length_loss = (
            torch.clamp_min(
                torch.abs(N_CA_bondlengths - BondLengths["N_CA"].value)
                - self._num_bond_len_stds * BondLengthStdDevs["N_CA"].value,
                0.0,
            )
            / N_CA_bondlengths.shape[0]
        )

        CA_C_length_loss = (
            torch.clamp_min(
                torch.abs(CA_C_bondlengths - BondLengths["CA_C"].value)
                - self._num_bond_len_stds * BondLengthStdDevs["CA_C"].value,
                0.0,
            )
            / CA_C_bondlengths.shape[0]
        )

        C_N_length_loss = (
            torch.clamp_min(
                torch.abs(C_N_bondlengths - BondLengths["C_N"].value)
                - self._num_bond_len_stds * BondLengthStdDevs["C_N"].value,
                0.0,
            )
            / C_N_bondlengths.shape[0]
        )

        return N_CA_length_loss + CA_C_length_loss + C_N_length_loss

    @staticmethod
    def get_mse_loss(
        output_structure: LinearStructure, target_structure: LinearStructure
    ) -> torch.Tensor:
        """Calculates MSE between true and predicted coordinates."""

        output_coords = combine_coords(
            output_structure.N_coords,
            output_structure.CA_coords,
            output_structure.C_coords,
            output_structure.CB_coords,
        )
        target_coords = combine_coords(
            target_structure.N_coords,
            target_structure.CA_coords,
            target_structure.C_coords,
            target_structure.CB_coords,
        )
        residue_indices = torch.repeat_interleave(
            torch.arange(len(target_structure), device=target_coords.device), 4
        )

        # sum over coordinates and atoms per-residue
        mse_loss = scatter_sum(
            torch.sum((output_coords - target_coords) ** 2, dim=-1),
            residue_indices,
            dim=0,
        )

        return mse_loss

    def forward(
        self,
        output_structure: LinearStructure,
        target_structure: LinearStructure,
        combine: bool = False,
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Calculates the structural loss, consisting of four loss terms:

            1. Coordinate mean-squared error
            2. Flat-bottom bond length loss
            3. Flat-bottom bond angle loss
            4. Flat-bottom clash loss

        :param output_structure: The predicted structure.
        :param target_structure: The target structure.
        :param combine: Whether to combine the loss terms into a single scalar.
        :return: The four loss terms.
        """

        # MSE loss
        mse_loss = self.get_mse_loss(output_structure, target_structure)

        # Bond length loss
        bond_length_loss = self.get_bond_length_loss(output_structure)

        # Bond angle loss
        bond_angle_loss = self.get_bond_angle_loss(output_structure)

        # Clash loss
        clash_loss = self.get_clash_loss(output_structure)

        if combine:
            return mse_loss + bond_length_loss + bond_angle_loss + clash_loss

        return mse_loss, bond_length_loss, bond_angle_loss, clash_loss
