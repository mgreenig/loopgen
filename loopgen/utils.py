""" Random utility functions. """

from typing import Tuple, Union, Optional, Dict, Literal, Sequence, Callable

from Bio import Align
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from e3nn.o3 import rand_matrix
import numpy as np
import logging as lg
import sys


def is_positive_semidefinite(matrix: torch.Tensor) -> bool:
    """
    Checks if matrix is positive semidefinite.
    :param matrix: A tensor of shape (N, N).
    :return: Boolean.
    """
    eigenvalues, _ = torch.linalg.eig(matrix)
    return torch.all(eigenvalues.real >= 0).item()


def is_symmetric(matrix: torch.Tensor) -> bool:
    """
    Checks if matrix is symmetric.
    :param matrix: A tensor of shape (..., N, N).
    :return: Boolean.
    """
    diff = matrix - matrix.transpose(-2, -1)
    return torch.allclose(diff, torch.zeros_like(matrix))


def is_equivariant(
    fn: Callable[[torch.Tensor, ...], torch.Tensor],
    transform: Callable[[torch.Tensor], torch.Tensor],
    *data: torch.Tensor,
    test_invariant: bool = False,
    atol: float = 1e-5,
) -> bool:
    """
    Tests if an input function that takes some tensor argument(s) `data`
    is equivariant with respect to a transformation of the data, returning a boolean.

    :param fn: Function to test, taking an arbitrary number of tensors as input and returning a single tensor.
    :param transform: Transformation to test equivariance with respect to.
    :param data: Tensors to test rotation equivariance on - these are rotated before being passed to `fn`.
        The output of `fn` on the rotated data is compared to the output of `fn` on the original data, rotated
        after the function.
    :param test_invariant: If True, tests if the function is invariant to rotation. If False, tests if the function
        is equivariant to rotation.
    :param atol: Absolute tolerance for comparing the rotated output to the output of the rotated input.
    """
    transformed_data = [transform(d) for d in data]
    transformed_data_output = fn(*transformed_data)
    output = fn(*data)

    if test_invariant:
        return torch.allclose(transformed_data_output, output, atol=atol)

    transformed_output = transform(output)
    return torch.allclose(transformed_data_output, transformed_output, atol=atol)


def is_rotation_equivariant(
    fn: Callable[[torch.Tensor, ...], torch.Tensor],
    *data: torch.Tensor,
    test_invariant: bool = False,
    atol: float = 1e-5,
) -> bool:
    """
    Tests if an input function that takes some tensor argument(s) `data`
    is equivariant with respect to a rotation of the data, returning a boolean.

    :param fn: Function to test, taking an arbitrary number of tensors as input and returning a single tensor.
    :param data: One or more (N, 3) tensors to test rotation equivariance on - these are rotated
        before being passed to `fn`. The output of `fn` on the rotated data is compared to the
        output of `fn` on the original data, rotated after the function.
    :param test_invariant: If True, tests if the function is invariant to rotation. If False, tests if the function
        is equivariant to rotation.
    :param atol: Absolute tolerance for comparing the rotated output to the output of the rotated input.
    """
    torch.manual_seed(123)
    rot_mat = rand_matrix(1)

    return is_equivariant(
        fn,
        lambda x: torch.matmul(x, rot_mat.transpose(-2, -1)),
        *data,
        test_invariant=test_invariant,
        atol=atol,
    )


def is_translation_equivariant(
    fn: Callable[[torch.Tensor, ...], torch.Tensor],
    *data: torch.Tensor,
    test_invariant: bool = False,
    atol: float = 1e-5,
) -> bool:
    """
    Tests if an input function that takes some tensor argument(s) `data`
    is equivariant with respect to a translation of the data, returning a boolean.

    :param fn: Function to test, taking an arbitrary number of tensors as input and returning a single tensor.
    :param data: One or more (N, 3) tensors to test translation equivariance on -
        these are translated before being passed to `fn`. The output of `fn` on the
        translated data is compared to the output of `fn` on the original data, translated
        after the function.
    :param test_invariant: If True, tests if the function is invariant to translation. If False, tests if the function
        is equivariant to translation.
    :param atol: Absolute tolerance for comparing the translated output to the output of the translated input.
    """
    torch.manual_seed(123)
    translation = torch.randn((1, 3))

    return is_equivariant(
        fn,
        lambda x: x + translation,
        *data,
        test_invariant=test_invariant,
        atol=atol,
    )


def get_logger(
    logfile: str, file_level: int = lg.DEBUG, stream_level: int = lg.INFO
) -> lg.Logger:
    """
    Sets up a generic logger that logs `file_level` messages to the file specified by `logfile` and `stream_level`
    messages to standard output.

    :param logfile: Filename for the log file.
    :param file_level: Level of messages logged to the log file.
    :param stream_level: Level of messages logged to stdout.
    :return: Logger.
    """

    logger = lg.getLogger(__name__)

    formatter = lg.Formatter("%(asctime)s %(levelname)-8s: %(message)s")
    filehandler = lg.FileHandler(logfile)
    filehandler.setLevel(file_level)
    filehandler.setFormatter(formatter)
    streamhandler = lg.StreamHandler(sys.stdout)
    streamhandler.setLevel(stream_level)
    streamhandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger_level = file_level if file_level <= stream_level else stream_level

    logger.setLevel(logger_level)
    logger.propagate = False

    return logger


def get_device(
    device_str: Optional[Literal["cpu", "gpu"]]
) -> Tuple[torch.device, Literal["cpu", "gpu"]]:
    """
    Converts an optional input string (usually from command line args)
    into a torch.device object and a string "cpu" or "gpu"
    (the accelerator argument used by the pytorch lightning Trainer).
    If the input string is None, searches for a GPU and uses one if available.
    """

    if device_str is not None:
        accelerator = device_str
        if accelerator.lower() == "gpu":
            device = torch.device("cuda")
        elif accelerator.lower() == "cpu":
            device = torch.device("cpu")
        else:
            raise ValueError(
                f"Accelerator {accelerator} not recognised: should be one of ('cpu', 'gpu')"
            )
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "gpu"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    return device, accelerator


def node_type_subgraph(graph: Data, node_type: int) -> Data:
    """
    Gets the subgraph of a graph for nodes of a particular type,
    where node types are stored under the `node_type` attribute of `graph`.
    Can handle graphs with `batch` and `ptr` attributes.
    """

    if not hasattr(graph, "node_type"):
        raise AttributeError("Input graph does not have an attribute node_type.")

    node_mask = graph.node_type == node_type
    node_indices = torch.nonzero(node_mask).squeeze(-1)
    node_subgraph = graph.subgraph(node_indices)

    if hasattr(graph, "ptr") and hasattr(node_subgraph, "batch"):
        _, node_subgraph.ptr = standardise_batch_and_ptr(node_subgraph.batch, None)

    delattr(node_subgraph, "node_type")
    delattr(node_subgraph, "edge_type")

    return node_subgraph


def expand_batch_tensor(
    batch_tensor: torch.Tensor,
    num_nodes_per_graph: torch.Tensor,
    pad_value: float = 0.0,
    return_padding_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Converts a batch tensor of shape (M, ...) (e.g. those used by the Batch class in torch_geometric) into a padded
    tensor of shape (B, N, ...), where B is the number of graphs and N is the maximum number of nodes
    of any graph in the batch. Pad elements by default are represented as zeros in the dtype of `batch_tensor`. Requires
    as input a 1-D tensor of length B giving the number of nodes in each graph in the batch.

    :param batch_tensor: 2-D batch tensor of shape (M, ...), where M is the total number of nodes in the batch.
    :param num_nodes_per_graph: Integer tensor of number of nodes per graph.
    :param pad_value: Value used to fill pad elements (default: 0.0).
    :param return_padding_mask: Whether to return a boolean mask tensor indicating pad elements as well as
        the expanded batch tensor.
    :return: Padded batch tensor of shape (B, N, ...), or a 2-tuple consisting of the padded batch tensor and
        a padding mask (if return_padding_mask=True).
    """

    num_graphs = len(num_nodes_per_graph)
    max_num_nodes = torch.max(num_nodes_per_graph).item()

    expanded_tensor = torch.zeros(
        (num_graphs, max_num_nodes) + batch_tensor.shape[1:],
        device=batch_tensor.device,
        dtype=batch_tensor.dtype,
    )

    node_index_tensor = (
        torch.arange(max_num_nodes, device=expanded_tensor.device)
        .unsqueeze(0)
        .expand(num_graphs, -1)
    )

    non_pad_mask = torch.zeros(
        (num_graphs, max_num_nodes), device=expanded_tensor.device, dtype=torch.bool
    )
    non_pad_mask[node_index_tensor < num_nodes_per_graph.unsqueeze(-1)] = True

    expanded_tensor[non_pad_mask] = batch_tensor

    pad_mask = ~non_pad_mask
    expanded_tensor[pad_mask] = torch.as_tensor(
        pad_value, device=batch_tensor.device, dtype=batch_tensor.dtype
    )

    if return_padding_mask is True:
        return expanded_tensor, pad_mask

    return expanded_tensor


def standardise_batch_and_ptr(
    batch: Optional[torch.Tensor], ptr: Optional[torch.Tensor]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    For (optional) inputs batch and ptr (as used in `torch_geometric`), standardises both in the
    case one is provided and the other is not. If both are not None, or
    both are None, simply returns the original objects.

    A `batch` object is a vector of length equal to the number of objects in the
    batch whose elements denote the number of the batch (a positive integer) to which each
    object belongs. A `ptr` object is a vector of length equal to the number
    of batches plus one, whose element at index `i` denotes the start index
    of objects in batch `i` in the vector `batch`. The final element of `ptr`
    is the number of objects in the batch.
    """

    if batch is not None and ptr is None:
        if len(batch.shape) < 1:
            batch = batch.unsqueeze(-1)
        ptr = torch.nn.functional.pad(
            torch.cumsum(torch.bincount(batch), dim=-1), (1, 0)
        )
    if ptr is not None and batch is None:
        num_nodes = torch.diff(ptr)
        batch = torch.repeat_interleave(
            torch.arange(len(num_nodes), device=ptr.device), repeats=num_nodes
        )

    return batch, ptr


def get_distance_matrix(
    coords: torch.Tensor,
    other_coords: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    other_batch: Optional[torch.Tensor] = None,
    pad_value: float = -1.0,
) -> torch.Tensor:
    """
    Calculates a euclidean distance matrix for one or two set(s) of coordinates. If the second
    set of coordinates is not provided, a symmetric distance matrix for `coords` will be returned.
    """
    if other_coords is None:
        other_coords = coords

    if batch is not None:
        num_nodes_per_graph = torch.bincount(batch)
        coords, padding_mask = expand_batch_tensor(
            coords, num_nodes_per_graph, return_padding_mask=True
        )

        if other_batch is not None:
            num_nodes_per_graph = torch.bincount(other_batch)

        other_coords, other_padding_mask = expand_batch_tensor(
            other_coords, num_nodes_per_graph, return_padding_mask=True
        )

    distance_matrix = torch.linalg.norm(
        coords.unsqueeze(-2) - other_coords.unsqueeze(-3), dim=-1
    )

    if batch is not None:
        matrix_padding_mask = torch.logical_or(
            padding_mask.unsqueeze(-1), other_padding_mask.unsqueeze(-2)
        )
        distance_matrix = torch.where(matrix_padding_mask, pad_value, distance_matrix)

    return distance_matrix


def get_covalent_bonds(
    atom_coords_1: torch.Tensor,
    atom_coords_2: torch.Tensor,
    min_len: float = 0.4,
    max_len: float = 1.9,
) -> torch.Tensor:
    """
    Calculates a binary tensor (represented in float32) specifying whether each pair
    atom coordinates represents a covalent bond, based on a thresholded euclidean distance between atoms.
    For input `atom_coords_1` and `atom_coords_2` both of shape (..., M, 3),
    the output has shape (..., M), where each element `output[i]` specifies whether
    `atom_coords_1[i]` and `atom_coords_2[i]` specify a pair of atoms that are covalently bonded.

    :param atom_coords_1: (..., M, 3) tensor of atomic coordinates.
    :param atom_coords_2: (..., M, 3) tensor of atomic coordinates.
    :param min_len: Minimum length of a covalent bond in angstroms (default: 0.4).
    :param max_len: Maximum length of a covalent bond in angstroms (default: 1.9).
    :return: (..., M) binary tensor represented in float32.
    """

    distances = torch.linalg.norm(atom_coords_2 - atom_coords_1, dim=-1)

    covalent_bonds = (distances < max_len) & (distances > min_len)

    return covalent_bonds.to(torch.float32)


def combine_coords(*coord_args: torch.Tensor, coord_dim: int = 3) -> torch.Tensor:
    """
    Combines a set of identically shaped 3-D coordinates into a tensor of shape (..., 3). The combination
    preserves ordering along the 2nd-to-last dimension among the input arguments.

    :param coord_args: Tensors of same shape (..., 3)
    :param coord_dim: Dimensionality of the coordinate vectors.
    :return: Tensor of shape (..., 3)
    """

    outshape = coord_args[0].shape[: len(coord_args[0].shape) - 2] + (-1, coord_dim)
    combined_coords = torch.cat(coord_args, dim=-1).view(outshape)

    return combined_coords


def get_unit_normals(v1s: torch.Tensor, v2s: torch.Tensor) -> torch.Tensor:
    """
    Calculates unit normals between pairs of vectors. Assumes `v1s.shape[-1] = 3` and `v2s.shape[-1] = 3`.
    """

    normals = torch.cross(v1s, v2s)
    unit_normals = torch.nn.functional.normalize(normals, dim=-1)
    return unit_normals


def get_angles(unit_v1s: torch.Tensor, unit_v2s: torch.Tensor) -> torch.Tensor:
    """
    Calculates the angle between two sets of unit vectors.

    :param unit_v1s: Unit vectors.
    :param unit_v2s: Unit vectors.
    :return: Tensor of angles in radians.
    """

    angle_inverses = torch.clamp(
        torch.sum(unit_v1s * unit_v2s, dim=-1), min=-1 + 1e-6, max=1 - 1e-6
    )
    angles = torch.acos(angle_inverses)
    return angles


# calculates orientation angles between adjacent orientation vectors
def get_orientation_angles(orientation_vectors: torch.Tensor) -> torch.Tensor:
    """
    Calculates angles between adjacent pairs of vectors in a tensor, where
    adjacency is measured along the 2nd-to-last dimension.
    Input vectors do not need to be normalised before passing to the function.

    :param orientation_vectors: Tensor of vectors.
    :return: Tensor of angles.
    """

    unit_orientation_vectors = torch.nn.functional.normalize(
        orientation_vectors, dim=-1
    )

    orientation_angles = get_angles(
        unit_orientation_vectors[..., 1:, :],
        unit_orientation_vectors[..., :-1, :],
    )

    return orientation_angles


# calculates dihedral angles for a list of coordinates
def get_dihedral_angles(coords: torch.Tensor) -> torch.Tensor:
    """
    Calculates dihedral angles for a tensor of coordinates of shape (..., 3).

    :param coords: Tensor with 3-D coordinates in last dimension.
    :return: Tensor of dihedral angles with shape (..., 1)
    """

    num_unit_normals = coords.shape[-2] - 2
    num_dihedrals = num_unit_normals - 1

    # get unit normal vectors defined by pairs of adjacent bonds
    unit_normals = get_unit_normals(
        (
            torch.narrow(coords, -2, 0, num_unit_normals)
            - torch.narrow(coords, -2, 1, num_unit_normals)
        ),
        (
            torch.narrow(coords, -2, 2, num_unit_normals)
            - torch.narrow(coords, -2, 1, num_unit_normals)
        ),
    )

    # get signs of each dihedral angle
    dihedral_signs = torch.sign(
        torch.sum(
            torch.narrow(unit_normals, -2, 0, num_dihedrals)
            * (
                torch.narrow(coords, -2, 3, num_dihedrals)
                - torch.narrow(coords, -2, 2, num_dihedrals)
            ),
            dim=-1,
        )
    )

    dihedral_angles = (
        get_angles(
            torch.narrow(unit_normals, -2, 1, num_dihedrals),
            torch.narrow(unit_normals, -2, 0, num_dihedrals),
        )
        * dihedral_signs
    )

    return dihedral_angles


def so3_hat(skew_sym: torch.Tensor) -> torch.Tensor:
    """
    Calculates the hat operator for SO(3), which maps a 3x3 skew-symmetric matrix
    to a 3-D rotation vector.

    :param skew_sym: (..., 3, 3) tensor of 3x3 skew-symmetric matrices.
    :return: (..., 3) tensor of 3-D rotation vectors.
    """
    return torch.stack(
        [skew_sym[..., 2, 1], skew_sym[..., 0, 2], skew_sym[..., 1, 0]], dim=-1
    )


def so3_hat_inv(log_rot: torch.Tensor) -> torch.Tensor:
    """
    Calculates the inverse of the hat operator for SO(3), which maps a
    3-D rotation vector to a 3x3 skew-symmetric matrix.

    :param log_rot: (..., 3) tensor of 3-D rotation vectors.
    :return: (..., 3, 3) tensor of 3x3 skew-symmetric matrices.
    """
    skew_symmetric = torch.zeros(log_rot.shape[:-1] + (3, 3), device=log_rot.device)
    skew_symmetric[..., 0, 1] = -log_rot[..., 2]
    skew_symmetric[..., 0, 2] = log_rot[..., 1]
    skew_symmetric[..., 1, 0] = log_rot[..., 2]
    skew_symmetric[..., 1, 2] = -log_rot[..., 0]
    skew_symmetric[..., 2, 0] = -log_rot[..., 1]
    skew_symmetric[..., 2, 1] = log_rot[..., 0]
    return skew_symmetric


def so3_exp_map(log_rot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculates the exponential map of a tensor representing a tangent
    element of SO(3) as a 3-D rotation vector.

    :param log_rot: (..., 3) tensor of 3-D rotation vectors.
    :param eps: Small number to avoid division by zero - the rotation angle is clamped to this value as a minimum.
    :return: Tensor of 3x3 rotation matrices.
    """

    skew_symmetric = so3_hat_inv(log_rot)

    theta = torch.linalg.norm(log_rot, dim=-1, keepdim=True).unsqueeze(-1)
    theta_sq = torch.clamp_min(theta**2, eps)

    matrix_exp = (
        torch.eye(3, device=skew_symmetric.device, dtype=skew_symmetric.dtype)
        + (torch.sin(theta) / theta) * skew_symmetric
        + ((1 - torch.cos(theta)) / theta_sq)
        * torch.matmul(skew_symmetric, skew_symmetric)
    )
    return matrix_exp


def so3_log_map(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculates the logarithmic map of a tensor representing an element of SO(3)
    as a 3x3 rotation matrix.

    :param matrix: (..., 3, 3) tensor of 3x3 rotation matrices.
    :param eps: Small number to avoid division by zero in the inverse cosine calculation.
    :return: Tensor of 3-D rotation vectors.
    """

    theta = torch.acos(
        torch.clamp(
            (torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1) - 1) / 2,
            -1 + eps,
            1 - eps,
        )
    ).unsqueeze(-1)
    skew_symmetric = matrix - matrix.transpose(-2, -1)
    skew_symmetric = skew_symmetric / (2 * torch.sin(theta.unsqueeze(-1)))

    log_rot = so3_hat(skew_symmetric) * theta

    return log_rot
