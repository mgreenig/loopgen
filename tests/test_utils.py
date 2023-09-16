""" Tests the utility functions in loopgen.utils."""

import pytest
import torch
from torch_geometric.data import Data

from loopgen.utils import (
    is_equivariant,
    is_rotation_equivariant,
    is_translation_equivariant,
    so3_log_map,
    so3_exp_map,
    so3_hat,
    so3_hat_inv,
    node_type_subgraph,
)

NUM_COORDS = 10


@pytest.fixture
def coords() -> torch.Tensor:
    """Returns a tensor of coordinates."""
    torch.manual_seed(123)
    return torch.randn(NUM_COORDS, 3)


def test_is_equivariant(coords):
    """
    Tests the is_equivariant() function, which determines if an
    input function is equivariant with respect to another function.
    """
    # assert that identity is equivariant with respect to addition and multiplication
    assert is_equivariant(
        lambda x: x, lambda x: x + 1, coords
    ), "Identity should be equivariant with respect to addition"
    assert is_equivariant(
        lambda x: x, lambda x: x * 2, coords
    ), "Identity should be equivariant with respect to multiplication"

    # check multiple arguments work
    assert is_equivariant(
        lambda x, y: x + y, lambda x: x * 2, coords, coords
    ), "Identity should be equivariant with respect to multiplication"

    # check that squaring is not equivariant with respect to addition
    assert not is_equivariant(
        lambda x: x**2, lambda x: x + 1, coords
    ), "Squaring should not be equivariant with respect to addition"


def test_is_rotation_equivariant(coords):
    """
    Tests the is_rotation_equivariant() function,
    which determines if an input function is equivariant to 3D rotation.
    """
    assert is_rotation_equivariant(
        lambda x, y: x + y, coords, coords
    ), "Addition of two vectors should  be rotation equivariant"
    assert not is_rotation_equivariant(
        lambda x: x + 1, coords
    ), "Addition by a constant should not be rotation equivariant"
    assert is_rotation_equivariant(
        lambda x: x * 2, coords
    ), "Multiplication should be rotation equivariant"
    assert not is_rotation_equivariant(
        lambda x: x * torch.as_tensor([1.0, 2.0, 3.0]), coords
    ), "Coordinate-wise multiplication should not be rotation equivariant"


def test_is_translation_equivariant(coords):
    """
    Tests the is_translation_equivariant() function,
    which determines if an input function is equivariant to 3D translation.
    """
    assert not is_translation_equivariant(
        lambda x, y: x + y, coords, coords
    ), "Addition of two vectors should not be translation equivariant"
    assert is_translation_equivariant(
        lambda x: x + 1, coords
    ), "Addition by a constant should be translation equivariant"
    assert is_translation_equivariant(
        lambda x, y: (x + y) / 2, coords, coords
    ), "Mean should be translation equivariant"


def test_so3_exp_map():
    """
    Tests the so3_exp_map() function, which maps a rotation vector to its
    corresponding rotation matrix.
    """
    # test that the output shape is correct
    torch.manual_seed(123)
    num_rots = 10
    rot_vecs = torch.randn(num_rots, 3)
    rot_mats = so3_exp_map(rot_vecs)
    assert rot_mats.shape == (
        10,
        3,
        3,
    ), "Output shape should be (..., 3, 3)"

    assert torch.allclose(
        torch.det(rot_mats), torch.ones(num_rots)
    ), "Determinant should be 1"
    assert torch.allclose(
        torch.linalg.inv(rot_mats), torch.transpose(rot_mats, -1, -2)
    ), "Inverse should be transpose"

    # test that the exp map undoes log map
    assert torch.allclose(
        so3_log_map(rot_mats), rot_vecs
    ), "Exp map should be the inverse of log map"


def test_so3_log_map():
    """
    Tests the so3_log_map() function, which maps a rotation matrix to its
    corresponding rotation vector.
    """

    # test that the output shape is correct
    torch.manual_seed(123)
    num_rots = 10
    rot_mat = torch.randn(num_rots, 3, 3)
    assert so3_log_map(rot_mat).shape == (num_rots, 3), "Output shape should be (3,)"

    # test that identity matrix maps to zero vector
    assert torch.all(
        so3_log_map(torch.eye(3)) == torch.zeros(3)
    ), "Identity matrix should map to zero vector"

    # test that the log map undoes exp map
    rot_vec = torch.tensor([0.1, 0.2, 0.3])
    rot_mat = so3_exp_map(rot_vec)
    assert torch.allclose(
        so3_log_map(rot_mat), rot_vec
    ), "Log map should be the inverse of exp map"


def test_so3_hat_inv():
    """
    Tests the so3_hat() function, which maps a rotation vector to its
    corresponding skew-symmetric matrix.
    """

    torch.manual_seed(123)
    num_rots = 10
    rot_vec = torch.randn(num_rots, 3)
    skew_sym = so3_hat_inv(rot_vec)

    assert skew_sym.shape == (num_rots, 3, 3), "Output shape should be (..., 3, 3)"
    assert torch.allclose(
        skew_sym, -torch.transpose(skew_sym, -1, -2)
    ), "Output matrix should be skew-symmetric"


def test_so3_hat():
    """
    Tests the so3_hat_inv() function, which converts from a skew-symmetric matrix
    into a rotation vector.
    """

    torch.manual_seed(123)
    num_rots = 10
    rot_vec = torch.randn(num_rots, 3)
    skew_sym = so3_hat_inv(rot_vec)
    output_rot_vec = so3_hat(skew_sym)

    assert torch.allclose(
        rot_vec, output_rot_vec
    ), "Calling hat() on the output of hat_inv() should be the identity"


def test_node_type_subgraph():
    """
    Tests the node_type_subgraph() function, which returns a subgraph
    containing only the specified node types.
    """
    num_nodes = 10
    num_features = 5
    node_type_0_features = torch.zeros(num_nodes // 2, num_features)
    node_type_1_features = torch.ones(num_nodes // 2, num_features)
    node_features = torch.cat(
        [
            node_type_0_features,
            node_type_1_features,
        ],
        dim=0,
    )
    edge_index = torch.cat(
        [
            torch.arange(num_nodes).repeat_interleave(num_nodes),
            torch.arange(num_nodes).repeat(num_nodes),
        ],
        dim=0,
    )
    node_types = torch.cat(
        [torch.zeros(num_nodes // 2), torch.ones(num_nodes // 2)], dim=0
    ).to(torch.long)
    graph = Data(x=node_features, node_type=node_types, edge_index=edge_index)

    node_type_0_subgraph = node_type_subgraph(graph, 0)
    node_type_1_subgraph = node_type_subgraph(graph, 1)

    assert torch.allclose(
        node_type_0_subgraph.x, node_type_0_features
    ), "Node type 0 subgraph should only contain node type 1 features"

    assert torch.allclose(
        node_type_1_subgraph.x, node_type_1_features
    ), "Node type 1 subgraph should only contain node type 1 features"
