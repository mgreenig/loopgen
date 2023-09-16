"""
Tests the Geometric Vector Perceptron module.
"""

from typing import Tuple
import torch
import pytest
from torch_scatter import scatter_sum
from e3nn.o3 import rand_matrix

from loopgen.nn.gvp import (
    combine_gvp_features,
    separate_gvp_features,
    GeometricVectorPerceptron,
    GVPDropout,
    GVPLayerNorm,
    GVPMessage,
    GVPUpdate,
    GVPAttentionTypes,
    GVPAttention,
    GVPAttentionStrategySelector,
    GVPMessagePassing,
)
from loopgen.utils import is_rotation_equivariant

# Some parameters for the testing data
NUM_NODES = 100
BATCH_SIZE = 10
NUM_SCALAR_FEATURES = 20
NUM_VECTOR_FEATURES = 5
NUM_COORDS = 1
VECTOR_DIM_SIZE = 3
COORD_DIM_SIZE = 3
NUM_EDGES = 500
NUM_EDGE_SCALAR_FEATURES = 10
NUM_EDGE_VECTOR_FEATURES = 4

assert NUM_NODES % BATCH_SIZE == 0, "NUM_NODES must be divisible by BATCH_SIZE"


@pytest.fixture
def gvp_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates some scalar and vector features for a GVP."""
    torch.manual_seed(123)
    scalar_features = torch.randn((NUM_NODES, NUM_SCALAR_FEATURES))
    vector_features = torch.randn((NUM_NODES, NUM_VECTOR_FEATURES, VECTOR_DIM_SIZE))
    return scalar_features, vector_features


@pytest.fixture
def gvp_edge_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates an edge index, scalar edge features, and vector edge features for a GVP-GNN."""
    torch.manual_seed(123)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
    edge_scalar_features = torch.randn((NUM_EDGES, NUM_EDGE_SCALAR_FEATURES))
    edge_vector_features = torch.randn(
        (NUM_EDGES, NUM_EDGE_VECTOR_FEATURES, VECTOR_DIM_SIZE)
    )
    return edge_index, edge_scalar_features, edge_vector_features


@pytest.fixture
def orientations() -> torch.Tensor:
    """Generates some orientation (rotation) matrices."""
    torch.manual_seed(123)
    return rand_matrix(NUM_NODES)


def test_combine_gvp_features(gvp_data):
    """
    Tests the combine_gvp_features() function, which combines separate scalar and vector features
    into a single tensor, where the vector features are flattened.
    """
    scalar_features, vector_features = gvp_data
    combined_features = combine_gvp_features(scalar_features, vector_features)
    assert combined_features.shape == (
        NUM_NODES,
        NUM_SCALAR_FEATURES + (NUM_VECTOR_FEATURES * VECTOR_DIM_SIZE),
    ), "Wrong shape for combined features"


def test_separate_gvp_features(gvp_data):
    """
    Tests the separate_gvp_features() function, which separates a flattened tensor of scalar/vector features
    into two separate tensors.
    """
    scalar_features, vector_features = gvp_data
    combined_features = combine_gvp_features(scalar_features, vector_features)
    output_scalar_features, output_vector_features = separate_gvp_features(
        combined_features, NUM_VECTOR_FEATURES, VECTOR_DIM_SIZE
    )
    assert torch.allclose(
        scalar_features, output_scalar_features
    ), "Scalar features do not match"
    assert torch.allclose(
        vector_features, output_vector_features
    ), "Vector features do not match"


def test_gvp(gvp_data):
    """
    Tests the GeometricVectorPerceptron module, which is a single layer that processes
    scalar and vector features, ensuring that output vector features are rotation equivariant.
    """
    scalar_features, vector_features = gvp_data
    out_scalar_channels = 10
    out_vector_channels = 5
    gvp_layer = GeometricVectorPerceptron(
        NUM_SCALAR_FEATURES,
        out_scalar_channels,
        NUM_VECTOR_FEATURES,
        out_vector_channels,
    )

    # Test forward pass
    output_scalar_features, output_vector_features = gvp_layer(gvp_data)
    assert output_scalar_features.shape == (
        NUM_NODES,
        out_scalar_channels,
    ), "Wrong shape for output scalar features"
    assert output_vector_features.shape == (
        NUM_NODES,
        out_vector_channels,
        VECTOR_DIM_SIZE,
    ), "Wrong shape for output vector features"

    # Test equivariance using a GVP function that takes only
    # vector inputs (scalar features are fixed) and returns only vector outputs
    gvp_fn = lambda vs: gvp_layer((scalar_features, vs))[1]

    assert is_rotation_equivariant(
        gvp_fn, vector_features
    ), "GVP should be rotation equivariant"

    # Test scalars/vectors interacting by swapping features and seeing if the other type of output features change
    torch.manual_seed(42)
    new_scalar_features = torch.randn((NUM_NODES, NUM_SCALAR_FEATURES))
    new_vector_features = torch.randn((NUM_NODES, NUM_VECTOR_FEATURES, VECTOR_DIM_SIZE))

    new_output_scalar_features, _ = gvp_layer((scalar_features, new_vector_features))
    assert not torch.allclose(
        output_scalar_features, new_output_scalar_features, atol=1e-3
    ), "Scalar features should change when vector features change"

    _, new_output_vector_features = gvp_layer((new_scalar_features, vector_features))
    assert not torch.allclose(
        output_vector_features, new_output_vector_features, atol=1e-3
    ), "Vector features should change when scalar features change"


def test_gvp_dropout(gvp_data):
    """
    Tests the GVPDropout module, which performs normal dropout on scalar features
    and zeros entire 3-D vector features at once.
    """
    scalar_features, vector_features = gvp_data

    torch.manual_seed(123)
    gvp_dropout = GVPDropout(0.5, NUM_VECTOR_FEATURES)

    # Sanity check that no input features are zero
    assert not torch.any(
        scalar_features == 0.0
    ).item(), "No scalar features should be zero"
    assert not torch.any(
        vector_features == 0.0
    ).item(), "No vector features should be zero"

    # Test forward pass
    output_scalar_features, output_vector_features = gvp_dropout(gvp_data)
    assert torch.any(
        output_scalar_features == 0.0
    ), "Some scalar features should be zero after dropout"
    assert torch.any(
        output_vector_features == 0.0
    ), "Some vector features should be zero dropout"

    # Test that entire vector channels are dropped out at once
    assert torch.all(
        torch.any(output_vector_features == 0.0, dim=-1)
        == torch.all(output_vector_features == 0.0, dim=-1)
    ).item(), "Entire vector channels should be dropped out at once"


def test_gvp_layer_norm(gvp_data):
    """
    Tests the GVPLayerNorm module, which performs typical layer normalization on scalar features
    and normalises vector features so that the norm of the vector feature matrix is equal to the
    number of vector features.
    """
    gvp_layer_norm = GVPLayerNorm(NUM_SCALAR_FEATURES, NUM_VECTOR_FEATURES)

    # Test forward pass
    output_scalar_features, output_vector_features = gvp_layer_norm(gvp_data)

    assert torch.allclose(
        torch.mean(output_scalar_features, dim=-1), torch.zeros(NUM_NODES), atol=1e-5
    ), "Scalar feature outputs of LayerNorm should be zero-mean"

    assert torch.allclose(
        torch.std(output_scalar_features, unbiased=False, dim=-1),
        torch.ones(NUM_NODES),
        atol=1e-5,
    ), "Scalar feature outputs of LayerNorm should be unit standard deviation"

    assert torch.allclose(
        torch.linalg.norm(output_vector_features, dim=(-2, -1)),
        torch.sqrt(torch.as_tensor(NUM_VECTOR_FEATURES)),
        atol=1e-5,
    ), "Vector feature outputs of LayerNorm should have norm equal to sqrt(num_vector_features)"


def test_gvp_message(gvp_data):
    """
    Tests the GVPMessage module, which transforms a message using sequential GVPs.
    """

    scalar_features, vector_features = gvp_data
    num_gvps = 3
    gvp_message = GVPMessage(NUM_SCALAR_FEATURES, NUM_VECTOR_FEATURES, num_gvps)

    assert (
        len(gvp_message._message_gvp_layers) == num_gvps
    ), "Wrong number of message GVP layers"

    out_scalar_features, out_vector_features = gvp_message(
        scalar_features, vector_features
    )

    assert (
        out_scalar_features.shape == scalar_features.shape
    ), "GVPMessage should not change shape of scalar inputs"
    assert (
        out_vector_features.shape == vector_features.shape
    ), "GVPMessage should not change shape of vector inputs"

    message_fn = lambda v_x: gvp_message(scalar_features, v_x)[1]
    assert is_rotation_equivariant(
        message_fn, vector_features
    ), "GVPMessage should be rotation equivariant"


def test_gvp_update(gvp_data):
    """
    Tests the GVPUpdate module, which transforms a message using sequential GVPs.
    """

    scalar_features, vector_features = gvp_data
    num_output_scalar_features = NUM_SCALAR_FEATURES + 1
    num_output_vector_features = NUM_VECTOR_FEATURES + 1
    state_scalar_features = torch.nn.functional.pad(scalar_features, (0, 1))
    state_vector_features = torch.nn.functional.pad(vector_features, (0, 0, 0, 1))
    num_gvps = 3
    dropout = 0.2
    gvp_update = GVPUpdate(
        num_output_scalar_features,
        NUM_SCALAR_FEATURES,
        num_output_vector_features,
        NUM_VECTOR_FEATURES,
        num_gvps,
        dropout,
    )
    gvp_update.eval()

    assert (
        len(gvp_update._update_gvp_layers) == num_gvps
    ), "Wrong number of update GVP layers"

    out_scalar_features, out_vector_features = gvp_update(
        state_scalar_features, scalar_features, state_vector_features, vector_features
    )

    assert (
        out_scalar_features.shape == state_scalar_features.shape
    ), "GVPUpdate did not produce the correct number of output scalar features"
    assert (
        out_vector_features.shape == state_vector_features.shape
    ), "GVPUpdate did not produce the correct number of output vector features"

    message_fn = lambda state_v_x, v_x: gvp_update(
        state_scalar_features, scalar_features, state_v_x, v_x
    )[1]
    assert is_rotation_equivariant(
        message_fn, state_vector_features, vector_features
    ), "GVPUpdate should be rotation equivariant"


def test_gvp_attention(gvp_data, orientations):
    """
    Tests the GVPAttentionStrategySelector - which selects a GVPAttention strategy for an input string -
    and the GVPAttention modules, which calculate attention coefficients based on input scalar and vector features.
    """

    scalar_features, vector_features = gvp_data
    batch = torch.arange(NUM_NODES // BATCH_SIZE).repeat_interleave(BATCH_SIZE)
    num_heads = 5

    selector = GVPAttentionStrategySelector(
        NUM_SCALAR_FEATURES,
        NUM_VECTOR_FEATURES,
        NUM_SCALAR_FEATURES,
        NUM_VECTOR_FEATURES,
        num_heads=num_heads,
    )
    for attn_type in GVPAttentionTypes.__args__:
        attn_module = selector.get_layer(attn_type)
        assert isinstance(
            attn_module, GVPAttention
        ), "GVPAttentionStrategySelector should return a GVPAttention module"
        attn_weights = attn_module(
            scalar_features,
            vector_features,
            orientations,
            scalar_features,
            vector_features,
            orientations,
            batch,
        )
        assert attn_weights.shape == (
            NUM_NODES,
            num_heads,
        ), "Wrong shape for attention weights"

        summed_weights = scatter_sum(attn_weights, batch, dim=0)
        assert torch.allclose(
            summed_weights, torch.ones_like(summed_weights)
        ), "Attention weights should sum to 1 over the softmax index"

        # Test equivariance using a GVP function that takes only
        # vector inputs (scalar features are fixed) and returns only vector outputs
        attn_fn = lambda v, o: attn_module(
            scalar_features, v, o, scalar_features, v, o, batch
        )
        assert is_rotation_equivariant(
            attn_fn, vector_features, orientations, test_invariant=True
        ), "Attention weights should be rotation invariant"

    # check that passing an invalid string raises a ValueError
    with pytest.raises(ValueError):
        selector.get_layer("invalid")


class TestGVPMessagePassing:
    """Tests the GVPMessagePassing module."""

    # some default message passing parameters
    aggr = "sum"
    num_message_gvps = 3
    num_update_gvps = 2
    dropout = 0.2
    num_heads = 1

    message_passing_layer = GVPMessagePassing(
        NUM_SCALAR_FEATURES,
        NUM_VECTOR_FEATURES,
        NUM_EDGE_SCALAR_FEATURES,
        NUM_EDGE_VECTOR_FEATURES,
        aggr,
        num_message_gvps,
        num_update_gvps,
        dropout,
        num_heads=num_heads,
        vector_dim_size=VECTOR_DIM_SIZE,
    )
    message_passing_layer.eval()

    message_passing_layer_attn = GVPMessagePassing(
        NUM_SCALAR_FEATURES,
        NUM_VECTOR_FEATURES,
        NUM_EDGE_SCALAR_FEATURES,
        NUM_EDGE_VECTOR_FEATURES,
        aggr,
        num_message_gvps,
        num_update_gvps,
        dropout,
        attention_type="flatten",
        num_heads=num_heads,
        vector_dim_size=VECTOR_DIM_SIZE,
    )
    message_passing_layer_attn.eval()

    message_out_channels = (
        NUM_SCALAR_FEATURES + (NUM_VECTOR_FEATURES * VECTOR_DIM_SIZE)
    ) * 2 + (NUM_EDGE_SCALAR_FEATURES + (NUM_EDGE_VECTOR_FEATURES * VECTOR_DIM_SIZE))

    def test_init(self):
        """Tests the constructor."""

        assert (
            len(self.message_passing_layer._message_layer._message_gvp_layers)
            == self.num_message_gvps
        ), "Wrong number of message GVP layers"
        assert (
            len(self.message_passing_layer._node_update_layer._update_gvp_layers)
            == self.num_update_gvps
        ), "Wrong number of update GVP layers"

    @staticmethod
    def get_features(
        node_features: Tuple[torch.Tensor, torch.Tensor],
        edge_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """Converts the GVP data into the features used by the message passing module."""
        scalar_features, vector_features = node_features
        edge_index, edge_scalar_features, edge_vector_features = edge_features

        x = torch.cat([scalar_features, vector_features.flatten(start_dim=-2)], dim=-1)
        edge_attr = torch.cat(
            [edge_scalar_features, edge_vector_features.flatten(start_dim=-2)],
            dim=-1,
        )

        return x, edge_attr

    def test_forward(self, gvp_data, gvp_edge_data, orientations):
        """Tests the forward pass and its constituent operations (message, aggregate and update)."""

        scalar_features = gvp_data[0]
        vector_features = gvp_data[1]

        edge_index, edge_scalar_features, edge_vector_features = gvp_edge_data

        x, edge_attr = self.get_features(gvp_data, gvp_edge_data)
        x_i = x[edge_index[1]]
        x_j = x[edge_index[0]]

        message = self.message_passing_layer.message(
            x_i,
            x_j,
            edge_attr,
            orientations[edge_index[1]].flatten(start_dim=-2),
            orientations[edge_index[0]].flatten(start_dim=-2),
            edge_index[1],
        )
        attn_message = self.message_passing_layer_attn.message(
            x_i,
            x_j,
            edge_attr,
            orientations[edge_index[1]].flatten(start_dim=-2),
            orientations[edge_index[0]].flatten(start_dim=-2),
            edge_index[1],
        )

        assert message.shape == (
            NUM_EDGES,
            self.message_out_channels,
        ), "Message shape is not correct"
        assert attn_message.shape == (
            NUM_EDGES,
            self.message_out_channels,
        ), "Message shape is not correct"
        assert not torch.any(
            message.isnan()
        ).item(), "Output of message() contains NaNs"
        assert not torch.any(
            attn_message.isnan()
        ).item(), "Output of message() contains NaNs"

        aggr_message = self.message_passing_layer.aggregate(message, edge_index[1])
        aggr_attn_message = self.message_passing_layer_attn.aggregate(
            attn_message, edge_index[1]
        )

        updated = self.message_passing_layer.update(aggr_message, x)
        updated_attn = self.message_passing_layer_attn.update(aggr_attn_message, x)

        assert updated.shape == x.shape, "Updated shape is not correct"
        assert (
            updated_attn.shape == x.shape
        ), "Updated shape is not correct (with attention)"
        assert not torch.any(updated.isnan()).item(), "Output of update() contains NaNs"
        assert not torch.any(
            updated_attn.isnan()
        ).item(), "Output of update() contains NaNs"

        def vector_forward_pass(layer, v_x, e_v_x, ors):
            """Performs a forward pass of the message passing layer as a function of the vector features."""
            node_features, edge_features = self.get_features(
                (scalar_features, v_x), (edge_index, edge_scalar_features, e_v_x)
            )
            orientation_features = ors.flatten(start_dim=-2)
            output = layer(
                node_features, edge_index, edge_features, orientation_features
            )
            _, v_features = separate_gvp_features(
                output, NUM_VECTOR_FEATURES, VECTOR_DIM_SIZE
            )
            return v_features

        # Test equivariance of the forward pass
        assert is_rotation_equivariant(
            lambda v_x, e_v_x, ors: vector_forward_pass(
                self.message_passing_layer,
                v_x,
                e_v_x,
                ors,
            ),
            vector_features,
            edge_vector_features,
            orientations,
        ), "Message passing should be rotation equivariant"

        # attention-based model should also be equivariant,
        # but lower sensitivity should be used due to numerical instability
        # with the softmax in the attention
        assert is_rotation_equivariant(
            lambda v_x, e_v_x, ors: vector_forward_pass(
                self.message_passing_layer_attn,
                v_x,
                e_v_x,
                ors,
            ),
            vector_features,
            edge_vector_features,
            orientations,
            atol=1e-2,
        ), "Message passing should be rotation equivariant"
