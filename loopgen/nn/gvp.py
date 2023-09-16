"""
Implements the GeometricVectorPerceptron from Dror et al. (2021, https://arxiv.org/pdf/2106.03843.pdf).
"""

from typing import Tuple, Union, List, Optional, Any, Sequence, Literal

from abc import ABC, abstractmethod

import torch
import einops
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, Aggregation, to_hetero
from torch_geometric.typing import PairTensor, Adj
from torch_scatter import scatter_softmax
from scipy.special import comb


def unflatten_vector_features(
    vector_features: torch.Tensor, vector_dim_size: int
) -> torch.Tensor:
    """
    Unflattens a tensor of vector features by reshaping
    the last dimension into two new dimensions, where the final dimension
    size is determined by the argument `vector_dim_size`.
    """

    vector_feature_shape = vector_features.shape[:-1] + (-1, vector_dim_size)
    unflattened_vector_features = vector_features.reshape(vector_feature_shape)

    return unflattened_vector_features


def separate_gvp_features(
    features: torch.Tensor,
    num_vector_features: int,
    vector_dim_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Separates scalar and vector features using the provided number of
    vector features and the provided vector dimensionality. This function
    always assumes that vector features are at the end of the feature tensor.
    """
    vec_feature_start_index = -num_vector_features * vector_dim_size
    scalar_features = features[:, :vec_feature_start_index]
    vector_features = unflatten_vector_features(
        features[:, vec_feature_start_index:], vector_dim_size
    )
    return scalar_features, vector_features


def combine_gvp_features(
    scalar_features: torch.Tensor, vector_features: torch.Tensor
) -> torch.Tensor:
    """Combines scalar and vector features into a single tensor."""
    flat_vector_features = vector_features.flatten(start_dim=-2)
    return torch.cat([scalar_features, flat_vector_features], dim=-1)


class GeometricVectorPerceptron(nn.Module):

    """
    Implements the geometric vector perceptron from Dror et al. (2021, https://arxiv.org/pdf/2106.03843.pdf).

    `forward()` takes a (..., K) tensor of scalar features and a (..., P, 3) tensor of
    geometric vector features. Transformations performed on the geometric vector
    features are rotation-equivariant.
    """

    def __init__(
        self,
        in_scalar_channels: int,
        out_scalar_channels: int,
        in_vector_channels: int,
        out_vector_channels: int,
        scalar_activation: nn.Module = nn.ReLU(),
        vector_activation: nn.Module = nn.Sigmoid(),
    ):
        super().__init__()

        self._in_scalar_channels = in_scalar_channels
        self._out_scalar_channels = out_scalar_channels
        self._in_vector_channels = in_vector_channels
        self._out_vector_channels = out_vector_channels

        h_dim = max(in_vector_channels, out_vector_channels)

        self._W_vec_for_scalar = nn.Linear(in_vector_channels, h_dim, bias=False)
        self._W_vec_for_vec = nn.Linear(
            in_vector_channels, out_vector_channels, bias=False
        )

        self._scalar_non_linearity = nn.Sequential(
            nn.Linear(h_dim + in_scalar_channels, out_scalar_channels),
            scalar_activation,
        )
        self._vector_non_linearity = nn.Sequential(
            nn.Linear(out_vector_channels + in_scalar_channels, out_vector_channels),
            vector_activation,
        )

    def forward(self, features: PairTensor) -> PairTensor:
        """
        Forward pass of the GVP, performing SO(3)-invariant transformations
        on the scalar features and SO(3)-equivariant transformations on the vector features.
        """
        scalar_features, vector_features = features

        vectors_for_scalars = self._W_vec_for_scalar(
            vector_features.transpose(-2, -1)
        ).transpose(-2, -1)
        vectors_for_vectors = self._W_vec_for_vec(
            vector_features.transpose(-2, -1)
        ).transpose(-2, -1)

        norms_for_scalars = torch.linalg.norm(vectors_for_scalars, dim=-1)
        norms_for_vectors = torch.linalg.norm(vectors_for_vectors, dim=-1)

        update_features_for_scalars = torch.cat(
            [scalar_features, norms_for_scalars], dim=-1
        )
        update_features_for_vectors = torch.cat(
            [scalar_features, norms_for_vectors], dim=-1
        )

        new_scalar_features = self._scalar_non_linearity(update_features_for_scalars)

        new_vector_features = (
            self._vector_non_linearity(update_features_for_vectors).unsqueeze(-1)
            * vectors_for_vectors
        )

        return new_scalar_features, new_vector_features

    def reset_parameters(self):
        """Resets learnable parameters."""
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    @property
    def in_scalar_channels(self):
        """Number of input scalar features."""
        return self._in_scalar_channels

    @property
    def out_scalar_channels(self):
        """Number of output scalar features."""
        return self._out_scalar_channels

    @property
    def in_vector_channels(self):
        """Number of input vector features."""
        return self._in_vector_channels

    @property
    def out_vector_channels(self):
        """Number of output vector features."""
        return self._out_vector_channels


class GVPDropout(nn.Module):

    """
    Dropout for a geometric vector perceptron.

    Dropout on scalar features works as normal, zero-ing
    components individually at random. Dropout on vector
    features zeroes entire 3-D vectors at once.
    """

    def __init__(self, p: float, vector_channels: int):
        super().__init__()

        self._dropout = nn.Dropout(p)
        self.register_buffer(
            "_vector_feature_indicator", torch.ones((vector_channels, 1))
        )

    def forward(self, features: PairTensor) -> PairTensor:
        """Performs dropout on a tuple of scalar and vector features."""
        scalar_features, vector_features = features

        dropout_scalar_features = self._dropout(scalar_features)

        # for vector feature dropout, run dropout on an
        # indicator tensor of the same shape as the vector features
        # and multiply the vector features by the result
        num_extra_vector_dims = len(vector_features.shape) - len(
            self._vector_feature_indicator.shape
        )
        vector_dropout_indicator = self._vector_feature_indicator.view(
            (1,) * num_extra_vector_dims + self._vector_feature_indicator.shape
        ).expand(*(vector_features.shape[:-2] + (2 * (-1,))))

        dropout_vector_features = (
            self._dropout(vector_dropout_indicator) * vector_features
        )

        return dropout_scalar_features, dropout_vector_features


class GVPLayerNorm(nn.Module):

    """
    LayerNorm for geometric vector perceptrons. Uses a standard LayerNorm for scalar features,
    and normalises the matrix of vector features to have frobenius norm K,
    where K is the number of vector features.
    """

    def __init__(self, scalar_channels: int, vector_channels: int):
        super().__init__()

        self._scalar_layer_norm = nn.LayerNorm(scalar_channels)
        self.register_buffer(
            "_sqrt_vector_channels", torch.sqrt(torch.as_tensor(vector_channels))
        )

    def forward(self, features: PairTensor) -> PairTensor:
        """Performs a layer normalisation for a tuple of scalar and vector features."""
        scalar_features, vector_features = features

        normalised_scalar_features = self._scalar_layer_norm(scalar_features)
        vector_feature_norm = torch.linalg.norm(
            vector_features, dim=(-2, -1), keepdim=True
        )
        normalised_vector_features = torch.where(
            vector_feature_norm > 0.0,
            self._sqrt_vector_channels * vector_features / vector_feature_norm,
            vector_features,
        )

        return normalised_scalar_features, normalised_vector_features


class GVPMessage(nn.Module):

    """
    Transforms scalar and vector features via sequential GVPs, applied identically to each pair
    of message scalar/vector features.
    """

    def __init__(
        self,
        message_scalar_channels: int,
        message_vector_channels: int,
        num_gvps: int,
        scalar_activation: Union[nn.Module, Sequence[nn.Module]] = nn.ReLU(),
        vector_activation: Union[nn.Module, Sequence[nn.Module]] = nn.Sigmoid(),
    ):
        super().__init__()

        self._message_scalar_channels = message_scalar_channels
        self._message_vector_channels = message_vector_channels

        if isinstance(scalar_activation, nn.Module):
            scalar_activations = [scalar_activation] * num_gvps
        elif isinstance(scalar_activation, Sequence):
            scalar_activations = scalar_activation
        else:
            raise ValueError(
                'Argument "scalar_activation" should be an nn.Module or a sequence of nn.Module'
            )

        if isinstance(vector_activation, nn.Module):
            vector_activations = [vector_activation] * num_gvps
        elif isinstance(vector_activation, Sequence):
            vector_activations = vector_activation
        else:
            raise ValueError(
                'Argument "vector_activation" should be an nn.Module or a sequence of nn.Module'
            )

        gvp_layers = []
        for i in range(num_gvps):
            gvp = GeometricVectorPerceptron(
                self._message_scalar_channels,
                self._message_scalar_channels,
                self._message_vector_channels,
                self._message_vector_channels,
                scalar_activations[i],
                vector_activations[i],
            )
            gvp_layers.append(gvp)

        self._message_gvp_layers = nn.Sequential(*gvp_layers)

    def forward(
        self,
        message_scalar_features: torch.Tensor,
        message_vector_features: torch.Tensor,
    ) -> PairTensor:
        """Passes messages through the GVP layers."""
        (
            message_scalar_features,
            message_vector_features,
        ) = self._message_gvp_layers((message_scalar_features, message_vector_features))

        return message_scalar_features, message_vector_features


class GVPUpdate(nn.Module):

    """
    Updates features using GVP transformations of the message features, with a residual connection
    to the current features. Input dimensionalities are "input" dims, and output dimensionalities
    are "state" dims in the constructor arguments.
    """

    def __init__(
        self,
        state_scalar_channels: int,
        in_scalar_channels: int,
        state_vector_channels: int,
        in_vector_channels: int,
        num_gvps: int,
        dropout: float,
        scalar_activation: Union[nn.Module, Sequence[nn.Module]] = nn.ReLU(),
        vector_activation: Union[nn.Module, Sequence[nn.Module]] = nn.Sigmoid(),
    ):
        super().__init__()

        self._state_scalar_channels = state_scalar_channels
        self._state_vector_channels = state_vector_channels

        self._dropout_1 = GVPDropout(dropout, in_vector_channels)
        self._dropout_2 = GVPDropout(dropout, state_vector_channels)

        self._gvp_layer_norm_1 = GVPLayerNorm(in_scalar_channels, in_vector_channels)
        self._gvp_layer_norm_2 = GVPLayerNorm(
            state_scalar_channels, state_vector_channels
        )

        if isinstance(scalar_activation, nn.Module):
            scalar_activations = [scalar_activation] * num_gvps
        elif isinstance(scalar_activation, Sequence):
            scalar_activations = scalar_activation
        else:
            raise ValueError(
                'Argument "scalar_activation" should be an nn.Module or a sequence of nn.Module'
            )

        if isinstance(vector_activation, nn.Module):
            vector_activations = [vector_activation] * num_gvps
        elif isinstance(vector_activation, Sequence):
            vector_activations = vector_activation
        else:
            raise ValueError(
                'Argument "vector_activation" should be an nn.Module or a sequence of nn.Module'
            )

        # first layer transforms message feature dim to state feature dim, others keep dim constant

        layers = [
            GeometricVectorPerceptron(
                in_scalar_channels,
                state_scalar_channels,
                in_vector_channels,
                state_vector_channels,
                scalar_activations[0],
                vector_activations[0],
            )
        ]
        for i in range(1, num_gvps):
            gvp = GeometricVectorPerceptron(
                state_scalar_channels,
                state_scalar_channels,
                state_vector_channels,
                state_vector_channels,
                scalar_activations[i],
                vector_activations[i],
            )
            layers.append(gvp)

        self._update_gvp_layers = nn.Sequential(*layers)

    @property
    def state_scalar_channels(self) -> int:
        """Number of output scalar features."""
        return self._state_scalar_channels

    @property
    def state_vector_channels(self) -> int:
        """Number of output vector features."""
        return self._state_vector_channels

    def forward(
        self,
        scalar_features: torch.Tensor,
        message_scalar_features: torch.Tensor,
        vector_features: torch.Tensor,
        message_vector_features: torch.Tensor,
    ) -> PairTensor:
        """
        Performs:
            1. Layer normalisation
            2. Dropout
            3. GVPs
            4. Dropout
            5. Residual connection
            6. Layer normalisation
        """
        message_scalar_features, message_vector_features = self._gvp_layer_norm_1(
            (message_scalar_features, message_vector_features)
        )

        dropped_out_messages = self._dropout_1(
            (message_scalar_features, message_vector_features)
        )

        updated_scalar_features, updated_vector_features = self._dropout_2(
            self._update_gvp_layers(dropped_out_messages)
        )

        updated_scalar_features += scalar_features
        updated_vector_features += vector_features

        (
            normed_updated_scalar_features,
            normed_updated_vector_features,
        ) = self._gvp_layer_norm_2((updated_scalar_features, updated_vector_features))

        return normed_updated_scalar_features, normed_updated_vector_features


# supported GVP attention types
GVPAttentionTypes = Literal["norm", "flatten"]


class GVPAttention(ABC, nn.Module):

    """
    GVP attention layers calculate attention weights
    between sets of queries and keys, where each query and key
    element has both scalar and vector features.
    """

    @abstractmethod
    def __init__(
        self,
        query_scalar_channels: int,
        query_vector_channels: int,
        key_scalar_channels: int,
        key_vector_channels: int,
        num_heads: int = 1,
        vector_dim_size: int = 3,
    ):
        super().__init__()

        if not (
            query_scalar_channels % num_heads == 0
            and query_vector_channels % num_heads == 0
            and key_scalar_channels % num_heads == 0
            and key_vector_channels % num_heads == 0
        ):
            raise ValueError(
                "Scalar dim or vector dim not divisible by number of heads"
            )

        self._query_scalar_dim = query_scalar_channels
        self._query_vector_dim = query_vector_channels

        self._key_scalar_dim = key_scalar_channels
        self._key_vector_dim = key_vector_channels

        self.register_buffer(
            "_sqrt_scalar_dim", torch.sqrt(torch.as_tensor(self._query_scalar_dim))
        )
        self.register_buffer(
            "_sqrt_vector_dim", torch.sqrt(torch.as_tensor(self._query_vector_dim))
        )

        self._num_heads = num_heads
        self._vector_dim_size = vector_dim_size

        self._scalar_dim_per_head = self._query_scalar_dim // self._num_heads
        self._vector_dim_per_head = self._query_vector_dim // self._num_heads

    @property
    def num_heads(self):
        """Number of attention heads."""
        return self._num_heads

    @abstractmethod
    def forward(
        self,
        query_x: torch.Tensor,
        query_vector_x: torch.Tensor,
        query_orientations: torch.Tensor,
        key_x: torch.Tensor,
        key_vector_x: torch.Tensor,
        key_orientations: torch.Tensor,
        softmax_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        For a graph with scalar/vector features for each node and scalar/vector messages for each edge,
        outputs attention weights of shape (M, H) where M is the number of edges in the graph
        and H is the number of heads in the multi-head attention. The user is required to specify groups of elements
        whose attention weights will be softmaxed. In GNNs, `softmax_index` is typically a vector of target nodes
        for each entry along the first dimension of `query_x/key_x`.

        :param query_x: Scalar features for query nodes (i.e. target nodes).
        :param query_vector_x: Vector features for query nodes (i.e. target nodes).
        :param query_orientations: Orthonormal orientation matrices to be used as rotation-invariant
            change of basis matrices, one for each query node.
        :param key_x: Scalar features for key nodes (i.e. source nodes).
        :param key_vector_x: Vector features for key nodes (i.e. source nodes).
        :param key_orientations: Orientation matrices for key nodes.
        :param softmax_index: Indices which define groupings of attention weights which will be softmax normalised.
        :return: (M, H) tensor of attention weights.
        """

        pass


class FlattenDotProductAttention(GVPAttention):

    """
    Calculates attention weights of the following form:

    `a_ij = dot(concat(s_i, flatten(V_i @ O_i)), concat(s_j, flatten(V_j @ O_j)))`,
    where `s` is a vector of scalar features, `V` is a matrix of vector features,
    and `O` is the residue's orientation matrix.
    """

    def __init__(
        self,
        query_scalar_channels: int,
        query_vector_channels: int,
        key_scalar_channels: int,
        key_vector_channels: int,
        num_heads: int = 1,
        vector_dim_size: int = 3,
    ):
        super().__init__(
            query_scalar_channels,
            query_vector_channels,
            key_scalar_channels,
            key_vector_channels,
            num_heads,
            vector_dim_size,
        )

        self.Q_scalar = nn.Linear(
            self._query_scalar_dim, self._query_scalar_dim, bias=False
        )
        self.K_scalar = nn.Linear(
            self._key_scalar_dim, self._query_scalar_dim, bias=False
        )

        self.Q_vector = nn.Linear(
            self._query_vector_dim, self._query_vector_dim, bias=False
        )
        self.K_vector = nn.Linear(
            self._key_vector_dim, self._query_vector_dim, bias=False
        )

    def forward(
        self,
        query_x: torch.Tensor,
        query_vector_x: torch.Tensor,
        query_orientations: torch.Tensor,
        key_x: torch.Tensor,
        key_vector_x: torch.Tensor,
        key_orientations: torch.Tensor,
        softmax_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates attention weights by first linearly transforming query and key features (both scalar and vector
        features), and then calculating an attention weight based on a dot product between concatenated query/key
        vectors, where the concatenated vectors are formed by generating SE(3)-invariant vector
        features with the orientation matrices and flattening and concatenating them with scalar features.
        """

        query = self.Q_scalar(query_x)
        key = self.K_scalar(key_x)

        vector_query = self.Q_vector(query_vector_x.transpose(-2, -1)).transpose(-2, -1)
        vector_key = self.K_vector(key_vector_x.transpose(-2, -1)).transpose(-2, -1)

        query_multihead = einops.rearrange(
            query, "b ... (h d) -> b ... h d", h=self._num_heads
        )
        key_multihead = einops.rearrange(
            key, "b ... (h d) -> b ... h d", h=self._num_heads
        )

        invar_vector_query_multihead = einops.rearrange(
            torch.matmul(vector_query, query_orientations),
            "b ... (h d) v -> b ... h d v",
            h=self._num_heads,
            v=self._vector_dim_size,
        )
        invar_vector_key_multihead = einops.rearrange(
            torch.matmul(vector_key, key_orientations),
            "b ... (h d) v -> b ... h d v",
            h=self._num_heads,
            v=self._vector_dim_size,
        )

        concat_query = torch.cat(
            [query_multihead, invar_vector_query_multihead.flatten(start_dim=-2)],
            dim=-1,
        )
        concat_key = torch.cat(
            [key_multihead, invar_vector_key_multihead.flatten(start_dim=-2)], dim=-1
        )

        unscaled_attn_weights = torch.sum(concat_query * concat_key, dim=-1)
        attn_weights = scatter_softmax(unscaled_attn_weights, softmax_index, dim=0)

        return attn_weights


class NormDotProductAttention(GVPAttention):

    """
    Calculates attention weights of the following form:

    `a_ij = dot(concat(s_i, row_norm(V_i)), concat(s_j, row_num(V_j)))`,
    where `s` is a vector of scalar features, `V` is a matrix of vector features,
    and `O` is the residue's orientation matrix.
    """

    def __init__(
        self,
        query_scalar_channels: int,
        query_vector_channels: int,
        key_scalar_channels: int,
        key_vector_channels: int,
        num_heads: int = 1,
        vector_dim_size: int = 3,
    ):
        super().__init__(
            query_scalar_channels,
            query_vector_channels,
            key_scalar_channels,
            key_vector_channels,
            num_heads,
            vector_dim_size,
        )

        self.Q_scalar = nn.Linear(
            self._query_scalar_dim, self._query_scalar_dim, bias=False
        )
        self.K_scalar = nn.Linear(
            self._key_scalar_dim, self._query_scalar_dim, bias=False
        )

        self.Q_vector = nn.Linear(
            self._query_vector_dim, self._query_vector_dim, bias=False
        )
        self.K_vector = nn.Linear(
            self._key_vector_dim, self._query_vector_dim, bias=False
        )

    def forward(
        self,
        query_x: torch.Tensor,
        query_vector_x: torch.Tensor,
        query_orientations: torch.Tensor,
        key_x: torch.Tensor,
        key_vector_x: torch.Tensor,
        key_orientations: torch.Tensor,
        softmax_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates attention weights by first linearly transforming query and key features (both scalar and vector
        features), and then calculating an attention weight based on a dot product between concatenated query/key
        vectors, where the concatenated vectors are formed by calculating the norm of the vector
        features with the orientation matrices and concatenating the norms with scalar features.
        """

        query = self.Q_scalar(query_x)
        key = self.K_scalar(key_x)

        vector_query = self.Q_vector(query_vector_x.transpose(-2, -1)).transpose(-2, -1)
        vector_key = self.K_vector(key_vector_x.transpose(-2, -1)).transpose(-2, -1)

        query_multihead = einops.rearrange(
            query, "b ... (h d) -> b ... h d", h=self._num_heads
        )
        key_multihead = einops.rearrange(
            key, "b ... (h d) -> b ... h d", h=self._num_heads
        )

        vector_query_norms_multihead = einops.rearrange(
            torch.linalg.norm(vector_query, dim=-1),
            "b ... (h d) -> b ... h d",
            h=self._num_heads,
        )
        vector_key_norms_multihead = einops.rearrange(
            torch.linalg.norm(vector_key, dim=-1),
            "b ... (h d) -> b ... h d",
            h=self._num_heads,
        )

        concat_query = torch.cat(
            [query_multihead, vector_query_norms_multihead], dim=-1
        )
        concat_key = torch.cat([key_multihead, vector_key_norms_multihead], dim=-1)

        unscaled_attn_weights = torch.sum(concat_query * concat_key, dim=-1)
        attn_weights = scatter_softmax(unscaled_attn_weights, softmax_index, dim=0)

        return attn_weights


class GVPAttentionStrategySelector:

    """
    Class for returning the correct GVPAttention class given an input string.
    """

    def __init__(
        self,
        query_scalar_dim: int,
        query_vector_dim: int,
        key_scalar_dim: int,
        key_vector_dim: int,
        num_heads: int = 1,
        vector_dim_size: int = 3,
    ):
        super().__init__()

        self._query_scalar_dim = query_scalar_dim
        self._query_vector_dim = query_vector_dim

        self._key_scalar_dim = key_scalar_dim
        self._key_vector_dim = key_vector_dim

        self._num_heads = num_heads
        self._vector_dim_size = vector_dim_size

    def get_layer(self, name: GVPAttentionTypes):
        """
        Returns the geometric attention layer specified by the string `name`.
        """

        if name == "norm":
            attn_class = NormDotProductAttention
        elif name == "flatten":
            attn_class = FlattenDotProductAttention
        else:
            raise ValueError(
                f"Argument name should be one of {GVPAttentionTypes.__args__}"
            )

        return attn_class(
            self._query_scalar_dim,
            self._query_vector_dim,
            self._key_scalar_dim,
            self._key_vector_dim,
            self._num_heads,
            self._vector_dim_size,
        )


class GVPMessagePassing(MessagePassing):

    """
    Performs message passing with a gated GVP architecture as described
    in Dror et al. (2021, https://arxiv.org/pdf/2106.03843.pdf).

    The forward() method of this takes a single rank 2 tensor of features,
    assuming that the vector features are flattened and appended to the **end**
    of the input feature vector `x`.
    """

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        edge_scalar_channels: int,
        edge_vector_channels: int,
        aggr: Union[str, List, Aggregation],
        num_message_gvps: int,
        num_update_gvps: int,
        dropout: float,
        attention_type: Optional[GVPAttentionTypes] = None,
        num_heads: int = 1,
        vector_dim_size: int = 3,
        **kwargs: Any,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self._scalar_channels = scalar_channels
        self._vector_channels = vector_channels
        self._edge_scalar_channels = edge_scalar_channels
        self._edge_vector_channels = edge_vector_channels

        self._message_layer = GVPMessage(
            self.message_scalar_channels,
            self.message_vector_channels,
            num_message_gvps,
        )

        self._node_update_layer = GVPUpdate(
            self._scalar_channels,
            self.message_scalar_channels,
            self._vector_channels,
            self.message_vector_channels,
            num_update_gvps,
            dropout,
        )

        if attention_type is not None:
            attn_selector = GVPAttentionStrategySelector(
                self._scalar_channels,
                self._vector_channels,
                self.message_scalar_channels,
                self.message_vector_channels,
                num_heads,
                vector_dim_size,
            )
            self._attention_layer = attn_selector.get_layer(attention_type)
        else:
            self._attention_layer = None

        self._vector_dim_size = vector_dim_size

    @property
    def message_scalar_channels(self) -> int:
        """
        The number of scalar channels in the message,
        equal to 2x the node scalar channels plus the edge scalar channels.
        """
        return self._scalar_channels * 2 + self._edge_scalar_channels

    @property
    def message_vector_channels(self) -> int:
        """
        The number of vector channels in the message,
        equal to 2x the node vector channels plus the edge vector channels.
        """
        return self._vector_channels * 2 + self._edge_vector_channels

    def reset_parameters(self):
        """Resets parameters for all layers."""
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        orientation_matrices_i: torch.Tensor,
        orientation_matrices_j: torch.Tensor,
        edge_index_i: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forms a message for each pair of connected nodes in the graph consisting
        of the source node features, target node features, and edge features,
        concatenating scalar and vector features separately.

        :param x_i: (M, P) tensor of target node features, with scalar features and vector features concatenated.
        :param x_j: (M, P) tensor of source node features, with scalar features and vector features concatenated.
        :param edge_attr: (M, K) tensor of edge features, with scalar features and vector features concatenated.
        :param orientation_matrices_i: (M, 3, 3) tensor of orientation matrices for target nodes
            (only used in attention calculation).
        :param orientation_matrices_j: (M, 3, 3) tensor of orientation matrices for source nodes
            (only used in attention calculation).
        :param edge_index_i: (M,) tensor of edge indices for target nodes (only used in attention calculation).
        :returns: (M, 2P + K) tensor of messages, with scalar features and vector features concatenated.
        """
        scalar_x_i, vector_x_i = separate_gvp_features(
            x_i,
            self._vector_channels,
            self._vector_dim_size,
        )
        scalar_x_j, vector_x_j = separate_gvp_features(
            x_j,
            self._vector_channels,
            self._vector_dim_size,
        )
        scalar_edge_attr, vector_edge_attr = separate_gvp_features(
            edge_attr, self._edge_vector_channels, self._vector_dim_size
        )

        initial_scalar_messages = torch.cat(
            [scalar_x_i, scalar_x_j, scalar_edge_attr], dim=-1
        )
        initial_vector_messages = torch.cat(
            [vector_x_i, vector_x_j, vector_edge_attr], dim=-2
        )

        scalar_message, vector_message = self._message_layer(
            initial_scalar_messages, initial_vector_messages
        )

        if self._attention_layer is None:
            return combine_gvp_features(scalar_message, vector_message)

        orientation_matrices_i = unflatten_vector_features(orientation_matrices_i, 3)
        orientation_matrices_j = unflatten_vector_features(orientation_matrices_j, 3)

        attn_weights = self._attention_layer(
            scalar_x_i,
            vector_x_i,
            orientation_matrices_i,
            scalar_message,
            vector_message,
            orientation_matrices_j,
            edge_index_i,
        )

        num_heads = self._attention_layer.num_heads
        scalar_message_multihead = einops.rearrange(
            scalar_message, "n (h d) -> n h d", h=num_heads
        )
        vector_message_multihead = einops.rearrange(
            vector_message, "n (h d) v -> n h d v", h=num_heads, v=self._vector_dim_size
        )

        weighted_scalar_message = einops.rearrange(
            attn_weights[..., None] * scalar_message_multihead,
            "n h d -> n (h d)",
            h=num_heads,
        )
        weighted_vector_message = einops.rearrange(
            attn_weights[..., None, None] * vector_message_multihead,
            "n h d v -> n (h d) v",
            h=num_heads,
        )

        return combine_gvp_features(weighted_scalar_message, weighted_vector_message)

    def update(
        self, inputs: torch.Tensor, x: Union[torch.Tensor, PairTensor]
    ) -> torch.Tensor:
        """
        Takes the combined messages as input and passes them - and the original node features - through
        the node update layer. Returns the updated node features.
        """
        aggr_scalar_message, aggr_vector_message = separate_gvp_features(
            inputs,
            self.message_vector_channels,
            self._vector_dim_size,
        )

        # if (source, target) feature tuples provided,
        # only use target node features (i.e. second element of the tuple), since
        # message contains information from the source node
        if isinstance(x, tuple):
            _, x = x

        scalar_x, vector_x = separate_gvp_features(
            x, self._vector_channels, self._vector_dim_size
        )

        updated_x, updated_vector_x = self._node_update_layer(
            scalar_x, aggr_scalar_message, vector_x, aggr_vector_message
        )

        updated_features = combine_gvp_features(updated_x, updated_vector_x)

        return updated_features

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_attr: torch.Tensor,
        orientation_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the geometric vector perceptron.

        :param x: (N, K) tensor of K features for a batch with N nodes, where vector features have
            been flattened and concatenated to the end of the scalar features.
        :param edge_index: (2, M) tensor of index pairs representing M directed edges.
        :param edge_attr: (M, P) tensor of P features for each edge in the batch, where vector features
            have been flattened and concatenated with scalar features.
        :param orientation_matrices: (N, 9) tensor of N flattened orthonormal orientation matrices, one for each node.
            Used to generate rotation-invariant coordinate representations in attention calculations.
        :return: Single node feature tensor with scalar and vector features concatenated.
        """

        updated_features = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            orientation_matrices=orientation_matrices,
        )

        return updated_features


class GVPN(nn.Module):
    """
    GVP message passing network, which
    contains node and edge feature embedding layers, as well
    as multiple GVPMessagePassing layers for message passing.
    """

    def __init__(
        self,
        in_scalar_channels: int,
        hidden_scalar_channels: int,
        in_vector_channels: int,
        hidden_vector_channels: int,
        in_edge_scalar_channels: int,
        hidden_edge_scalar_channels: int,
        in_edge_vector_channels: int,
        hidden_edge_vector_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggr: Union[str, List, Aggregation] = "add",
        attention_type: Optional[GVPAttentionTypes] = None,
        num_heads: int = 4,
        share_params: bool = False,
        num_message_gvps: int = 3,
        num_update_gvps: int = 2,
        vector_dim_size: int = 3,
    ):
        """
        :param in_scalar_channels: Number of input scalar node features.
        :param hidden_scalar_channels: Number of hidden/output scalar node features.
        :param in_vector_channels: Number of input vector node features (each feature is a vector).
        :param hidden_vector_channels: Number of hidden/output vector node features (each feature is a vector).
        :param in_edge_scalar_channels: Number of input scalar edge features.
        :param hidden_edge_scalar_channels: Number of hidden/output scalar edge features.
        :param in_edge_vector_channels: Number of input vector edge features (each feature is a vector).
        :param hidden_edge_vector_channels: Number of hidden/output vector edge features (each feature is a vector).
        :param num_layers: Number of message passing layers.
        :param dropout: Dropout probability.
        :param aggr: Aggregation method to use for message passing.
        :param attention_type: The type of attention to use for message passing (defaults to None, i.e. no attention).
        :param share_params: Whether to share parameters across message passing layers.
        :param num_message_gvps: Number of GVP layers to use in the message function(s).
        :param num_update_gvps: Number of GVP layers to use in the update function(s).
        :param vector_dim_size: The dimensionality of the vector features (i.e. the size of each vector feature).
        :param num_heads: Number of attention heads to use (if attention is used).
        """
        super().__init__()

        self._hidden_scalar_channels = hidden_scalar_channels
        self._hidden_vector_channels = hidden_vector_channels
        self._hidden_edge_scalar_channels = hidden_edge_scalar_channels
        self._hidden_edge_vector_channels = hidden_edge_vector_channels
        self._num_layers = num_layers
        self._dropout = dropout
        self._aggr = aggr
        self._attention_type = attention_type
        self._num_heads = num_heads
        self._num_message_gvps = num_message_gvps
        self._num_update_gvps = num_update_gvps
        self._share_params = bool(share_params)
        self._vector_dim_size = vector_dim_size

        self._node_embedding = GeometricVectorPerceptron(
            in_scalar_channels,
            hidden_scalar_channels,
            in_vector_channels,
            hidden_vector_channels,
        )

        self._edge_embedding = GeometricVectorPerceptron(
            in_edge_scalar_channels,
            hidden_edge_scalar_channels,
            in_edge_vector_channels,
            hidden_edge_vector_channels,
        )

        self._message_passing_layers = self._create_message_passing_layers()

    @property
    def message_passing_layers(self) -> nn.ModuleList:
        """The constituent message passing layers."""
        return self._message_passing_layers

    def to_hetero(
        self,
        example_graph: HeteroData,
        aggr: str = "sum",
        node_scalar_feature_name: str = "x",
        node_vector_feature_name: str = "vector_x",
        edge_scalar_feature_name: str = "edge_attr",
        edge_vector_feature_name: str = "vector_edge_attr",
        **kwargs: Any,  # kwargs to torch_geometric.nn.to_hetero()
    ):
        """
        Converts the GVP message passing network into a heterogeneous MPN.
        Compared with the `torch_geometric` implementation `to_hetero()`, this function
        generates new node and edge embedding layers based on the feature dimensionalities
        of each node and edge type, allowing for the use of different node and edge
        feature dimensionalities in the heterogeneous graph.

        :param example_graph: An example heterogeneous graph.
        :param aggr: Aggregation method to use for message passing.
        :param node_scalar_feature_name: The name of the node scalar feature attribute.
        :param node_vector_feature_name: The name of the node vector feature attribute.
        :param edge_scalar_feature_name: The name of the edge scalar feature attribute.
        :param edge_vector_feature_name: The name of the edge vector feature attribute.
        :param kwargs: Keyword arguments to torch_geometric.nn.to_hetero().
        :returns: A heterogeneous GVP message passing network.
        """

        if not isinstance(example_graph, HeteroData):
            raise TypeError(
                f"Argument 'example_graph' should be a torch_geometric HeteroData object."
            )

        node_types, edge_types = example_graph.metadata()

        # make new node embedding layers
        new_node_embedding_dict = nn.ModuleDict()
        for node_type in node_types:
            node_scalar_channels = getattr(
                example_graph[node_type], node_scalar_feature_name
            ).shape[-1]
            node_vector_channels = getattr(
                example_graph[node_type], node_vector_feature_name
            ).shape[-2]

            new_node_embedding = GeometricVectorPerceptron(
                in_scalar_channels=node_scalar_channels,
                out_scalar_channels=self._hidden_scalar_channels,
                in_vector_channels=node_vector_channels,
                out_vector_channels=self._hidden_vector_channels,
            )

            new_node_embedding_dict[node_type] = new_node_embedding

        # make new edge embedding layers
        new_edge_embedding_dict = nn.ModuleDict()
        for edge_type in edge_types:
            edge_scalar_channels = getattr(
                example_graph[edge_type], edge_scalar_feature_name
            ).shape[-1]
            edge_vector_channels = getattr(
                example_graph[edge_type], edge_vector_feature_name
            ).shape[-2]

            new_edge_embedding = GeometricVectorPerceptron(
                in_scalar_channels=edge_scalar_channels,
                out_scalar_channels=self._hidden_edge_scalar_channels,
                in_vector_channels=edge_vector_channels,
                out_vector_channels=self._hidden_edge_vector_channels,
            )

            new_edge_embedding_dict["__".join(edge_type)] = new_edge_embedding

        hetero_gvpn = to_hetero(self, (node_types, edge_types), aggr, **kwargs)
        hetero_gvpn._node_embedding = new_node_embedding_dict
        hetero_gvpn._edge_embedding = new_edge_embedding_dict

        return hetero_gvpn

    def _create_message_passing_layers(self) -> Union[nn.Module, nn.ModuleList]:
        """
        Stacks multiple GVPMessagePassingModule objects into a ModuleList.
        """

        if self._share_params is True:
            message_passing_layers = GVPMessagePassing(
                scalar_channels=self._hidden_scalar_channels,
                vector_channels=self._hidden_vector_channels,
                edge_scalar_channels=self._hidden_edge_scalar_channels,
                edge_vector_channels=self._hidden_edge_vector_channels,
                aggr=self._aggr,
                num_message_gvps=self._num_message_gvps,
                num_update_gvps=self._num_update_gvps,
                dropout=self._dropout,
                attention_type=self._attention_type,
                num_heads=self._num_heads,
                vector_dim_size=self._vector_dim_size,
            )

        else:
            message_passing_layers = nn.ModuleList(
                [
                    GVPMessagePassing(
                        scalar_channels=self._hidden_scalar_channels,
                        vector_channels=self._hidden_vector_channels,
                        edge_scalar_channels=self._hidden_edge_scalar_channels,
                        edge_vector_channels=self._hidden_edge_vector_channels,
                        aggr=self._aggr,
                        num_message_gvps=self._num_message_gvps,
                        num_update_gvps=self._num_update_gvps,
                        dropout=self._dropout,
                        attention_type=self._attention_type,
                        num_heads=self._num_heads,
                        vector_dim_size=self._vector_dim_size,
                    )
                    for _ in range(self._num_layers)
                ]
            )

        return message_passing_layers

    def get_scalar_node_features(
        self, features: torch.Tensor, num_features: int
    ) -> torch.Tensor:
        """
        Gets scalar features from a tensor of all features. Called before the edge update to
        extract SE(3)-invariant features.
        """
        scalar_features = features[..., : self._hidden_scalar_channels]
        return scalar_features

    def message_passing(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        orientation_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs message passing using the input node/edge features and returns the updated node features.
        Assumes node and edge features are rank 2 tensors containing scalar and flattened vector
        features, and have been embedded into the correct hidden layer feature dimensionality.
        """
        if self._share_params:
            for _ in range(self._num_layers):
                node_features = self._message_passing_layers(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_features,
                    orientation_matrices=orientation_matrices,
                )
        else:
            for layer in self._message_passing_layers:
                node_features = layer(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_features,
                    orientation_matrices=orientation_matrices,
                )

        return node_features

    def forward(
        self,
        scalar_x: torch.Tensor,
        vector_x: torch.Tensor,
        edge_index: Adj,
        scalar_edge_attr: torch.Tensor,
        vector_edge_attr: torch.Tensor,
        orientation_matrices: torch.Tensor,
    ) -> PairTensor:
        """
        Applies the node/edge embeddings and the message passing layers, performing SO(3)-invariant
        updates on the scalar features and SO(3)-equivariant updates on the vector features.

        :param scalar_x: Tensor containing SO(3)-invariant scalar node features.
        :param vector_x: Tensor containing SO(3)-equivariant vector node features.
        :param edge_index: Edge index tensor of shape (2, M) for a graph with M edges.
        :param scalar_edge_attr: Tensor containing SO(3)-invariant scalar edge features.
        :param vector_edge_attr: Tensor containing SO(3)-equivariant vector edge features.
        :param orientation_matrices: Tensor containing the orientation matrices for each node.
        :returns: A tuple containing the updated scalar and vector node features.
        """
        scalar_x, vector_x = self._node_embedding((scalar_x, vector_x))
        scalar_edge_attr, vector_edge_attr = self._edge_embedding(
            (scalar_edge_attr, vector_edge_attr)
        )

        if isinstance(vector_x, tuple):
            vector_x = tuple(v.flatten(start_dim=-2) for v in vector_x)
            vector_edge_attr = tuple(v.flatten(start_dim=-2) for v in vector_edge_attr)
            orientation_matrices = tuple(
                o.flatten(start_dim=-2) for o in orientation_matrices
            )

            node_features = tuple(
                [torch.cat([x, v_x], dim=-1) for x, v_x in zip(scalar_x, vector_x)]
            )
            edge_features = tuple(
                [
                    torch.cat([e, v_e], dim=-1)
                    for e, v_e in zip(scalar_edge_attr, vector_edge_attr)
                ]
            )
        else:
            vector_x = vector_x.flatten(start_dim=-2)
            vector_edge_attr = vector_edge_attr.flatten(start_dim=-2)
            orientation_matrices = orientation_matrices.flatten(start_dim=-2)

            node_features = torch.cat([scalar_x, vector_x], dim=-1)
            edge_features = torch.cat([scalar_edge_attr, vector_edge_attr], dim=-1)

        node_features = self.message_passing(
            node_features, edge_index, edge_features, orientation_matrices
        )

        scalar_x, vector_x = separate_gvp_features(
            node_features, self._hidden_vector_channels, self._vector_dim_size
        )

        return scalar_x, vector_x
