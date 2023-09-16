"""
Contains score prediction networks for CDR backbone diffusion model,
which predict 3-D score vectors (gradient of the forward process log density)
for each node in a graph.
"""

from __future__ import annotations

from typing import Union, List, Optional, Any, Dict

import torch
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import Aggregation, to_hetero

from einops.layers.torch import Rearrange

from .types import Score, VectorFeatureGraph, ProteinGraph
from ..nn import GVPN, GVPAttentionTypes


class GVPR3ScorePredictor(nn.Module):
    """
    Module that uses a geometric vector perceptron to approximate
    scores (gradient of log density) in an R3 diffusion model.

    By default, the model outputs a single SO(3)-equivariant 3-D vector
    for the score.
    """

    def __init__(
        self,
        example_batch: ProteinGraph,
        out_vector_channels: int = 1,
        hidden_scalar_channels: int = 128,
        hidden_vector_channels: int = 64,
        hidden_edge_scalar_channels: int = 64,
        hidden_edge_vector_channels: int = 32,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggr: Union[str, List, Aggregation] = "sum",
        attention_type: Optional[GVPAttentionTypes] = None,
        use_scaling_gate: bool = True,
        share_params: bool = True,
        vector_dim_size: int = 3,
        **kwargs: Any,
    ):
        """
        :param example_batch: Example graph batch object (Data/HeteroData) to infer input feature dimensions from.
        :param out_vector_channels: Number of output vector features. Defaults to 1 for the single 3-D
            SO(3)-equivariant translation score.
        :param hidden_scalar_channels: Dimensionality of the hidden scalar features.
        :param hidden_vector_channels: Dimensionality of the hidden vector features.
        :param hidden_edge_scalar_channels: Dimensionality of the hidden scalar edge features.
        :param hidden_edge_vector_channels: Dimensionality of the hidden vector edge features.
        :param num_layers: Number of message passing layers.
        :param dropout: Dropout rate.
        :param aggr: Aggregation scheme used in message passing.
        :param attention_type: Type of attention layer used in the graph network (GVPN).
        :param use_scaling_gate: Uses a final layer after message passing to output a positive scalar,
            which is used to scale the output vector features. This is potentially
            useful to allow the network to learn norm information about its outputs.
        :param share_params: Whether to share parameters across message passing layers.
        :param vector_dim_size: Dimensionality of the vectors used as vector features
            (not the number of vector features).
        :param kwargs: Additional keyword arguments passed to the graph network (GVPN).
        """

        super().__init__()

        self._hidden_scalar_channels = hidden_scalar_channels
        self._hidden_vector_channels = hidden_vector_channels
        self._out_vector_channels = out_vector_channels
        self._use_scaling_gate = bool(use_scaling_gate)

        # get feature dimensions depending on whether graph is heterogeneous or homogeneous
        if isinstance(example_batch, HeteroData):
            node_storage = example_batch["ligand"]
            in_scalar_channels = node_storage.x.shape[-1]
            in_vector_channels = node_storage.vector_x.shape[-2]
            edge_storage = example_batch[("ligand", "ligand")]
            in_edge_scalar_channels = edge_storage.edge_attr.shape[-1]
            in_edge_vector_channels = edge_storage.vector_edge_attr.shape[-2]
        else:
            in_scalar_channels = example_batch.x.shape[-1]
            in_vector_channels = example_batch.vector_x.shape[-2]
            in_edge_scalar_channels = example_batch.edge_attr.shape[-1]
            in_edge_vector_channels = example_batch.vector_edge_attr.shape[-2]

        graph_network = GVPN(
            in_scalar_channels,
            hidden_scalar_channels,
            in_vector_channels,
            hidden_vector_channels,
            in_edge_scalar_channels,
            hidden_edge_scalar_channels,
            in_edge_vector_channels,
            hidden_edge_vector_channels,
            num_layers=num_layers,
            dropout=dropout,
            aggr=aggr,
            attention_type=attention_type,
            share_params=share_params,
            vector_dim_size=vector_dim_size,
            **kwargs,
        )

        self._graph_network = graph_network
        self._vector_pred_layer = None
        self._vector_scaling_gate = None

        self._create_prediction_layers()

        self._is_hetero = False

        if isinstance(example_batch, HeteroData):
            self.as_hetero(example_batch, aggr)

    def as_hetero(
        self,
        example_graph: HeteroData,
        aggr: str = "sum",
        **kwargs: Any,
    ):
        """
        Modifies the underlying GVPN network to be
        compatible with a heterogeneous graph, with
        separate message passing parameters for each edge type.
        """

        self._vector_pred_layer = to_hetero(
            self._vector_pred_layer, example_graph.metadata(), aggr=aggr
        )

        if self._use_scaling_gate:
            self._vector_scaling_gate = to_hetero(
                self._vector_scaling_gate, example_graph.metadata(), aggr=aggr
            )

        self._graph_network = self._graph_network.to_hetero(example_graph, aggr)

        if self._is_hetero is True:
            return

        if not isinstance(example_graph, HeteroData):
            raise TypeError(
                f"Argument 'example_graph' should be a torch_geometric HeteroData object."
            )

        self._is_hetero = True

    @property
    def is_hetero(self):
        """Whether the network is heterogeneous."""
        return self._is_hetero

    def _create_prediction_layers(self):
        """Generates the prediction layers to be used to generate outputs after message passing."""

        # add a final prediction layer for SO(3)-equivariant vector features.
        vector_pred_layer = nn.Sequential(
            Rearrange("... n d -> ... d n"),
            nn.Linear(
                self._hidden_vector_channels, self._out_vector_channels, bias=False
            ),
            Rearrange("... d n -> ... n d"),
        )

        self._vector_pred_layer = vector_pred_layer

        if self._use_scaling_gate:
            # the features used to calculate the scaling factors are the scalar features
            # concatenated with the norms of the vector features
            num_gate_channels = (
                self._hidden_scalar_channels + self._hidden_vector_channels
            )
            self._vector_scaling_gate = nn.Sequential(
                nn.Linear(num_gate_channels, num_gate_channels),
                nn.ReLU(),
                nn.Linear(num_gate_channels, 1),
                nn.Softplus(),
            )

    def _predict_scores(
        self,
        scalar_features: Union[torch.Tensor, Dict[str, torch.Tensor]],
        vector_features: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Union[Score, Dict[str, Score]]:
        """
        Gets the translation scores from the scalar features and vector features
        outputted after message passing.

        :param scalar_features: Scalar features outputted after message passing.
        :param vector_features: Vector features outputted after message passing.
        :returns: Tensor containing the predicted translation scores.
        """
        translation_scores = self._vector_pred_layer(vector_features)

        if self._use_scaling_gate:
            if self._is_hetero:
                gate_features = {
                    node_type: torch.cat(
                        [
                            scalar_features[node_type],
                            torch.linalg.norm(vector_features[node_type], dim=-1),
                        ],
                        dim=-1,
                    )
                    for node_type in scalar_features
                }
            else:
                gate_features = torch.cat(
                    [
                        scalar_features,
                        torch.linalg.norm(vector_features, dim=-1),
                    ],
                    dim=-1,
                )

            translation_scale_factors = self._vector_scaling_gate(gate_features)

            if self._is_hetero:
                translation_scores = {
                    node_type: v_x * translation_scale_factors[node_type].unsqueeze(-1)
                    for node_type, v_x in translation_scores.items()
                }
            else:
                translation_scores *= translation_scale_factors.unsqueeze(-1)

        return translation_scores

    def forward(
        self, graph: VectorFeatureGraph
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns an SO(3)-equivariant (N, 3) tensor containing the translation score for each node in the graph.

        :param graph: Vector feature graph representation of a protein or protein complex.
        :returns: Tensor of translation score vectors.
        """
        if self._is_hetero:
            x = graph.x_dict
            vector_x = graph.vector_x_dict
            edge_index = graph.edge_index_dict
            edge_attr = graph.edge_attr_dict
            vector_edge_attr = graph.vector_edge_attr_dict
            orientations = graph.orientations_dict
        else:
            x = graph.x
            vector_x = graph.vector_x
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr
            vector_edge_attr = graph.vector_edge_attr
            orientations = graph.orientations

        node_scalar_embeddings, node_vector_embeddings = self._graph_network(
            x,
            vector_x,
            edge_index,
            edge_attr,
            vector_edge_attr,
            orientations,
        )

        return self._predict_scores(node_scalar_embeddings, node_vector_embeddings)


class GVPSE3ScorePredictor(GVPR3ScorePredictor):

    """
    Module that uses a geometric vector perceptron to approximate
    scores (gradient of log density) in an SE(3) diffusion model.

    This class is an extension of GVPR3ScorePredictor (which by default
    outputs a single SO(3)-equivariant 3-D vector for the translation score).
    This class outputs the same translation score, but also outputs a
    single SE(3)-invariant 3-D vector for the rotation score.
    """

    def __init__(
        self,
        example_batch: ProteinGraph,
        out_channels: int = 3,
        out_vector_channels: int = 1,
        hidden_scalar_channels: int = 128,
        hidden_vector_channels: int = 64,
        hidden_edge_scalar_channels: int = 64,
        hidden_edge_vector_channels: int = 32,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggr: Union[str, List, Aggregation] = "sum",
        attention_type: Optional[GVPAttentionTypes] = None,
        use_scaling_gate: bool = True,
        share_params: bool = True,
        vector_dim_size: int = 3,
        **kwargs: Any,
    ):
        """
        :param example_batch: Example graph batch object (Data/HeteroData) to infer input feature dimensions from.
        :param out_channels: Number of output scalar features. Defaults to 3 for the single SE(3)-invariant
            rotation score.
        :param out_vector_channels: Number of output vector features. Defaults to 1 for the single 3-D
            SO(3)-equivariant translation score.
        :param hidden_scalar_channels: Dimensionality of the hidden scalar features.
        :param hidden_vector_channels: Dimensionality of the hidden vector features.
        :param hidden_edge_scalar_channels: Dimensionality of the hidden scalar edge features.
        :param hidden_edge_vector_channels: Dimensionality of the hidden vector edge features.
        :param num_layers: Number of message passing layers.
        :param dropout: Dropout rate.
        :param aggr: Aggregation scheme used in message passing.
        :param attention_type: Type of attention layer used in the graph network (GVPN).
        :param use_scaling_gate: Uses two final layers after message passing to output a positive scalar each,
            which are used to scale the output scalar and vector features (separately). This is potentially
            useful to allow the network to learn norm information about its outputs.
        :param share_params: Whether to share parameters across message passing layers.
        :param vector_dim_size: Dimensionality of the vectors used as vector features
            (not the number of vector features).
        :param kwargs: Additional keyword arguments passed to the graph network (GVPN).
        """

        self._out_channels = out_channels
        self._scalar_pred_layer = None
        self._scalar_scaling_gate = None

        super().__init__(
            example_batch,
            out_vector_channels,
            hidden_scalar_channels,
            hidden_vector_channels,
            hidden_edge_scalar_channels,
            hidden_edge_vector_channels,
            num_layers,
            dropout,
            aggr,
            attention_type,
            use_scaling_gate,
            share_params,
            vector_dim_size,
            **kwargs,
        )

    def as_hetero(
        self,
        example_graph: HeteroData,
        aggr: str = "sum",
        **kwargs: Any,
    ):
        """
        Modifies the underlying GVPN network to be
        compatible with a heterogeneous graph, with
        separate message passing parameters for each edge type.
        """

        super().as_hetero(example_graph, aggr, **kwargs)

        self._scalar_pred_layer = to_hetero(
            self._scalar_pred_layer, example_graph.metadata(), aggr=aggr
        )
        if self._use_scaling_gate:
            self._scalar_scaling_gate = to_hetero(
                self._scalar_scaling_gate, example_graph.metadata(), aggr=aggr
            )

    def _create_prediction_layers(self):
        """Generates the prediction layers to be used to generate predictions after message passing."""
        super()._create_prediction_layers()
        # add a final prediction layer for SE(3)-invariant scalar features
        scalar_pred_layer = nn.Sequential(
            nn.Linear(self._hidden_scalar_channels, self._hidden_scalar_channels),
            nn.ReLU(),
            nn.Linear(self._hidden_scalar_channels, self._out_channels),
        )
        self._scalar_pred_layer = scalar_pred_layer

        if self._use_scaling_gate:
            # the features used to calculate the scaling factors are the scalar features
            # concatenated with the norms of the vector features
            num_gate_channels = (
                self._hidden_scalar_channels + self._hidden_vector_channels
            )
            self._scalar_scaling_gate = nn.Sequential(
                nn.Linear(num_gate_channels, num_gate_channels),
                nn.ReLU(),
                nn.Linear(num_gate_channels, 1),
                nn.Softplus(),
            )

    def _predict_scores(
        self,
        scalar_features: Union[torch.Tensor, Dict[str, torch.Tensor]],
        vector_features: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Union[Score, Dict[str, Score]]:
        """
        Gets the rotation and translation scores from the scalar features and vector features
        outputted after message passing.

        :param scalar_features: Scalar features outputted after message passing.
        :param vector_features: Vector features outputted after message passing.
        :returns: 2-tuple of tensors containing the predicted rotation and translation scores.
        """

        rotation_scores = self._scalar_pred_layer(scalar_features)
        translation_scores = self._vector_pred_layer(vector_features)

        if self._use_scaling_gate:
            if self._is_hetero:
                gate_features = {
                    node_type: torch.cat(
                        [
                            scalar_features[node_type],
                            torch.linalg.norm(vector_features[node_type], dim=-1),
                        ],
                        dim=-1,
                    )
                    for node_type in scalar_features
                }
            else:
                gate_features = torch.cat(
                    [
                        scalar_features,
                        torch.linalg.norm(vector_features, dim=-1),
                    ],
                    dim=-1,
                )

            rotation_scale_factors = self._scalar_scaling_gate(gate_features)
            translation_scale_factors = self._vector_scaling_gate(gate_features)

            if self._is_hetero:
                rotation_scores = {
                    node_type: x * rotation_scale_factors[node_type]
                    for node_type, x in rotation_scores.items()
                }
                translation_scores = {
                    node_type: v_x * translation_scale_factors[node_type].unsqueeze(-1)
                    for node_type, v_x in translation_scores.items()
                }
            else:
                rotation_scores *= rotation_scale_factors
                translation_scores *= translation_scale_factors.unsqueeze(-1)

        return rotation_scores, translation_scores
