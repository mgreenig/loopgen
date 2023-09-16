"""
Some useful utility functions, many of which are related
to modifying or interacting with CDR graphs.
"""

from typing import Optional, Tuple, Any, Union

from functools import partial

import torch
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean

from .types import ProteinGraph
from ..utils import node_type_subgraph
from ..structure import Structure, OrientationFrames
from ..graph import StructureData


def center_complex_by_cdr(
    epitope: Structure,
    cdr_frames: OrientationFrames,
) -> Tuple[Structure, OrientationFrames]:
    """
    Centers the whole complex so the CDR's center of mass is at the origin,
    returning the translated epitope and CDR as a 2-tuple.
    """
    if not cdr_frames.has_batch:
        cdr_batch = torch.zeros(
            len(cdr_frames), device=cdr_frames.translations.device, dtype=torch.int64
        )
        cdr_ptr = torch.as_tensor(
            [0, len(cdr_frames)],
            device=cdr_frames.translations.device,
            dtype=torch.int64,
        )
    else:
        cdr_batch = cdr_frames.batch
        cdr_ptr = cdr_frames.ptr

    if not epitope.has_batch:
        epitope_batch = torch.zeros(
            len(epitope), device=epitope.CA_coords.device, dtype=torch.int64
        )
    else:
        epitope_batch = epitope.batch

    cdr_center_of_mass = scatter_mean(cdr_frames.translations, cdr_batch, dim=0)
    centered_cdr_coords = cdr_frames.translations - cdr_center_of_mass[cdr_batch]

    centered_cdr_frames = OrientationFrames(
        cdr_frames.rotations,
        centered_cdr_coords,
        batch=cdr_batch,
        ptr=cdr_ptr,
    )
    centered_epitope = epitope.translate(-cdr_center_of_mass[epitope_batch])

    return centered_epitope, centered_cdr_frames


def get_cdr_epitope_subgraphs(
    graph: ProteinGraph,
) -> Tuple[StructureData, Optional[StructureData]]:
    """
    Gets the CDR and epitope subgraphs of a StructureData or ComplexData object.
    If the graph is heterogeneous, gets the node type subgraphs
    for "ligand" (epitope) and "receptor" (CDR) and converts
    each into a homogeneous form. If the graph is homogeneous, it is
    checked for a `node_type` attribute, and if it is present,
    it is assumed that epitope nodes are
    `node_type == 0` and CDR nodes are `node_type == 1`.
    If `node_type` is not present, assumes the graph is just
    a PeptideGraph of an unbound CDR (epitope is None).
    """
    if isinstance(graph, HeteroData):
        cdr = graph.node_type_subgraph(["ligand"]).to_homogeneous()
        epitope = graph.node_type_subgraph(["receptor"]).to_homogeneous()
    elif hasattr(graph, "node_type"):
        cdr = node_type_subgraph(graph, node_type=1)
        epitope = node_type_subgraph(graph, node_type=0)
    else:
        cdr = graph
        epitope = None

    return cdr, epitope


def _get_feature(
    graph: ProteinGraph, feature_attr_name: str, key: str, node_type: int
) -> Any:
    """
    Gets the feature tensor for a specified node-level attribute
    in a StructureData/ComplexData graph object.

    If the underlying object is a ComplexData object, looks up the attribute under `key`.
    If the underlying object is a StructureData object with a node_type attribute, returns the
    feature tensor for nodes with `node_type == node_type`. If the underlying
    object is a StructureData object with no attribute node_type, returns
    the whole feature tensor.
    """

    if isinstance(graph, HeteroData):
        cdr_features = getattr(graph[key], feature_attr_name)
    elif hasattr(graph, "node_type"):
        cdr_mask = graph.node_type == node_type
        cdr_features = getattr(graph, feature_attr_name)[cdr_mask]
    else:
        cdr_features = getattr(graph, feature_attr_name)

    return cdr_features


get_node_feature_docstring = """
    Gets the {} feature tensor for a specified node-level attribute
    in a StructureData/ComplexData graph object.
    
    :param graph: The graph object to get the features from.
    :param feature_attr_name: The name of the attribute storing the feature tensor.
"""

get_cdr_feature = partial(_get_feature, key="ligand", node_type=1)
get_cdr_feature.__doc__ = get_node_feature_docstring.format("CDR")
get_epitope_feature = partial(_get_feature, key="receptor", node_type=0)
get_epitope_feature.__doc__ = get_node_feature_docstring.format("epitope")


def _replace_features(
    graph: ProteinGraph,
    replacement_features: Any,
    key: str,
    node_type: int,
    inplace: bool = True,
    feature_attr_name: str = "x",
) -> ProteinGraph:
    """
    Replaces features for residues in a StructureData/ComplexData graph object.

    If the underlying object is a ComplexData object, looks up the feature attribute under `key`.
    If the underlying object is a StructureData object with a node_type attribute, returns the
    feature tensor for nodes with `node_type == node_type`. If the underlying
    object is a StructureData object with no attribute node_type, returns
    the whole feature tensor.
    """
    if not inplace:
        graph = graph.clone()

    if isinstance(graph, HeteroData):
        setattr(graph[key], feature_attr_name, replacement_features)
    elif hasattr(graph, "node_type"):
        cdr_mask = graph.node_type == node_type
        features = getattr(graph, feature_attr_name)
        features[cdr_mask] = replacement_features
    else:
        setattr(graph, feature_attr_name, replacement_features)

    return graph


replace_features_docstring = """
    Replaces features for {} residues in a StructureData/ComplexData graph object.

    :param graph: The graph object to modify.
    :param replacement_features: The replacement features.
    :param inplace: Whether to modify the graph in-place or return a copy. (default: True)
    :param feature_attr_name: The name of the attribute to modify. (default: "x")
"""


replace_cdr_features = partial(_replace_features, key="ligand", node_type=1)
replace_cdr_features.__doc__ = replace_features_docstring.format("CDR")
replace_epitope_features = partial(_replace_features, key="receptor", node_type=0)
replace_epitope_features.__doc__ = replace_features_docstring.format("epitope")


def _pad_features(
    graph: ProteinGraph,
    num_pads: int,
    pad_value: float,
    key: str,
    inplace: bool = True,
    pad_dim: int = -1,
    feature_attr_name: str = "x",
) -> ProteinGraph:
    """
    Pads node features in a graph by adding `num_pads` new padded features,
    each with a value of `pad_value`. Used in recycling to add initial recurrent
    features to the input graph (i.e. where the model's own predictions are provided
    as features in subsequent iterations).

    If the underlying object is a ComplexData object, pads the feature attribute under `key`.
    If the underlying object is a StructureData object, pads the whole underlying
    feature tensor.
    """

    if not inplace:
        graph = graph.clone()

    graph_is_hetero = isinstance(graph, HeteroData)

    if graph_is_hetero:
        features_to_pad = getattr(graph[key], feature_attr_name)
    else:
        features_to_pad = getattr(graph, feature_attr_name)

    if pad_dim < 0:
        num_dims_before_pad = -1 - pad_dim
    else:
        num_dims_before_pad = len(features_to_pad.shape) - pad_dim - 1

    pad = ((0, 0) * num_dims_before_pad) + (0, num_pads)

    padded_features = torch.nn.functional.pad(features_to_pad, pad=pad, value=pad_value)

    if graph_is_hetero:
        setattr(graph[key], feature_attr_name, padded_features)
    else:
        setattr(graph, feature_attr_name, padded_features)

    return graph


pad_features_docstring = """
    Pads {} features in a graph by adding `num_pads` new padded features,
    each with a value of `pad_value`. 
    
    :param graph: A ProteinGraph object.
    :param num_pads: The number of new padded features to add.
    :param pad_value: The value to use for the new padded features.
    :param inplace: Whether to modify the graph in place. (default: True)
    :param pad_dim: The dimension along which to pad the features. (default: -1)
    :param feature_attr_name: The name of the feature attribute to pad. (default: "x")
"""

pad_cdr_features = partial(_pad_features, key="ligand")
pad_cdr_features.__doc__ = pad_features_docstring.format("CDR")
pad_epitope_features = partial(_pad_features, key="receptor")
pad_epitope_features.__doc__ = pad_features_docstring.format("epitope")


def _update_structure(
    graph: ProteinGraph,
    new_structure: Union[OrientationFrames, Structure],
    key: str,
    inplace: bool = True,
) -> ProteinGraph:
    """
    Updates the underlying structure in a graph object.

    If the underlying object is a ComplexData object, pads the Structure under `key`.
    If the underlying object is a StructureData object, updates the whole underlying
    Structure.
    """

    if not inplace:
        graph = graph.clone()

    not_hetero = not isinstance(graph, HeteroData)
    no_node_type = not hasattr(graph, "node_type")

    if not_hetero:
        if no_node_type:
            graph.update_structure(new_structure)
            return graph
        graph.to_heterogeneous()

    graph.update_structure(new_structure, key="ligand")

    if not_hetero:
        graph.to_homogeneous()

    return graph


update_structure_docstring = """
    Updates the underlying {} structure in a graph object.
    
    :param graph: A ProteinGraph object.
    :param new_structure: The new structure to use.
    :param inplace: Whether to modify the graph in place. (default: True)
"""

update_cdr_structure = partial(_update_structure, key="ligand")
update_cdr_structure.__doc__ = update_structure_docstring.format("CDR")
update_epitope_structure = partial(update_cdr_structure, key="receptor")
update_epitope_structure.__doc__ = update_structure_docstring.format("epitope")


def sinusoidal_encoding(value: torch.Tensor, channels: torch.Tensor, base: int = 100):
    """
    Sinusoidal encoding of a value for some number of channels, as in the original
    Transformer (https://arxiv.org/abs/1706.03762).

    :param value: Rank 1 tensor of containing N values to be encoded.
    :param channels: Channels over which the encoding is generated. This should be
        a rank 1 tensor of M positive integers, increasing by 1.
    :param base: Value used to scale the sin/cos function outputs in the final encoding.
        The larger this number is, the less variability there will be between encodings for different values.
        The original Transformer paper used 10000 here but we prefer a smaller number to introduce
        more variation between encodings for shorter sequences.
    :returns: Tensor of shape (N, M) containing the M-dimensional encoding for
        each of the N input values.
    """
    encoding = torch.where(
        channels % 2 == 0,
        torch.sin(
            value.unsqueeze(-1) / (base ** (2 * channels / channels[-1])).unsqueeze(0)
        ),
        torch.cos(
            value.unsqueeze(-1) / (base ** (2 * channels / channels[-1])).unsqueeze(0)
        ),
    )
    return encoding
