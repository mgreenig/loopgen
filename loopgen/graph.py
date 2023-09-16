"""
Module containing code for graph representations of proteins.
"""

from __future__ import annotations
from typing import (
    Sequence,
    Tuple,
    Optional,
    Union,
    Callable,
    List,
    Literal,
    Any,
    Dict,
    Type,
)

from collections import defaultdict

from abc import ABC, abstractmethod
from enum import Enum
from itertools import product

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.hetero_data import to_homogeneous_edge_index
from torch_geometric.typing import PairTensor
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import knn, radius

from e3nn.o3 import matrix_to_quaternion

from .structure import Structure, OrientationFrames, AminoAcid3
from .utils import standardise_batch_and_ptr

# the number of amino acids (used for the one hot encoder)
NUM_AMINO_ACIDS: int = len(AminoAcid3.__members__)


class EdgeIndexCalculator(ABC):
    """
    A class that calculates edge indices when called,
    taking a single Data/HeteroData object as input.
    """

    def __init__(
        self,
        self_loops: bool = False,
        flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
        **kwargs: Any,
    ):
        self.self_loops = self_loops
        self.flow = flow

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(self_loops={self.self_loops}, flow={self.flow})"
        )

    def __call__(
        self, data: Union[Data, HeteroData]
    ) -> Union[torch.Tensor, Dict[Tuple[str, str], torch.Tensor]]:
        """Gets the edge indices returning them as one or more tensors of shape (2, num_edges)."""
        if isinstance(data, HeteroData):
            return self._get_edge_indices_heterogeneous(data)
        else:
            return self._get_edge_indices_homogeneous(data)

    def _expand_attr_for_heterogeneous(
        self, attr_name: str, num_edge_types: int
    ) -> None:
        """
        Converts a non-sequence-valued attribute into a sequence
        of identical values, in order to match the number of edges in a heterogeneous graph.
        Throws an error if the attribute is already a sequence, but has the wrong length for
        the specified number of edge types. This modifies the underlying attribute.
        """

        value = getattr(self, attr_name)

        # if the attribute is not a sequence, this will throw a type error due to len() call
        try:
            if len(value) != num_edge_types:
                raise ValueError(
                    f"Expected {num_edge_types} values for {attr_name}, but constructor received {len(value)}."
                )
        # set the instance attribute permanently to save time on the next iteration
        except TypeError:
            setattr(self, attr_name, [value] * num_edge_types)

    def _contract_attr_for_homogeneous(self, attr_name: str) -> None:
        """
        Converts a sequence-valued attribute into a
        single value so it can be used with a homogeneous graph.
        Throws an error if the attribute is already a sequence with multiple
        distinct values. This modifies the underlying attribute.
        """
        value = getattr(self, attr_name)
        if isinstance(value, Sequence):
            # if all values are the same, set the instance attribute permanently
            if len(set(value)) == 1:
                setattr(self, attr_name, value[0])
            else:
                raise TypeError(
                    f"Expected a single value for {attr_name}, but constructor received "
                    f"multiple distinct values {value}."
                )

    @abstractmethod
    def _get_edge_indices_heterogeneous(
        self, data: HeteroData
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """
        Defines how edge indices should be calculated for a heterogeneous graph, returning
        a dict mapping pairs of (source, target) node types to edge indices. Note
        that the ordering of the pair of node types does NOT depend on flow - it
        is always (source, target).
        """
        pass

    @abstractmethod
    def _get_edge_indices_homogeneous(self, data: Data) -> torch.Tensor:
        """Defines how edge indices should be calculated for a homogeneous graph."""
        pass


class KNNEdgeIndexCalculator(EdgeIndexCalculator):
    """
    Gets edge indices based off K-nearest neighbors. Assumes
    node coordinates are stored under the `pos` attribute of the Data/HeteroData objects.
    """

    def __init__(
        self,
        k: Union[int, Sequence[int]] = 6,
        self_loops: bool = False,
        flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
    ):
        super().__init__(self_loops, flow)

        if isinstance(k, int):
            k_is_invalid = k <= 0
        elif isinstance(k, Sequence):
            k_is_invalid = any([k_i <= 0 for k_i in k])
        else:
            k_is_invalid = True

        if k_is_invalid:
            raise ValueError(
                f"Expected k to be an integer or sequence of integers all greater than 0, but received {k}."
            )

        self.k = k

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k}, self_loops={self.self_loops}, flow={self.flow})"

    def _get_edge_indices_heterogeneous(
        self, data: HeteroData
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """Calculates K-nearest neighbors edge indices for a HeteroData object."""
        node_types, _ = data.metadata()
        num_edge_types = len(node_types) ** 2

        self._expand_attr_for_heterogeneous("k", num_edge_types)

        all_edge_indices = {}
        for i, (source_node_type, target_node_type) in enumerate(
            product(node_types, node_types)
        ):
            self_loops_to_be_removed = (
                not self.self_loops and source_node_type == target_node_type
            )

            k = self.k[i]
            if self_loops_to_be_removed:
                k += 1

            source_node_storage = data[source_node_type]
            target_node_storage = data[target_node_type]

            edge_indices = knn(
                source_node_storage.pos,
                target_node_storage.pos,
                k=k,
                batch_x=source_node_storage.get("batch"),
                batch_y=target_node_storage.get("batch"),
            )

            edge_type = (source_node_type, target_node_type)

            if self.flow == "source_to_target":
                edge_indices = edge_indices[[1, 0]]

            if self_loops_to_be_removed:
                edge_indices, _ = remove_self_loops(edge_indices)

            all_edge_indices[edge_type] = edge_indices

        return all_edge_indices

    def _get_edge_indices_homogeneous(self, data: Data) -> torch.Tensor:
        """Calculates K-nearest neighbors edge indices for a Data object."""

        self._contract_attr_for_homogeneous("k")

        edge_indices = knn(
            data.pos,
            data.pos,
            self.k if not self.self_loops else self.k + 1,
            data.batch,
            data.batch,
        )

        if self.flow == "source_to_target":
            edge_indices = edge_indices[[1, 0]]

        if not self.self_loops:
            edge_indices, _ = remove_self_loops(edge_indices)

        return edge_indices


class RadiusEdgeIndexCalculator(EdgeIndexCalculator):
    """
    Calculates edge indices for all nodes within a certain radius of each other node. Assumes
    node coordinates are stored under the `pos` attribute of the Data/HeteroData objects.
    """

    def __init__(
        self,
        distance: Union[float, Sequence[float]] = 15.0,
        max_num_neighbors: Union[int, Sequence[int]] = 32,
        self_loops: bool = False,
        flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
    ):
        super().__init__(self_loops, flow)
        self.distance = distance
        self.max_num_neighbors = max_num_neighbors

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(distance={self.distance}, max_num_neighbors={self.max_num_neighbors}, "
            f" self_loops={self.self_loops}, flow={self.flow})"
        )

    def _get_edge_indices_heterogeneous(
        self, data: HeteroData
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """Calculates radius-based edge indices for a HeteroData object."""
        node_types, _ = data.metadata()
        num_edge_types = len(node_types) ** 2

        self._expand_attr_for_heterogeneous("radius", num_edge_types)
        self._expand_attr_for_heterogeneous("max_num_neighbors", num_edge_types)

        all_edge_indices = {}
        for i, (source_node_type, target_node_type) in enumerate(
            product(node_types, node_types)
        ):
            source_node_storage = data[source_node_type]
            target_node_storage = data[target_node_type]

            edge_indices = radius(
                source_node_storage.pos,
                target_node_storage.pos,
                self.distance[i],
                source_node_storage.batch,
                target_node_storage.batch,
                self.max_num_neighbors[i],
            )

            edge_type = (source_node_type, target_node_type)

            if self.flow == "source_to_target":
                edge_indices = edge_indices[[1, 0]]

            if not self.self_loops and source_node_type == target_node_type:
                edge_indices, _ = remove_self_loops(edge_indices)

            all_edge_indices[edge_type] = edge_indices

        return all_edge_indices

    def _get_edge_indices_homogeneous(self, data: Data) -> torch.Tensor:
        """Calculates radius-based edge indices for a Data object."""

        self._contract_attr_for_homogeneous("radius")
        self._contract_attr_for_homogeneous("max_num_neighbors")

        edge_indices = radius(
            data.pos,
            data.pos,
            self.distance,
            data.batch,
            data.batch,
            self.max_num_neighbors,
        )

        if not self.self_loops:
            edge_indices, _ = remove_self_loops(edge_indices)

        return edge_indices


class FullyConnectedEdgeIndexCalculator(EdgeIndexCalculator):
    """Calculates edge indices for all nodes with all other nodes in the graph."""

    def _get_edge_indices_heterogeneous(
        self, data: HeteroData
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """Calculates fully connected edge indices for a HeteroData object."""
        node_types, _ = data.metadata()

        all_edge_indices = {}
        for i, (source_node_type, target_node_type) in enumerate(
            product(node_types, node_types)
        ):
            source_node_storage = data[source_node_type]
            target_node_storage = data[target_node_type]

            if source_node_storage.batch is None:
                source_batch = torch.zeros(
                    source_node_storage.num_nodes, dtype=torch.long
                )
            else:
                source_batch = source_node_storage.batch

            if target_node_storage.batch is None:
                target_batch = torch.zeros(
                    target_node_storage.num_nodes, dtype=torch.long
                )
            else:
                target_batch = target_node_storage.batch

            edge_indices = torch.nonzero(
                torch.eq(
                    source_batch.unsqueeze(-1),
                    target_batch.unsqueeze(0),
                ),
            ).transpose(-2, -1)

            edge_type = (source_node_type, target_node_type)

            if self.flow == "target_to_source":
                edge_indices = edge_indices[[1, 0]]

            if not self.self_loops and source_node_type == target_node_type:
                edge_indices, _ = remove_self_loops(edge_indices)

            all_edge_indices[edge_type] = edge_indices

        return all_edge_indices

    def _get_edge_indices_homogeneous(self, data: Data) -> torch.Tensor:
        """Calculates fully connected edge indices for a Data object."""

        if data.batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long)
        else:
            batch = data.batch

        edge_indices = torch.nonzero(
            torch.eq(
                batch.unsqueeze(-1),
                batch.unsqueeze(0),
            ),
        ).transpose(-2, -1)

        if self.flow == "target_to_source":
            edge_indices = edge_indices[[1, 0]]

        if not self.self_loops:
            edge_indices, _ = remove_self_loops(edge_indices)

        return edge_indices


# Methods for calculating edge indices:
# - knn: k-nearest neighbors
# - radius: all neighbors within a radius
# - fc: fully connected
class EdgeIndexMethods(Enum):
    """Enum for the parameters of different edge index methods."""

    knn = KNNEdgeIndexCalculator
    radius = RadiusEdgeIndexCalculator
    fc = FullyConnectedEdgeIndexCalculator


def get_edge_index_calculator(method: str, **kwargs: Any) -> EdgeIndexCalculator:
    """Returns an EdgeIndexCalculator for the given method, with the given kwargs passed to the constructor."""
    if method in EdgeIndexMethods.__members__:
        return EdgeIndexMethods[method].value(**kwargs)
    else:
        raise ValueError(
            f"Unknown edge index calculation method: {method}. "
            f"Should be one of {tuple(EdgeIndexMethods.__members__.keys())}."
        )


def get_backbone_angle_features(
    N_CA_bond_vectors: torch.Tensor, CA_C_bond_vectors: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the angle between N-CA and CA-C bond vectors stored in two tensors of shape (..., 3).
    """

    return torch.acos(
        torch.sum(
            torch.nn.functional.normalize(N_CA_bond_vectors, dim=-1)
            * torch.nn.functional.normalize(CA_C_bond_vectors, dim=-1),
            dim=-1,
            keepdim=True,
        )
    )


def se3_inv_node_features(structure: Structure) -> torch.Tensor:
    """
    Returns SE(3)-invariant node features for a protein structure, stored in a rank 2 tensor
    of shape (N, 22) where N is the number of nodes, with 22 features calculated per node:
        1. One-hot encoding of amino acid type (20 features)
        2. Beta carbon orientation, expressed in local frame (1 feature)
        3. Backbone angle (1 feature)
    """
    sequence_features = torch.nn.functional.one_hot(
        structure.sequence, num_classes=NUM_AMINO_ACIDS
    ).to(torch.float32)

    CB_CA_vectors = structure.CB_coords - structure.CA_coords
    N_CA_vectors = structure.N_coords - structure.CA_coords
    C_CA_vectors = structure.C_coords - structure.CA_coords

    CB_orientation_features = torch.nn.functional.normalize(CB_CA_vectors, dim=-1)

    # make orientation features SE(3)-invariant by rotating
    CB_orientation_features = torch.matmul(
        CB_orientation_features.unsqueeze(-2),
        structure.orientation_frames.rotations,
    ).squeeze(-2)

    backbone_angle_features = get_backbone_angle_features(N_CA_vectors, C_CA_vectors)

    features = torch.cat(
        [
            sequence_features,
            CB_orientation_features,
            backbone_angle_features,
        ],
        dim=-1,
    )

    return features


def so3_equiv_node_features(structure: Structure) -> PairTensor:
    """
    Returns SE(3)-invariant scalar features and SO(3)-equivariant vector features for a protein structure.
    The scalar features are a one-hot encoding of the amino acid sequence of shape (N, 20)
    and the vector features are stored in a rank 3 tensor of shape (N, 3, 3), with 3 unit-length
    3-D vector features calculated for each of the N residues:
        1. Beta carbon orientation
        2. N-CA bond vector
        3. C-CA bond vector
    """
    scalar_features = torch.nn.functional.one_hot(
        structure.sequence, num_classes=NUM_AMINO_ACIDS
    ).to(torch.float32)

    CB_CA_vectors = structure.CB_coords - structure.CA_coords
    N_CA_vectors = structure.N_coords - structure.CA_coords
    C_CA_vectors = structure.C_coords - structure.CA_coords

    CB_CA_orientations = torch.nn.functional.normalize(CB_CA_vectors, dim=-1)
    N_CA_orientations = torch.nn.functional.normalize(N_CA_vectors, dim=-1)
    C_CA_orientations = torch.nn.functional.normalize(C_CA_vectors, dim=-1)

    vector_features = torch.stack(
        [CB_CA_orientations, N_CA_orientations, C_CA_orientations], dim=-2
    )

    return scalar_features, vector_features


def rbf(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    Gaussian radial basis function, parameterised by `centers`.

    :param values: Set of distances used as input to the RBF.
    :param centers: Centers (means) of the radial basis functions.
    :return: Tensor of RBF values.
    """

    return torch.exp(-((values - centers) ** 2))


def to_distance_features(vectors: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor of norm features for and input set of vectors of shape (..., N, 3).
    These features are obtained by passing the norm of each vector through `self.rbf`

    :param vectors: Input set of vectors for which norm features will be calculated.
    :param centers: Centers (means) of the radial basis functions.
    :return: Tensor of length features, with last dimension size equal to the number of RBF bandwidths.
    """

    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    length_features = rbf(lengths, centers)

    return length_features


def so3_equiv_edge_features(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    rbf_centers: torch.Tensor,
) -> PairTensor:
    """
    Gets a 2-tuple of edge features from a tensor of positions and edge indices,
    the first element of which are the SE(3)-invariant distance features and the second element of
    which are the SO(3)-equivariant orientation features.
    """

    pos_differences = source_pos - target_pos

    distance_features = to_distance_features(pos_differences, rbf_centers)
    orientation_features = torch.nn.functional.normalize(
        pos_differences, dim=-1
    ).unsqueeze(-2)

    return distance_features, orientation_features


def se3_inv_edge_features(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    rbf_centers: torch.Tensor,
    source_orientation_matrices: torch.Tensor,
    target_orientation_matrices: torch.Tensor,
):
    """
    Gets a rank 2 tensor of SE(3)-invariant edge features from tensors of source/target positions
    and orientations and radial basis function centers, consisting of:
        1. Distance features (radial basis functions of the distance between source and target positions)
        2. Position difference features (SE(3)-invariant representation of the difference
            between the source and target positions)
        3. Orientation difference features (SE(3)-invariant representation of the difference
            between the source and target orientations)
    """

    distance_features, orientation_features = so3_equiv_edge_features(
        source_pos, target_pos, rbf_centers
    )
    se3_inv_pos_diff_features = torch.matmul(
        orientation_features, target_orientation_matrices
    ).squeeze(-2)
    se3_inv_orientation_diff_features = matrix_to_quaternion(
        torch.matmul(
            target_orientation_matrices.transpose(-2, -1), source_orientation_matrices
        )
    )

    edge_features = torch.cat(
        [
            distance_features,
            se3_inv_pos_diff_features,
            se3_inv_orientation_diff_features,
        ],
        dim=-1,
    )

    return edge_features


# A conformer is defined as either a Structure (which has atomic coordinates and a sequence)
# or an OrientationFrames object (which consists of a set of rotations and translations)
Conformer = Union[Structure, OrientationFrames]


def _conformer_fn_selector(
    conformer: Conformer,
    structure_fn: Callable[[Structure], Any],
    orientation_frame_fn: Callable[[OrientationFrames], Any],
) -> Any:
    """
    For an input object of type Structure or OrientationFrames,
    checks the object type and applies the `structure_fn` if the object
    is a Structure object and the `orientation_frame_fn` if the object is an
    OrientationFrames object.
    """
    if isinstance(conformer, Structure):
        output = structure_fn(conformer)
    elif isinstance(conformer, OrientationFrames):
        output = orientation_frame_fn(conformer)
    else:
        raise TypeError(
            f"Input object should be type Structure or OrientationFrames, got: {type(conformer)}"
        )

    return output


def structure_scalar_features(structure: Structure) -> Dict[str, Any]:
    """Returns a dictionary of node-level attributes for a structure."""
    node_features = se3_inv_node_features(structure)

    node_attrs = dict(
        x=node_features,
        sequence=structure.sequence,
        orientations=structure.orientation_frames.rotations,
        pos=structure.orientation_frames.translations,
        batch=structure.batch,
        ptr=structure.ptr,
    )

    return node_attrs


def structure_vector_features(structure: Structure) -> Dict[str, Any]:
    """Node attributes for a structure."""
    scalar_features, vector_features = so3_equiv_node_features(structure)

    node_attrs = dict(
        x=scalar_features,
        vector_x=vector_features,
        sequence=structure.sequence,
        orientations=structure.orientation_frames.rotations,
        pos=structure.orientation_frames.translations,
        batch=structure.batch,
        ptr=structure.ptr,
    )

    return node_attrs


def orientation_frame_vector_features(frames: OrientationFrames) -> Dict[str, Any]:
    """
    Node attributes for orientation frames, which are:
        1. A single scalar feature of 1 for each node (since there is no sequence information)
        2. The first two column vectors for each orientation matrix, defining the orientation of the residue.
        Since the third column vector is a deterministic function of these two (cross product), there is
        no need to include it.
    """
    scalar_features = torch.ones(
        (frames.num_residues, 1), device=frames.rotations.device
    )
    vector_features = frames.rotations[..., :2].transpose(-2, -1)

    node_attrs = dict(
        x=scalar_features,
        vector_x=vector_features,
        orientations=frames.rotations,
        pos=frames.translations,
        batch=frames.batch,
        ptr=frames.ptr,
    )
    return node_attrs


class StructureData(Data, ABC):

    """
    A graph representation of a protein structure. Inherits from the `torch_geometric.data.Data` class.
    """

    complex_data_class: Type[ComplexData]

    def __init__(
        self,
        rbf_centers: torch.Tensor = torch.arange(2, 22, 2, dtype=torch.float32),
        edge_method: str = "knn",
        edge_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if edge_kwargs is None:
            edge_kwargs = {}

        edge_index_fn = get_edge_index_calculator(edge_method, **edge_kwargs)

        super().__init__(
            rbf_centers=rbf_centers,
            edge_index_fn=edge_index_fn,
            _edge_method=edge_method,
            _edge_kwargs=edge_kwargs,
            _fully_connected=edge_method == "fc",
            **kwargs,
        )

        if not hasattr(self, "batch"):
            self.batch = None
        if not hasattr(self, "ptr"):
            self.ptr = None

    @classmethod
    def from_structure(
        cls,
        structure: Conformer,
        rbf_centers: torch.Tensor = torch.arange(2, 22, 2, dtype=torch.float32),
        edge_method: str = "knn",
        edge_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        :param structure: Structure object containing the protein structure.
        :param rbf_centers: Centers of the radial basis functions used to compute distance-related features.
        :param edge_method: Method used to calculate edges in the graph.
            Should be one of ("knn", "radius", "fc").
        :param edge_kwargs: Dictionary of key word arguments to be passed to the EdgeIndexCalculator.
        :param kwargs: Additional key word arguments that will be passed to the Data constructor.
        """

        device = _conformer_fn_selector(
            structure,
            lambda x: x.CA_coords.device,
            lambda x: x.translations.device,
        )

        graph = cls(
            rbf_centers=rbf_centers.to(device),
            edge_method=edge_method,
            edge_kwargs=edge_kwargs,
            **kwargs,
        )

        attrs = graph.get_attrs(structure)

        for attr, value in attrs.items():
            setattr(graph, attr, value)

        return graph

    @abstractmethod
    def _get_node_attrs(self, structure: Structure) -> Dict[str, Any]:
        """Determines how to get node-level attributes from a Structure object."""
        pass

    @abstractmethod
    def _get_edge_attrs(
        self,
        edge_index: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Determines how to get edge-level attributes for a specified set of edge indices.
        """
        pass

    def get_attrs(self, structure: Structure) -> Dict[str, Any]:
        """
        Returns a dictionary of node-level attributes for a given structure, which are `x`,
        `sequence`, `orientations`, `pos`, `batch`, and `ptr`.
        """
        node_attrs = self._get_node_attrs(structure)

        self.pos = node_attrs["pos"]
        self.orientations = node_attrs["orientations"]
        self.batch = node_attrs["batch"]

        edge_index = self.edge_index_fn(self)

        edge_attrs = self._get_edge_attrs(edge_index)

        attrs = {**node_attrs, **edge_attrs}

        return attrs

    def update_structure(self, new_structure: Structure):
        """
        Updates the graph with a new underlying structure. Node and edge features are recomputed.
        """
        new_attrs = self.get_attrs(new_structure)
        for attr, value in new_attrs.items():
            setattr(self, attr, value)

    def to_heterogeneous(
        self,
        node_type: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
        node_type_names: Sequence[str] = ("receptor", "ligand"),
        edge_type_names: Sequence[Tuple[str, str, str]] = (
            ("receptor", "to", "receptor"),
            ("receptor", "to", "ligand"),
            ("ligand", "to", "receptor"),
            ("ligand", "to", "ligand"),
        ),
    ):
        """
        Converts a StructureData object to a ComplexData object.

        Most of this function is pasted directly from `torch_geometric`
        (https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/data.html#Data.to_heterogeneous),
        however we modify it so that the output is a `ComplexData` object rather than a `HeteroData` object. Also,
        the `node_type_names` and `edge_type_names` here default to the node and edge types from a ComplexData object.
        """
        if node_type is None:
            node_type = self._store.get("node_type", None)
        if node_type is None:
            node_type = torch.zeros(self.num_nodes, dtype=torch.long)
            node_type_names = ["0"]

        if edge_type is None:
            edge_type = self._store.get("edge_type", None)
        if edge_type is None:
            edge_type = torch.zeros(self.num_edges, dtype=torch.long)
            edge_type_names = ["0"]

        # We iterate over node types to find the local node indices belonging
        # to each node type. Furthermore, we create a global `index_map` vector
        # that maps global node indices to local ones in the final
        # heterogeneous graph:
        node_ids, index_map = {}, torch.empty_like(node_type)
        for i, key in enumerate(node_type_names):
            node_ids[i] = (node_type == i).nonzero(as_tuple=False).view(-1)
            index_map[node_ids[i]] = torch.arange(
                len(node_ids[i]), device=index_map.device
            )

        # We iterate over edge types to find the local edge indices:
        edge_ids = {}
        for i, _ in enumerate(edge_type_names):
            edge_ids[i] = (edge_type == i).nonzero(as_tuple=False).view(-1)

        data = self.complex_data_class(
            rbf_centers=self.rbf_centers,
            edge_method=self._edge_method,
            edge_kwargs=self._edge_kwargs,
        )

        for i, key in enumerate(node_type_names):
            for attr, value in self.items():
                if attr in {"node_type", "edge_type", "ptr"}:
                    continue
                elif isinstance(value, torch.Tensor) and self.is_node_attr(attr):
                    cat_dim = self.__cat_dim__(attr, value)
                    data[key][attr] = value.index_select(cat_dim, node_ids[i])

            if len(data[key]) == 0:
                data[key].num_nodes = node_ids[i].size(0)

        for i, key in enumerate(edge_type_names):
            src, _, dst = key
            for attr, value in self.items():
                if attr in {"node_type", "edge_type", "ptr"}:
                    continue
                elif attr == "edge_index":
                    edge_index = value[:, edge_ids[i]]
                    edge_index[0] = index_map[edge_index[0]]
                    edge_index[1] = index_map[edge_index[1]]
                    data[key].edge_index = edge_index
                elif isinstance(value, torch.Tensor) and self.is_edge_attr(attr):
                    cat_dim = self.__cat_dim__(attr, value)
                    data[key][attr] = value.index_select(cat_dim, edge_ids[i])

        # Add global attributes.
        exclude_keys = set(data.keys) | {
            "node_type",
            "edge_type",
            "edge_index",
            "num_nodes",
            "ptr",
        }
        for attr, value in self.items():
            if attr in exclude_keys:
                continue
            data[attr] = value

        # add ptr variable to each node type
        for node_type in data.node_types:
            if hasattr(data[node_type], "batch"):
                _, data[node_type]["ptr"] = standardise_batch_and_ptr(
                    data[node_type]["batch"], None
                )

        return data


class ScalarFeatureStructureData(StructureData):

    """
    A StructureData class that contains SE(3)-invariant scalar features for each node and edge in the graph.
    """

    # this will take the value ScalarFeatureComplexData -
    # the value is added below after the class is defined
    complex_data_class: Type[ComplexData]

    def _get_node_attrs(self, structure: Structure) -> Dict[str, Any]:
        return structure_scalar_features(structure)

    def _get_edge_attrs(
        self,
        edge_index: torch.Tensor,
    ):
        source_index, target_index = edge_index

        edge_features = se3_inv_edge_features(
            self.pos[source_index],
            self.pos[target_index],
            self.rbf_centers,
            self.orientations[source_index],
            self.orientations[target_index],
        )

        edge_attrs = dict(edge_attr=edge_features, edge_index=edge_index)

        return edge_attrs


class VectorFeatureStructureData(StructureData):

    """
    A StructureData class that contains SE(3)-invariant scalar features
    `x` and SO(3)-equivariant vector features `vector_x` for each node
    and edge in the graph. Unlike ScalarFeatureStructureData, this class
    can be used to describe a set of orientation frames as well as a
    complete structure.
    """

    # this will take the value VectorFeatureComplexData -
    # the value is added below after the class is defined
    complex_data_class: Type[ComplexData]

    def _get_node_attrs(self, structure: Conformer) -> Dict[str, Any]:
        """
        Gets the node attributes for an input conformer, depending on whether
        it is a complete structure or just a set of orientation frames.
        """
        node_attrs = _conformer_fn_selector(
            structure,
            structure_vector_features,
            orientation_frame_vector_features,
        )

        return node_attrs

    def _get_edge_attrs(self, edge_index: torch.Tensor) -> Dict[str, Any]:
        """
        Gets edge-level attributes for a specified set of edge indices,
        including `edge_index`, `edge_attr` and `vector_edge_attr`.
        """

        source_index, target_index = edge_index

        scalar_features, vector_features = so3_equiv_edge_features(
            self.pos[source_index], self.pos[target_index], self.rbf_centers
        )

        edge_attrs = dict(
            edge_attr=scalar_features,
            vector_edge_attr=vector_features,
            edge_index=edge_index,
        )

        return edge_attrs


class ComplexData(HeteroData, ABC):

    """
    A heterogeneous graph representation of the structure of a protein complex, with separate structures
    for the receptor and ligand. The receptor/ligand distinction is arbitrary - it is only used
    to label the node types in the graph. This class inherits from `torch_geometric.data.HeteroData`.
    """

    structure_data_class: Type[StructureData]

    def __init__(
        self,
        rbf_centers: torch.Tensor = torch.arange(2, 22, 2, dtype=torch.float32),
        edge_method: str = "knn",
        edge_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if edge_kwargs is None:
            edge_kwargs = {}

        edge_index_fn = get_edge_index_calculator(edge_method, **edge_kwargs)

        super().__init__(
            rbf_centers=rbf_centers,
            edge_index_fn=edge_index_fn,
            _edge_method=edge_method,
            _fully_connected=edge_method == "fc",
            **kwargs,
        )

        # important to set this attribute after the constructor
        # so it does not think it is a node type (since it is a dict)
        self._edge_kwargs = edge_kwargs

        for node_type in self.node_types:
            if not hasattr(self[node_type], "batch"):
                self[node_type].batch = None
            if not hasattr(self, "ptr"):
                self[node_type].ptr = None

    @classmethod
    def from_structures(
        cls,
        receptor_structure: Conformer,
        ligand_structure: Conformer,
        rbf_centers: torch.Tensor = torch.arange(2, 22, 2, dtype=torch.float32),
        edge_method: str = "knn",
        edge_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ComplexData:
        """
        :param receptor_structure: Structure of the receptor protein.
        :param ligand_structure: Structure of the ligand protein.
        :param rbf_centers: Centers of the radial basis functions used to compute distance-related features.
        :param edge_method: Method for calculating edge indices -
            either "knn", "radius", or "fc" (fully connected).
        :param edge_kwargs: Dictionary of key word arguments to be passed to the EdgeIndexCalculator.
        :param kwargs: Additional keyword arguments to be passed to the `torch_geometric.data.HeteroData` constructor.
        """

        device = _conformer_fn_selector(
            receptor_structure,
            lambda x: x.CA_coords.device,
            lambda x: x.translations.device,
        )

        graph = cls(
            rbf_centers=rbf_centers.to(device),
            edge_method=edge_method,
            edge_kwargs=edge_kwargs,
            **kwargs,
        )

        attrs = graph.get_attrs(receptor_structure, ligand_structure)

        for key, attr_dict in attrs.items():
            for attr, value in attr_dict.items():
                setattr(graph[key], attr, value)

        return graph

    @abstractmethod
    def _get_node_attrs(self, structure: Structure) -> Dict[str, Any]:
        """Determines how to get node-level attributes from a Structure object."""
        pass

    @abstractmethod
    def _get_edge_attrs(
        self,
        edge_type: Tuple[str, str],
        edge_index: torch.Tensor,
        pos_storage: HeteroData,
    ) -> Dict[str, Any]:
        """
        Determines how to get edge-level attributes for a particular edge type with specified edge index
        from a HeteroData `pos_storage` object containing:
            - The position of each node `pos`
            - The orientation of each node `orientations`
            - An optional batch index tensor `batch`
        """
        pass

    def get_attrs(
        self, receptor_structure: Structure, ligand_structure: Structure
    ) -> Dict[Union[str, Tuple[str, str]], Dict[str, Any]]:
        """
        Returns a dictionary of dictionaries, storing the graph attributes under keys for each
        node or edge type. Node type keys are simple strings, while edge type keys are 2-tuples
        of strings, representing the source and target node types, respectively.
        """

        receptor_node_attrs = self._get_node_attrs(receptor_structure)
        ligand_node_attrs = self._get_node_attrs(ligand_structure)

        node_attrs = dict(receptor=receptor_node_attrs, ligand=ligand_node_attrs)

        for node_type in node_attrs:
            self[node_type].batch = node_attrs[node_type]["batch"]
            self[node_type].pos = node_attrs[node_type]["pos"]
            self[node_type].orientations = node_attrs[node_type]["orientations"]

        edge_index_dict = self.edge_index_fn(self)

        edge_attrs = {}

        # add the edge indices/features to the graph
        for (source, target), edge_index in edge_index_dict.items():
            if edge_index.shape[-1] > 0:
                edge_attrs[(source, target)] = self._get_edge_attrs(
                    (source, target), edge_index, self
                )

        attrs = {**node_attrs, **edge_attrs}

        return attrs

    def update_structure(self, new_structure: Conformer, key: str):
        """
        Updates the graph with a new underlying structure under the specified key (node type).
        Node and edge features are recomputed if needed.
        """
        if key not in self.node_types:
            raise KeyError(
                f"Tried to add new structure under key {key}, but this is not a node type."
            )

        node_attrs = self._get_node_attrs(new_structure)

        for attr, value in node_attrs.items():
            setattr(self[key], attr, value)

        self[key].pos = _conformer_fn_selector(
            new_structure, lambda x: x.CA_coords, lambda x: x.translations
        )
        self[key].batch = new_structure.batch
        self[key].orientations = _conformer_fn_selector(
            new_structure,
            lambda x: x.orientation_frames.rotations,
            lambda x: x.rotations,
        )

        if self._fully_connected:
            edge_index = {
                edge_type: self[edge_type].edge_index for edge_type in self.edge_types
            }
        else:
            edge_index = self.edge_index_fn(self)

        # only recalculate edge features involving the new structure
        new_edge_attrs = {}
        for source, _, target in self.edge_types:
            if source == key or target == key:
                new_edge_attrs[(source, target)] = self._get_edge_attrs(
                    (source, target), edge_index[(source, target)], self
                )

        for edge_type, attr_dict in new_edge_attrs.items():
            for attr, value in attr_dict.items():
                setattr(self[edge_type], attr, value)

    def to_homogeneous(
        self,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        add_node_type: bool = True,
        add_edge_type: bool = True,
    ) -> StructureData:
        """
        Converts the heterogeneous ComplexData object to a homogeneous StructureData object with
        node and edge features concatenated into a single tensor.
        """
        edge_index, node_slices, edge_slices = to_homogeneous_edge_index(self)
        device = edge_index.device if edge_index is not None else None

        homogeneous_attrs = defaultdict(list)

        for node_type in self.node_types:
            for attr, value in self[node_type].items():
                if node_attrs is None or attr in node_attrs:
                    homogeneous_attrs[attr].append(value)

        for edge_type in self.edge_types:
            for attr, value in self[edge_type].items():
                if attr == "edge_index":
                    continue
                if edge_attrs is None or attr in edge_attrs:
                    homogeneous_attrs[attr].append(value)

        for attr, values in homogeneous_attrs.items():
            homogeneous_attrs[attr] = torch.cat(values, dim=0)

        homogeneous_attrs["edge_index"] = edge_index
        homogeneous_attrs["rbf_centers"] = self.rbf_centers
        homogeneous_attrs["edge_method"] = self._edge_method
        homogeneous_attrs["edge_kwargs"] = self._edge_kwargs

        if add_node_type:
            sizes = [offset[1] - offset[0] for offset in node_slices.values()]
            sizes = torch.as_tensor(sizes, dtype=torch.long, device=device)
            node_type = torch.arange(len(sizes), device=device)
            homogeneous_attrs["node_type"] = node_type.repeat_interleave(sizes)

        if add_edge_type and edge_index is not None:
            sizes = [offset[1] - offset[0] for offset in edge_slices.values()]
            sizes = torch.as_tensor(sizes, dtype=torch.long, device=device)
            edge_type = torch.arange(len(sizes), device=device)
            homogeneous_attrs["edge_type"] = edge_type.repeat_interleave(sizes)

        return self.structure_data_class(**homogeneous_attrs)


class ScalarFeatureComplexData(ComplexData):

    """
    A heterogeneous graph representation of the structure of a protein complex using SE(3)-invariant
    scalar-valued node and edge features.
    """

    structure_data_class = ScalarFeatureStructureData

    def _get_node_attrs(self, structure: Structure) -> Dict[str, Any]:
        """Returns a dictionary of node-level attributes for a structure."""
        return structure_scalar_features(structure)

    def _get_edge_attrs(
        self,
        edge_type: Tuple[str, str],
        edge_index: torch.Tensor,
        pos_storage: HeteroData,
    ) -> Dict[str, Any]:
        """
        Returns a dictionary of edge-level attributes for the graph under each edge type.
        Takes a position storage HeteroData object as input, with the `pos`, `orientations`,
        and `batch` attribute stored under each node type.
        """
        edge_attrs = {}

        source, target = edge_type
        edge_attrs["edge_index"] = edge_index
        source_index, target_index = edge_index
        edge_attr = se3_inv_edge_features(
            pos_storage[source].pos[source_index],
            pos_storage[target].pos[target_index],
            self.rbf_centers,
            pos_storage[source].orientations[source_index],
            pos_storage[target].orientations[target_index],
        )
        edge_attrs["edge_attr"] = edge_attr

        return edge_attrs


class VectorFeatureComplexData(ComplexData):

    """
    A heterogeneous graph representation of the structure of a protein complex with separated
    SE(3)-invariant scalar features `x` and SO(3)-equivariant vector features `vector_x`.
    Unlike the `ScalarFeatureComplexData` class, this class can be used to describe a complex
    where the receptor or ligand (or both) are OrientationFrames rather than complete structures.
    """

    structure_data_class = VectorFeatureStructureData

    def _get_node_attrs(self, structure: Conformer) -> Dict[str, Any]:
        """
        Gets the node attributes for an input object, depending on whether
        it is a complete structure or just a set of orientation frames.
        """
        node_attrs = _conformer_fn_selector(
            structure,
            structure_vector_features,
            orientation_frame_vector_features,
        )

        return node_attrs

    def _get_edge_attrs(
        self,
        edge_type: Tuple[str, str],
        edge_index: torch.Tensor,
        pos_storage: HeteroData,
    ) -> Dict[str, Any]:
        """
        Returns a dictionary of edge-level attributes for the graph under each edge type.
        Takes a position storage HeteroData object as input, with the `pos`, `orientations`,
        and `batch` attribute stored under each node type.
        """
        edge_attrs = {}

        source, target = edge_type
        edge_attrs["edge_index"] = edge_index
        source_index, target_index = edge_index
        scalar_edge_attr, vector_edge_attr = so3_equiv_edge_features(
            pos_storage[source].pos[source_index],
            pos_storage[target].pos[target_index],
            self.rbf_centers,
        )
        edge_attrs["edge_attr"] = scalar_edge_attr
        edge_attrs["vector_edge_attr"] = vector_edge_attr

        return edge_attrs


# add ComplexData class variables to the structure data classes -
# must be done after the ComplexData classes are defined
ScalarFeatureStructureData.complex_data_class = ScalarFeatureComplexData
VectorFeatureStructureData.complex_data_class = VectorFeatureComplexData
