"""
Contains the datamodule class that loads epitope and CDR structures
for LoopGen diffusion models.
"""

from abc import ABC
from typing import Optional, Sequence, Tuple, Callable, Dict, List, Union, Set

import torch
from e3nn.o3 import rand_matrix
from pytorch_lightning import LightningDataModule
from torch_geometric.data import HeteroData

from .types import CDRFramesBatch

from .utils import (
    pad_cdr_features,
    get_cdr_feature,
    replace_cdr_features,
    replace_epitope_features,
    get_cdr_epitope_subgraphs,
    sinusoidal_encoding,
)
from .types import ProteinGraph, VectorFeatureGraph

from ..data import ReceptorLigandDataset
from ..structure import Structure, LinearStructure, OrientationFrames
from ..graph import VectorFeatureStructureData, VectorFeatureComplexData


def add_time_step_encoding(
    graph: ProteinGraph, time_step_encoding: torch.Tensor
) -> None:
    """
    Adds an input encoding of the time step to the node features `x` in a heterogeneous graph
    for all node types. The argument `time_step` is expected to be a tensor of size 1.
    This modifies the graph in-place.
    """
    cdr_graph, epitope_graph = get_cdr_epitope_subgraphs(graph)
    cdr_node_features = cdr_graph.x
    cdr_node_features_with_time = torch.cat(
        [
            cdr_node_features,
            time_step_encoding.expand(cdr_node_features.shape[0], -1),
        ],
        dim=-1,
    )
    graph = replace_cdr_features(graph, cdr_node_features_with_time)

    if epitope_graph is not None:
        epitope_node_features = epitope_graph.x
        epitope_node_features_with_time = torch.cat(
            [
                epitope_node_features,
                time_step_encoding.expand(epitope_node_features.shape[0], -1),
            ],
            dim=-1,
        )
        graph = replace_epitope_features(graph, epitope_node_features_with_time)

    return graph


def add_cdr_positional_encoding(graph: HeteroData, num_channels: int) -> None:
    """
    Adds a sinusoidal encoding of each CDR's sequence position -
    assumed to be the position of the CDR's features in the node features `x` in a heterogeneous graph.
    This modifies the input graph in-place.
    """
    cdr_features = get_cdr_feature(graph, "x")
    cdr_ptr = get_cdr_feature(graph, "ptr")

    num_per_graph = torch.diff(cdr_ptr)
    num_graphs = len(num_per_graph)
    max_num_per_batch = torch.max(num_per_graph).item()

    all_positions = (
        torch.arange(max_num_per_batch, device=cdr_features.device)
        .unsqueeze(0)
        .expand(num_graphs, -1)
    )

    cdr_positions = all_positions[all_positions < num_per_graph.unsqueeze(-1)]
    cdr_positional_encoding_channels = torch.arange(
        num_channels, device=cdr_features.device
    )
    positional_encodings = sinusoidal_encoding(
        cdr_positions, cdr_positional_encoding_channels
    )

    new_cdr_features = torch.cat([cdr_features, positional_encodings], dim=-1)
    replace_cdr_features(graph, new_cdr_features)


def create_frame_graph(
    cdr_frames: OrientationFrames,
    epitope: Optional[Structure],
    time_step_encoding: torch.Tensor,
    use_cdr_positional_encoding: bool = True,
    num_pos_encoding_channels: int = 5,
    add_pad_cdr_features: bool = False,
    num_pad_cdr_features: int = 0,
    num_pad_cdr_vec_features: int = 0,
    pad_feature_value: float = 0.0,
) -> VectorFeatureGraph:
    """
    Creates a graph representation of an epitope Structure and CDR frames,
    concatenating a sinusoidal time step encoding to the node features
    and optionally a sinusoidal sequence positional encoding of the CDR sequence position.
    """

    if epitope is not None:
        graph = VectorFeatureComplexData.from_structures(epitope, cdr_frames)
    else:
        graph = VectorFeatureStructureData.from_structure(cdr_frames)

    if use_cdr_positional_encoding:
        add_cdr_positional_encoding(graph, num_pos_encoding_channels)

    add_time_step_encoding(graph, time_step_encoding)

    if add_pad_cdr_features is True:
        if num_pad_cdr_features > 0:
            graph = pad_cdr_features(graph, num_pad_cdr_features, pad_feature_value)
        if num_pad_cdr_vec_features > 0:
            graph = pad_cdr_features(
                graph,
                num_pad_cdr_vec_features,
                pad_feature_value,
                pad_dim=-2,
                feature_attr_name="vector_x",
            )

    return graph


def create_coord_graph(
    cdr_frames: OrientationFrames,
    epitope: Optional[Structure],
    time_step_encoding: torch.Tensor,
    use_cdr_positional_encoding: bool = True,
    num_pos_encoding_channels: int = 5,
    add_pad_cdr_features: bool = False,
    num_pad_cdr_features: int = 0,
    num_pad_cdr_vec_features: int = 0,
    pad_feature_value: float = 0.0,
) -> VectorFeatureGraph:
    """
    Creates a graph representation from an epitope Structure and CDR coordinates
    (stored within OrientationFrames), concatenating a sinusoidal time step encoding to the node features
    and optionally a sinusoidal sequence positional encoding of the CDR sequence position.
    Removing frame information involves swapping the CDR vector features for a vector of zeros.
    """

    graph = create_frame_graph(
        cdr_frames,
        epitope,
        time_step_encoding,
        use_cdr_positional_encoding,
        num_pos_encoding_channels,
        add_pad_cdr_features,
        num_pad_cdr_features,
        num_pad_cdr_vec_features,
        pad_feature_value,
    )

    # replace the vector features with a single vector feature of zeros
    if epitope is not None:
        graph["ligand"].vector_x = torch.zeros(
            (cdr_frames.num_residues, 1, 3), device=cdr_frames.translations.device
        )
    else:
        graph.vector_x = torch.zeros(
            (cdr_frames.num_residues, 1, 3), device=cdr_frames.translations.device
        )

    if num_pad_cdr_vec_features > 0:
        graph = pad_cdr_features(
            graph,
            num_pad_cdr_vec_features,
            pad_feature_value,
            pad_dim=-2,
            feature_attr_name="vector_x",
        )

    return graph


class CDRDiffusionDataModule(LightningDataModule, ABC):
    """
    Lightning data module for loading complexes as epitope structures and CDR frames.
    """

    # callable that creates a graph from CDR frames, an epitope Structure, and a time step encoding
    create_graph: Callable

    def __init__(
        self,
        dataset: Optional[ReceptorLigandDataset] = None,
        splits: Optional[Dict[str, Set[str]]] = None,
        fix_cdr_centre: bool = True,
        fixed_cdr_coord: Sequence[float] = (0.0, 0.0, 0.0),
        fix_epitope_centre: bool = False,
        fixed_epitope_coord: Sequence[float] = (0.0, 0.0, 0.0),
        self_conditioning_rate: float = 0.5,
        pad_feature_value: float = 0.0,
        time_step_encoding_channels: int = 5,
        use_cdr_positional_encoding: bool = True,
        positional_encoding_channels: int = 5,
        batch_size: int = 128,
    ):
        """
        :param dataset: Optional ReceptorLigandDataset for loading antigen/CDR Structure objects,
            which yields 3-tuples of the form (name, antigen Structure, cdr Structure)
            when indexed. If this is None, the datamodule throws an error when setup() is called.
        :param splits: Optional dictionary mapping keys ("train", "validation", "test") to sequences of
            antigen/CDR complex names. If this is None, the datamodule throws an error when setup() is called.
        :param fix_cdr_centre: Whether to apply a translation to move every complex so that the CDR
            centre of massed is placed at a fixed coordinate. This must be mutually exclusive with
            fix_epitope_centre.
        :param fixed_cdr_coord: The coordinate at which the CDR centre of mass is fixed to. Only applies
            if fix_cdr_centre is True.
        :param fix_epitope_centre: Whether to apply a translation to move every complex so that the epitope
            centre of mass is placed at a fixed coordinate. This must be mutually exclusive with
            fix_cdr_centre.
        :param fixed_epitope_coord: The coordinate at which the epitope centre of mass is fixed to. Only applies
            if fix_epitope_centre is True.
        :param self_conditioning_rate: The rate at which samples for training "self-conditioning"
            will be used. This is the rate at which the model will make a prediction
            on samples drawn from q(x_{t+1} | x_t) and use that information
            to condition its predictions for q(x_t | x_0). This is only passed as an argument
            here to check whether pad features are needed.
        :param pad_feature_value: The feature value for added pad features (if self-conditioning
            is used).
        :param time_step_encoding_channels: Number of channels to use for the sinusoidal time step
            encoding, which is concatenated onto the node features for both the epitope and CDR.
        :param use_cdr_positional_encoding: Whether to use a sequence positional encoding for CDR
            residues. If True, a sinusoidal positional encoding of each residue's sequence position
            is concatenated to each CDR residue feature. Note this is only used for the CDR residues,
            since it is not guaranteed that the epitope will be a linear sequence.
        :param positional_encoding_channels: Number of channels to use for the sinusoidal positional
            encoding for CDR residues, which is concatenated onto the node features for the CDR only.
        :param batch_size: Number of antigen/CDR complexes per batch.
        :param train_test_val_splits: 3-tuple of floats denoting
            relative proportions of samples allocated
            to the train, test, and validation sets (in that order).
        :param exclude_antigen: Whether to exclude the antigen structure from
            the loaded graphs.
        :param split_by_pdb: Whether to split the dataset by PDB ID,
            so that no structures with the same PDB ID
            are found across train/test/validation sets.
            Useful when multiple CDRs are taken from single
            PDB structures.
        :param cdr_type_dict: Optional dictionary mapping CDR IDs
            (some function of the antigen/CDR complex name)
            to CDR types (an arbitrary label) - only used to
            look-up the sampling weight of each antigen/CDR complex
            if `cdr_type_weights` is provided, specifying the weight of each CDR type.
        :param cdr_type_weights: Optional dictionary mapping CDR types (arbitrary labels)
            to sampling weights to be used by the dataloader.
        :param name_to_id_fn: Function called on each antigen/CDR complex
            name to generate the complex's ID. By default, splits on the
            character '/' and takes the last element.
        :param name_to_pdb_id_fn: Function called on each antigen/CDR complex
            name to obtain the complex's PDB ID. By default, splits on the character
            '/' and takes the second element.
        """

        self._dataset = dataset
        self._splits = splits

        self._train_dataset = None
        self._validation_dataset = None
        self._test_dataset = None

        self._batch_size = batch_size

        if self._dataset is not None:
            self._device = self._dataset.device
        else:
            self._device = torch.device("cpu")

        self._fix_cdr_centre = fix_cdr_centre
        self._fixed_cdr_coord = torch.as_tensor(fixed_cdr_coord, device=self._device)

        self._fix_epitope_centre = fix_epitope_centre
        self._fixed_epitope_coord = torch.as_tensor(
            fixed_epitope_coord, device=self._device
        )

        if self._fix_cdr_centre and self._fix_epitope_centre:
            raise ValueError(
                "Only one of fix_cdr_centre and fix_epitope_centre can be True."
            )

        self._using_self_conditioning = self_conditioning_rate > 0

        self._pad_feature_value = pad_feature_value
        self._time_step_encoding_channels = time_step_encoding_channels
        self._use_cdr_positional_encoding = use_cdr_positional_encoding
        self._positional_encoding_channels = positional_encoding_channels

    def setup(self, stage: str):
        """Performs the train/test/validation split according to the proportions passed to the constructor."""
        if self._dataset is None or self._splits is None:
            raise ValueError(
                "No dataset and/or splits dictionary provided to constructor, setup failed."
            )

        self._train_dataset = self._dataset.subset_by_name(self._splits["train"])
        self._validation_dataset = self._dataset.subset_by_name(
            self._splits["validation"]
        )
        self._test_dataset = self._dataset.subset_by_name(self._splits["test"])

    @property
    def dataset(self) -> ReceptorLigandDataset:
        """The complete underlying dataset."""
        return self._dataset

    @property
    def train_dataset(self) -> Optional[ReceptorLigandDataset]:
        """The training dataset."""
        return self._train_dataset

    @property
    def validation_dataset(self) -> Optional[ReceptorLigandDataset]:
        """The validation dataset."""
        return self._validation_dataset

    @property
    def test_dataset(self) -> Optional[ReceptorLigandDataset]:
        """The test dataset."""
        return self._test_dataset

    def generate_example(self) -> ProteinGraph:
        """
        Generates an example graph from random data.
        """
        num_res_per_batch = 5
        num_residues = self._batch_size * num_res_per_batch

        cdr_rotations = rand_matrix(num_residues, device=self._device)
        cdr_translations = torch.randn((num_residues, 3), device=self._device)
        cdr_batch = torch.arange(
            self._batch_size, device=self._device
        ).repeat_interleave(num_res_per_batch)
        cdr_frames = OrientationFrames(cdr_rotations, cdr_translations, batch=cdr_batch)

        dummy_time_step_encoding = torch.zeros(
            self._time_step_encoding_channels, device=cdr_frames.translations.device
        )

        epitope_N = torch.randn((num_residues, 3), device=self._device)
        epitope_CA = torch.randn((num_residues, 3), device=self._device)
        epitope_C = torch.randn((num_residues, 3), device=self._device)
        epitope_CB = torch.randn((num_residues, 3), device=self._device)
        epitope_sequence = torch.randint(20, size=(num_residues,), device=self._device)
        epitope_batch = torch.arange(
            self._batch_size, device=self._device
        ).repeat_interleave(num_res_per_batch)
        epitope_structure = Structure(
            epitope_N,
            epitope_CA,
            epitope_C,
            epitope_CB,
            epitope_sequence,
            batch=epitope_batch,
        )

        example_graph = self.create_graph(
            cdr_frames,
            epitope_structure,
            time_step_encoding=dummy_time_step_encoding,
            use_cdr_positional_encoding=self._use_cdr_positional_encoding,
            num_pos_encoding_channels=self._positional_encoding_channels,
            add_pad_cdr_features=self.add_pad_cdr_features,
            num_pad_cdr_features=self.num_pad_cdr_features,
            num_pad_cdr_vec_features=self.num_pad_cdr_vec_features,
            pad_feature_value=self._pad_feature_value,
        )

        return example_graph

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """The train dataloader using the train dataset."""
        return torch.utils.data.DataLoader(
            self._train_dataset,
            collate_fn=self.collate,
            batch_size=self._batch_size,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        The validation dataloader using the validation dataset. This is
        typically used for hyperparameter searches or early stopping.
        """
        return torch.utils.data.DataLoader(
            self._validation_dataset,
            collate_fn=self.collate,
            batch_size=self._batch_size,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        The test dataloader using the test dataset. This is typically used
        for final model evaluation.
        """

        return torch.utils.data.DataLoader(
            self._test_dataset,
            collate_fn=self.collate,
            batch_size=self._batch_size,
        )

    def fix_complex_by_epitope(
        self,
        epitope: Structure,
        cdr: Union[OrientationFrames, LinearStructure],
    ) -> Tuple[Structure, Union[OrientationFrames, LinearStructure]]:
        """
        Translates the coordinates of the CDR and epitope structures so the epitope
        structure is centered at the coordinate specified by `fixed_epitope_coord`. Returns two new objects
        (one epitope Structure and one CDR LinearStructure or OrientationFrames) with translated coordinates.
        """
        centered_epitope, epitope_centre_of_mass = epitope.center()
        fixed_epitope = centered_epitope.translate(self._fixed_epitope_coord)

        if not cdr.has_batch:
            cdr_batch = torch.zeros(
                len(cdr), device=epitope_centre_of_mass.device, dtype=torch.int64
            )
        else:
            cdr_batch = cdr.batch

        translations = epitope_centre_of_mass - self._fixed_epitope_coord
        fixed_cdr = cdr.translate(-translations[cdr_batch])

        return fixed_epitope, fixed_cdr

    def fix_complex_by_cdr(
        self,
        epitope: Structure,
        cdr: OrientationFrames,
    ) -> Tuple[Structure, Union[OrientationFrames, LinearStructure]]:
        """
        Translates the coordinates of the CDR and epitope structures so the CDR
        structure is centered at the coordinate specified by `fixed_cdr_coord`. Returns two new objects
        (one epitope Structure and one CDR OrientationFrames) with translated coordinates.
        """
        fixed_cdr, cdr_centroids = cdr.center()

        translations = cdr_centroids - self._fixed_cdr_coord
        fixed_epitope = epitope.translate(-translations[epitope.batch])

        return fixed_epitope, fixed_cdr

    def collate(
        self, structures: List[Tuple[str, Structure, LinearStructure]]
    ) -> CDRFramesBatch:
        """
        Combines a list of tuples of the form (names, antigen Structure, cdr Structure) into a
        graph, with the CDR represented only as a set of orientation frames. Then samples a time
        step and uses the forward process to noise the CDR orientation frames and gets the
        rotation/translation scores for the new samples under the noising distribution.

        Returns a tuple consisting of:
            1. Tuple of CDR names, one for each in the batch
            2. Batch of epitope structures (if `exclude_antigen` is False, otherwise None)
            3. Batch of CDR structures
        """
        names, epitopes, cdrs = map(tuple, zip(*structures))

        epitope = Structure.combine(epitopes)
        cdr_frames = OrientationFrames.combine([cdr.orientation_frames for cdr in cdrs])

        if self._fix_cdr_centre:
            epitope, cdr_frames = self.fix_complex_by_cdr(epitope, cdr_frames)
        elif self._fix_epitope_centre:
            epitope, cdr_frames = self.fix_complex_by_epitope(epitope, cdr_frames)

        return names, epitope, cdr_frames

    @property
    def add_pad_cdr_features(self):
        """Whether to add pad features (default False)."""
        return False

    @property
    def num_pad_cdr_features(self) -> int:
        """Number of pad CDR features (default 0)."""
        return 0

    @property
    def num_pad_cdr_vec_features(self) -> int:
        """Number of pad CDR vector features (default 0)."""
        return 0


class CDRCoordinateDataModule(CDRDiffusionDataModule):
    """
    A datamodule that loads epitope/CDR structures for coordinate diffusion.
    """

    create_graph = staticmethod(create_coord_graph)


class CDRFrameDataModule(CDRDiffusionDataModule):
    """
    A datamodule that loads epitope/CDR structures for frame diffusion.
    """

    create_graph = staticmethod(create_frame_graph)

    @property
    def add_pad_cdr_features(self) -> bool:
        """Add pad CDR features for frame diffusion if using self conditions."""
        return self._using_self_conditioning

    @property
    def num_pad_cdr_features(self) -> int:
        """Number of pad CDR features for frame diffusion."""
        return 3 if self._using_self_conditioning else 0

    @property
    def num_pad_cdr_vec_features(self) -> int:
        """Number of pad CDR vector features for frame diffusion."""
        return 1 if self._using_self_conditioning else 0
