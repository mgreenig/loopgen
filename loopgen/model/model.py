"""
Contains CDRBackboneDiffusionModel classes, which defines the behaviour for a
diffusion model that generates CDR backbone conformations. This class is stores a score
prediction network, as well as the diffusion reverse process modules that use predicted
scores to update the CDR backbone via Langevin dynamics. 
"""

from typing import Type, Union, List, Tuple, Optional, Sequence

from random import random

import torch
from torch import nn
from torch_geometric.typing import PairTensor
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.seed import isolate_rng
from scipy.spatial.transform import Rotation

from .datamodule import (
    CDRDiffusionDataModule,
    CDRCoordinateDataModule,
    CDRFrameDataModule,
)
from .network import GVPSE3ScorePredictor, GVPR3ScorePredictor
from .types import CDRFramesBatch, ForwardProcessOutput, Score

from .types import ProteinGraph, VectorFeatureGraph
from .utils import (
    get_cdr_feature,
    replace_cdr_features,
    sinusoidal_encoding,
)

from ..distributions import IGSO3Distribution, Gaussian3DDistribution
from ..nn.diffusion import (
    BetaScheduleTypes,
    BetaScheduleSelector,
    IGSO3ForwardProcess,
    Gaussian3DForwardProcess,
    SO3ReverseProcess,
    R3ReverseProcess,
)
from ..structure import Structure, OrientationFrames


def update_time_step_encoding(
    features: torch.Tensor,
    time_step_encoding: torch.Tensor,
) -> torch.Tensor:
    """
    Updates the time step encoding in a feature tensor, assuming the time step
    encoding was added to the end of the last dimension of the tensor.
    """
    encoding_dim = time_step_encoding.shape[-1]
    features[..., -encoding_dim:] = time_step_encoding

    return features


class CDRCoordinateDiffusionModel(LightningModule):
    """
    Diffusion model that generates CDR structures by diffusion
    over CDR alpha carbon coordinates.
    """

    network_class: Type[nn.Module] = GVPR3ScorePredictor
    datamodule_class: Type[CDRDiffusionDataModule] = CDRCoordinateDataModule

    def __init__(
        self,
        network: nn.Module,
        learning_rate: float = 1e-4,  # only needed if training
        self_conditioning_rate: float = 0.5,
        translation_beta_schedule: BetaScheduleTypes = "linear",
        num_time_steps: int = 100,
        min_trans_beta: float = 1e-4,
        max_trans_beta: float = 20,
        weight_loss_by_norm: bool = True,
        pad_feature_value: float = 0.0,
        time_step_encoding_channels: int = 5,
        use_cdr_positional_encoding: bool = True,
        positional_encoding_channels: int = 5,
        test_results_filepath: str = "test_results.csv",  # for storing prediction results on the test set
    ):
        """
        :param network: Network used to predict the rotation and translation scores.
        :param learning_rate: Optimizer learning rate.
        :param self_conditioning_rate: The rate at which samples for training "self-conditioning"
            will be used. This is the proportion of training steps in which the model will make a prediction
            on samples drawn from q(x_{t+1} | x_t) and use that information
            to condition its predictions for q(x_t | x_0). Implemented the same way as RFDiffusion.
        :param translation_beta_schedule: Type of variance schedule used for translations.
        :param num_time_steps: Number of diffusion time steps.
        :param min_trans_beta: Minimum variance of the translation diffusion process.
        :param max_trans_beta: Maximum variance of the translation diffusion process.
        :param weight_loss_by_norm: Whether to weight the loss by the norm of the predicted
            rotation and translation scores.
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
        :param test_results_filepath: Filepath under which test results will be saved.
        """
        super().__init__()

        self.save_hyperparameters(ignore=["network"])

        self._network = network
        self._learning_rate = learning_rate

        # for storing outputs for individual batches in training/testing/validation
        self._train_outputs = []
        self._test_outputs = []
        self._val_outputs = []

        self._loss_fn = nn.MSELoss(reduction="none")
        self._num_time_steps = num_time_steps
        self._time_steps = torch.arange(
            num_time_steps, dtype=torch.long, device=self._device
        )

        get_trans_schedule = BetaScheduleSelector(
            num_time_steps, min_trans_beta, max_trans_beta
        )
        self.register_buffer(
            "_trans_beta_schedule", get_trans_schedule(translation_beta_schedule)
        )
        self._num_time_steps = num_time_steps

        # forward process
        self._translation_forward_process = Gaussian3DForwardProcess(
            self._trans_beta_schedule,
        )

        self._self_conditioning_rate = self_conditioning_rate
        self._using_self_conditioning = self._self_conditioning_rate > 0
        self._add_pad_cdr_features = False
        self._pad_feature_value = pad_feature_value

        self._translation_stationary_dist = Gaussian3DDistribution(
            mean=torch.zeros(3, device=self._device)
        )

        self.register_buffer(
            "_time_step_encoding_channels",
            torch.arange(1, time_step_encoding_channels + 1),
        )
        # precompute time step encodings
        self.register_buffer(
            "_time_step_encodings",
            sinusoidal_encoding(
                torch.arange(self._num_time_steps + 1),
                self._time_step_encoding_channels,
            ),
        )

        self._use_cdr_positional_encoding = bool(use_cdr_positional_encoding)
        self._positional_encoding_channels = positional_encoding_channels

        self._translation_reverse_process = R3ReverseProcess(self._trans_beta_schedule)

        if self._num_time_steps <= 0:
            raise ValueError(
                f"num_time_steps must be a positive integer, got {num_time_steps}."
            )

        self._weight_loss_by_norm = weight_loss_by_norm

        self.register_buffer("_rev_time_steps", torch.arange(num_time_steps - 1, 0, -1))
        self._test_results_filepath = test_results_filepath

        self._num_pad_cdr_features = 0
        self._num_pad_cdr_vec_features = 0

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """By default, model use the AdamW optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self._learning_rate)

    @property
    def network(self) -> nn.Module:
        """Property for accessing the underlying network."""
        return self._network

    def create_graph(
        self,
        cdr_frames: OrientationFrames,
        epitope: Structure,
        time_step_encoding: torch.Tensor,
    ) -> VectorFeatureGraph:
        """
        Creates a VectorFeatureComplexGraph from the given CDR frames and epitope structure and
        sets the CDR vector features to a single vector of 0s, so as to remove any conformational
        information outside of the CDR CA coordinates.
        """
        graph = self.datamodule_class.create_graph(
            cdr_frames,
            epitope,
            time_step_encoding,
            use_cdr_positional_encoding=self._use_cdr_positional_encoding,
            num_pos_encoding_channels=self._positional_encoding_channels,
            add_pad_cdr_features=self._add_pad_cdr_features,
            num_pad_cdr_features=self._num_pad_cdr_features,
            num_pad_cdr_vec_features=self._num_pad_cdr_vec_features,
            pad_feature_value=self._pad_feature_value,
        )

        return graph

    def sample_time_step(self) -> torch.Tensor:
        """
        Uniformly samples an integer time step. Currently this samples a single time step
        for the whole batch, to save on compute time when sampling from noise distributions
        (since samples need only to be drawn from one distribution).
        """
        return self._time_steps[
            torch.randint(self._num_time_steps, (1,), device=self._time_steps.device)
        ]

    def sample_forward(
        self,
        cdr_frames: OrientationFrames,
        epitope: Optional[Structure],
        time_step: torch.Tensor,
    ) -> ForwardProcessOutput:
        """
        For an input epitope/CDR structure and time step, returns a graph with noised CDR frames
        sampled from q(x_t | x_0). If self-conditioning, also returns a graph with
        noised CDR frames sampled from q(x_{t+1} | x_t).

        Although the rotations are passed forward (to be compatible with larger library), they
        are ignored in the create_graph() function, so this model is in fact ablating the rotational information.
        See create_coord_graph() in models/abdiff/datamodule for details.
        """
        noised_coords, score = self._translation_forward_process.sample(
            cdr_frames.translations, time_step.item()
        )

        noised_cdr_frames = OrientationFrames(
            cdr_frames.rotations,
            noised_coords,
            batch=cdr_frames.batch,
            ptr=cdr_frames.ptr,
        )

        noised_graph = self.create_graph(
            noised_cdr_frames,
            epitope,
            time_step_encoding=self._time_step_encodings[time_step],
        )

        # stochastically determine whether to return an additional sample for self-conditioning
        training_self_conditioning = random() < self._self_conditioning_rate
        if training_self_conditioning:
            # if the time step is the last one, sample from the stationary distribution
            if time_step.item() == self._num_time_steps - 1:
                self_cond_coords = self._translation_stationary_dist.sample(
                    len(noised_cdr_frames.translations), device=self._device
                )
            else:
                # samples for self-conditioning are conditioned on the noised CDR
                self_cond_coords, _ = self._translation_forward_process.sample(
                    noised_cdr_frames.translations,
                    (time_step + 1).item(),
                    from_x0=False,
                )

            self_cond_cdr_frames = OrientationFrames(
                cdr_frames.rotations,
                self_cond_coords,
                batch=cdr_frames.batch,
                ptr=cdr_frames.ptr,
            )
            self_cond_graph = self.create_graph(
                self_cond_cdr_frames,
                epitope,
                time_step_encoding=self._time_step_encodings[time_step],
            )
        else:
            self_cond_graph = None

        return score, noised_graph, self_cond_graph

    def forward(self, graph: ProteinGraph, time_step: torch.Tensor) -> Score:
        """
        Returns an SO(3)-equivariant translation score tensor of shape (N, 3).
        """
        pred_trans_scores = self._network(graph)

        if self._network.is_hetero:
            cdr_trans_scores = pred_trans_scores["ligand"]
        elif hasattr(graph, "node_type"):
            cdr_mask = graph.node_type == 1
            cdr_trans_scores = pred_trans_scores[cdr_mask]
        else:
            cdr_trans_scores = pred_trans_scores

        cdr_trans_scores = cdr_trans_scores[..., 0, :]

        return cdr_trans_scores

    @staticmethod
    def self_condition_translations(
        graph: ProteinGraph,
        pred_trans_scores: torch.Tensor,
    ) -> ProteinGraph:
        """
        Adds self-conditioning information - i.e predicted scores obtained by
        running a forward pass - to `graph`, returning the modified graph.
        This modifies `graph` inplace.
        """

        # add predicted translation scores to features
        self_cond_vector_x = get_cdr_feature(graph, "vector_x").clone()
        self_cond_vector_x[..., -1:, :] = pred_trans_scores.unsqueeze(-2)

        graph = replace_cdr_features(
            graph, self_cond_vector_x, feature_attr_name="vector_x"
        )

        return graph

    def self_condition(self, graph: ProteinGraph, scores: torch.Tensor):
        """
        Adds self-conditioning information to the graph by adding translation scores
        to the node features.
        """
        graph = self.self_condition_translations(graph, scores)
        return graph

    def calculate_loss(
        self, batch: CDRFramesBatch, time_step: torch.Tensor
    ) -> torch.Tensor:
        """
        Runs a forward pass and calculates MSE loss for rotational and translational scores.
        The returned loss tensors contain the MSE loss for each individual score, stored
        within a rank 1 tensor.
        """
        names, epitope, cdr = batch
        (
            trans_scores,
            graph,
            self_cond_graph,
        ) = self.sample_forward(cdr, epitope, time_step)

        if self_cond_graph is not None:
            pred_trans_scores = self(self_cond_graph, time_step + 1)
            graph = self.self_condition_translations(graph, pred_trans_scores)

        # predict scores and get losses
        pred_trans_scores = self(graph, time_step)
        trans_score_errors = self._loss_fn(pred_trans_scores, trans_scores)
        trans_score_mse = torch.mean(trans_score_errors, dim=-1)

        return trans_score_mse

    def training_step(self, batch: CDRFramesBatch) -> torch.Tensor:
        """
        Predicts the translation score for each residue
        in the batch and computes the loss between the predicted and
        ground truth scores.
        """
        time_step = self.sample_time_step()

        trans_score_mse = self.calculate_loss(batch, time_step)

        mean_trans_loss = torch.mean(trans_score_mse)

        self.log("translation_train_loss", mean_trans_loss.item())
        self.log("time_step", time_step.item())

        return mean_trans_loss

    def validation_step(self, batch: CDRFramesBatch, batch_idx: int) -> None:
        """
        Calculates rotation and translation loss on the validation set for every time step,
        storing them within a list of validation outputs.
        """

        for time_step in self._time_steps:
            self._val_outputs.append(self.calculate_loss(batch, time_step))

    def test_step(self, batch: CDRFramesBatch, batch_idx: int) -> None:
        """
        Calculates rotation and translation loss on the test set for every time step,
        storing them within a protected list of test outputs.
        """
        for time_step in self._time_steps:
            self._test_outputs.append(self.calculate_loss(batch, time_step))

    def on_validation_epoch_end(self) -> None:
        """
        Calculates the mean MSE on the whole validation set.
        """
        trans_score_mse = torch.cat(self._val_outputs)

        mean_trans_loss = torch.mean(trans_score_mse)

        self.log("validation_loss", mean_trans_loss.item())

        self._val_outputs.clear()

    def on_test_epoch_end(self) -> None:
        """
        Calculates the mean rotational and translational losses
        on the whole test set.
        """
        trans_score_mse = torch.cat(self._test_outputs)

        mean_trans_loss = torch.mean(trans_score_mse)

        self.log("test_loss", mean_trans_loss.item())

        self._test_outputs.clear()

    def _update_cdr(
        self,
        scores: torch.Tensor,
        current_cdr: OrientationFrames,
        time_step: torch.Tensor,
        noise_scale: float,
        self_cond: bool,
    ) -> OrientationFrames:
        """
        Updates the CDR backbone conformation by performing a reverse diffusion step,
        returning the updated CDR OrientationFrames object.
        """
        int_time_step = time_step.item()

        cdr_positions = current_cdr.translations

        new_cdr_positions = self._translation_reverse_process(
            cdr_positions,
            scores,
            int_time_step,
        )

        updated_cdr = OrientationFrames(
            current_cdr.rotations,
            new_cdr_positions,
            batch=current_cdr.batch,
            ptr=current_cdr.ptr,
        )

        return updated_cdr

    def sample_stationary(
        self,
        size: int,
        batch: Optional[torch.Tensor] = None,
    ) -> OrientationFrames:
        """
        Draw an OrientationFrames object from the stationary distribution.
        The rotations are initialised with the identity, and the translations
        are sampled from a standard normal.

        :param size: Number of frames to sample.
        :param batch: Optional batch assignment tensor of shape (size,) for the sampled frames,
            assigning each frame to a batch element. This can be used to generate a batch of
            frames from the stationary distribution.
        :returns: OrientationFrames object.
        """

        random_translations = self._translation_stationary_dist.sample(
            size=(size,), device=self._device
        )

        # Rotations are constant
        cdr_rotations = (
            torch.eye(3, device=random_translations.device)
            .unsqueeze(0)
            .expand(size, -1, -1)
        )
        random_cdr_frames = OrientationFrames(
            cdr_rotations,
            random_translations,
            batch=batch,
        )

        return random_cdr_frames

    def generate(
        self,
        batch: CDRFramesBatch,
        noise_scale: float = 0.2,
        self_cond: bool = True,
        return_intermediates: bool = False,
        n: int = 1,
    ) -> Union[
        Tuple[Structure, List[OrientationFrames]],
        Tuple[Structure, List[List[OrientationFrames]]],
    ]:
        """
        Initialises the CDR with random translations and
        performs the reverse diffusion process over CDR coordinates
        to generate a CDR backbone conformation.

        :param batch: 3-tuple containing the names of the CDRs in the batch,
            the epitope structure, and the CDR orientation frames.
        :param noise_scale: Noise scale for the reverse diffusion process.
        :param self_cond: Whether to use self-conditioning information.
        :param return_intermediates: Whether to return a list of intermediate CDRs.
        :param n: Number of CDRs to generate for each epitope.
        :returns: 2-tuple containing the epitope Structure and the n generated CDR OrientationFrames
            objects. If return_intermediates is True, returns a list of 2-tuples containing the
            epitope Structure and the generated CDR OrientationFrames objects at each time step.
        """

        old_noise_scale = self._translation_reverse_process.noise_scale
        self._translation_reverse_process.noise_scale = noise_scale

        names, epitope, cdr = batch
        epitope_repeated = epitope.repeat(n)
        cdr_repeated = cdr.repeat(n)

        cdr_frames = self.sample_stationary(len(cdr_repeated), batch=cdr_repeated.batch)

        time_step = self._time_steps[-1] + 1

        graph = self.create_graph(
            cdr_frames,
            epitope_repeated,
            time_step_encoding=self._time_step_encodings[time_step],
        )

        all_states = []

        for time_step in self._rev_time_steps:
            scores = self(graph, time_step)
            cdr_frames = self._update_cdr(
                scores, cdr_frames, time_step, noise_scale, self_cond
            )

            cdr_frames = cdr_frames.center(return_centre=False)

            # make a new graph
            graph = self.create_graph(
                cdr_frames,
                epitope_repeated,
                time_step_encoding=self._time_step_encodings[time_step],
            )

            # add self-conditioning information only if passed to generate
            if self_cond:
                graph = self.self_condition(graph, scores)

            all_states.append(cdr_frames.split())

        self._translation_reverse_process.noise_scale = old_noise_scale

        # assemble the generated CDRs into individual OrientationFrames objects
        if cdr.has_batch:
            # if batched, created a batched OrientationFrames object for each round of generation performed
            n_frames = len(cdr.ptr) - 1
            all_generated_cdr_states = []
            for state in all_states:
                generated_cdrs = []
                for i in range(0, n_frames * n, n_frames):
                    start, end = i, i + n_frames
                    cdr_gen_frames = state[start:end]
                    gen_frames_batch = OrientationFrames.combine(cdr_gen_frames)
                    generated_cdrs.append(gen_frames_batch)
                all_generated_cdr_states.append(generated_cdrs)
        else:
            all_generated_cdr_states = all_states

        if return_intermediates:
            return epitope, all_generated_cdr_states

        return epitope, all_generated_cdr_states[-1]


class CDRFrameDiffusionModel(CDRCoordinateDiffusionModel):
    """
    Diffusion model that generates CDR structures by
    diffusing over residue frames, where each frame is a combination
    of a 3-D rotation (backbone orientation) and a 3-D translation (CA position).

    This class is implemented as an extension of CDRCoordinateDiffusionModel
    that still defines diffusion processes over coordinates, but also
    includes a rotation diffusion process.
    """

    network_class: Type[nn.Module] = GVPSE3ScorePredictor
    datamodule_class: Type[LightningDataModule] = CDRFrameDataModule

    def __init__(
        self,
        network: nn.Module,
        learning_rate: float = 1e-4,  # only needed if training
        self_conditioning_rate: float = 0.5,
        rotation_beta_schedule: BetaScheduleTypes = "logarithmic",
        translation_beta_schedule: BetaScheduleTypes = "linear",
        num_time_steps: int = 100,
        min_rot_beta: float = 0.1,
        max_rot_beta: float = 1.5,
        min_trans_beta: float = 1e-4,
        max_trans_beta: float = 20,
        weight_loss_by_norm: bool = True,
        igso3_support_n: int = 2000,
        igso3_expansion_n: int = 2000,
        use_igso3_cache: bool = False,
        n_score_samples: int = 50000,
        pad_feature_value: float = 0.0,
        time_step_encoding_channels: int = 5,
        use_cdr_positional_encoding: bool = True,
        positional_encoding_channels: int = 5,
        test_results_filepath: str = "test_results.csv",  # for storing prediction results on the test set
    ):
        """
        :param network: Network used to predict the rotation and translation scores.
        :param learning_rate: Optimizer learning rate.
        :param self_conditioning_rate: The rate at which samples for training "self-conditioning"
            will be used. This is the proportion of training steps in which the model will make a prediction
            on samples drawn from q(x_{t+1} | x_t) and use that information
            to condition its predictions for q(x_t | x_0). Implemented the same way as RFDiffusion.
        :param rotation_beta_schedule: Type of variance schedule used for rotations.
        :param translation_beta_schedule: Type of variance schedule used for translations.
        :param num_time_steps: Number of diffusion time steps.
        :param min_rot_beta: Minimum variance of the rotation diffusion process.
        :param max_rot_beta: Maximum variance of the rotation diffusion process.
        :param min_trans_beta: Minimum variance of the translation diffusion process.
        :param max_trans_beta: Maximum variance of the translation diffusion process.
        :param weight_loss_by_norm: Whether to weight the loss by the norm of the predicted
            rotation and translation scores.
        :param igso3_support_n: Number of points sampled to approximate the
            support of the IGSO(3) distribution.
        :param igso3_expansion_n: Number of terms used to truncate the
            infinite sum in the IGSO(3) PDF.
        :param use_igso3_cache: Whether to load IGSO(3) PDF values from the cache (or save them
            to the cache if they cannot be found). Defaults to False so that IGSO(3) values from multiple
            runs with multiple variance schedules do not get mixed up.
        :param n_score_samples: Number of samples to use to estimate the mean norm of the score
            at each step of the forward process.
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
        :param test_results_filepath: Filepath under which test results will be saved.
        """
        super().__init__(
            network,
            learning_rate,
            self_conditioning_rate,
            translation_beta_schedule,
            num_time_steps,
            min_trans_beta,
            max_trans_beta,
            weight_loss_by_norm,
            pad_feature_value,
            time_step_encoding_channels,
            use_cdr_positional_encoding,
            positional_encoding_channels,
            test_results_filepath,
        )

        self._add_pad_cdr_features = self._using_self_conditioning

        get_rot_schedule = BetaScheduleSelector(
            num_time_steps, min_rot_beta, max_rot_beta
        )
        self.register_buffer(
            "_rot_beta_schedule", get_rot_schedule(rotation_beta_schedule)
        )

        self._igso3_support_n = igso3_support_n
        self._igso3_expansion_n = igso3_expansion_n

        # forward process
        self._rotation_forward_process = IGSO3ForwardProcess(
            self._rot_beta_schedule,
            self._igso3_support_n,
            self._igso3_expansion_n,
            use_cache=use_igso3_cache,
        )

        if self._using_self_conditioning:
            self._num_pad_cdr_features = 3
            self._num_pad_cdr_vec_features = 1

        # estimate the mean and variance of the norms of the rotation scores
        # at each time step in the forward process using some simulated
        # data as inputs - these average norms will be used to weight each loss
        # term depending on the time step
        with isolate_rng():
            random_rots = torch.as_tensor(
                Rotation.random(num=n_score_samples, random_state=123).as_matrix(),
                dtype=torch.float32,
                device=self.device,
            )

        self._mean_rot_score_norms = []
        for i in range(self._num_time_steps):
            _, rot_scores = self._rotation_forward_process.sample(random_rots, i)
            mean_rot_score_norm = torch.linalg.norm(rot_scores, dim=-1).mean().item()
            self._mean_rot_score_norms.append(mean_rot_score_norm)

        self._rotation_stationary_dist = IGSO3Distribution(
            support_n=igso3_support_n, expansion_n=igso3_expansion_n, uniform=True
        )

        self._rotation_reverse_process = SO3ReverseProcess(self._rot_beta_schedule)

    def sample_forward(
        self,
        cdr_frames: OrientationFrames,
        epitope: Optional[Structure],
        time_step: torch.Tensor,
    ) -> ForwardProcessOutput:
        """
        For an input epitope/CDR structure and time step, returns a graph with noised CDR frames
        sampled from q(x_t | x_0). If self-conditioning, also returns a graph with
        noised CDR frames sampled from q(x_{t+1} | x_t).
        """
        noised_rots, rot_scores = self._rotation_forward_process.sample(
            cdr_frames.rotations, time_step.item()
        )
        noised_coords, trans_scores = self._translation_forward_process.sample(
            cdr_frames.translations, time_step.item()
        )

        noised_cdr_frames = OrientationFrames(
            noised_rots, noised_coords, batch=cdr_frames.batch, ptr=cdr_frames.ptr
        )

        noised_graph = self.create_graph(
            noised_cdr_frames,
            epitope,
            time_step_encoding=self._time_step_encodings[time_step],
        )

        # stochastically determine whether to return an additional sample for self-conditioning
        training_self_conditioning = random() < self._self_conditioning_rate
        if training_self_conditioning:
            # if the time step is the last one, sample from the stationary distribution
            if time_step.item() == self._num_time_steps - 1:
                self_cond_rots = self._rotation_stationary_dist.sample(
                    len(noised_cdr_frames.rotations), device=self._device
                )
                self_cond_coords = self._translation_stationary_dist.sample(
                    len(noised_cdr_frames.translations), device=self._device
                )
            else:
                # samples for self-conditioning are conditioned on the noised CDR
                self_cond_rots, _ = self._rotation_forward_process.sample(
                    noised_cdr_frames.rotations,
                    (time_step + 1).item(),
                    from_x0=False,
                )
                self_cond_coords, _ = self._translation_forward_process.sample(
                    noised_cdr_frames.translations,
                    (time_step + 1).item(),
                    from_x0=False,
                )

            self_cond_cdr_frames = OrientationFrames(
                self_cond_rots,
                self_cond_coords,
                batch=cdr_frames.batch,
                ptr=cdr_frames.ptr,
            )
            self_cond_graph = self.create_graph(
                self_cond_cdr_frames,
                epitope,
                time_step_encoding=self._time_step_encodings[time_step],
            )
        else:
            self_cond_graph = None

        scores = (rot_scores, trans_scores)

        return scores, noised_graph, self_cond_graph

    def forward(self, graph: ProteinGraph, time_step: torch.Tensor) -> PairTensor:
        """
        Returns a tuple consisting of two tensors,
        one an SE(3)-invariant 3-vector representing
        the rotation score and the other an SE(3)-equivariant
        translation score tensor of shape (N, 3).
        """
        pred_rot_scores, pred_trans_scores = self._network(graph)

        if self._network.is_hetero:
            cdr_rot_scores = pred_rot_scores["ligand"]
            cdr_trans_scores = pred_trans_scores["ligand"]
        elif hasattr(graph, "node_type"):
            cdr_mask = graph.node_type == 1
            cdr_rot_scores = pred_rot_scores[cdr_mask]
            cdr_trans_scores = pred_trans_scores[cdr_mask]
        else:
            cdr_rot_scores = pred_rot_scores
            cdr_trans_scores = pred_trans_scores

        cdr_trans_scores = cdr_trans_scores[..., 0, :]

        return cdr_rot_scores, cdr_trans_scores

    @staticmethod
    def self_condition_rotations(
        graph: ProteinGraph,
        pred_rot_scores: torch.Tensor,
    ) -> ProteinGraph:
        """
        Adds self-conditioning information - i.e predicted rotation scores obtained by
        running a forward pass - to `graph`, returning the modified graph.
        This modifies `graph` inplace.
        """

        self_cond_x = get_cdr_feature(graph, "x").clone()
        rot_score_dim = pred_rot_scores.shape[-1]
        self_cond_x[..., -rot_score_dim:] = pred_rot_scores.detach()

        graph = replace_cdr_features(graph, self_cond_x)

        return graph

    def self_condition(
        self, graph: ProteinGraph, scores: Tuple[torch.Tensor, torch.Tensor]
    ):
        """Adds self-conditioning information to the graph."""
        rot_scores, trans_scores = scores
        graph = self.self_condition_translations(graph, trans_scores)
        graph = self.self_condition_rotations(graph, rot_scores)
        return graph

    def sample_stationary(
        self,
        size: int,
        batch: Optional[torch.Tensor] = None,
    ) -> OrientationFrames:
        """
        Draw an OrientationFrames object from the stationary distribution.
        The rotations are sampled from the uniform IGSO(3) distribution, and the translations
        are sampled from a standard normal.

        :param size: Number of frames to sample.
        :param batch: Optional batch assignment tensor of shape (size,) for the sampled frames,
            assigning each frame to a batch element. This can be used to generate a batch of
            frames from the stationary distribution.
        :returns: OrientationFrames object.
        """

        random_rotations = self._rotation_stationary_dist.sample(
            size=(size,), device=self._device
        )
        random_translations = self._translation_stationary_dist.sample(
            size=(size,), device=self._device
        )
        random_cdr_frames = OrientationFrames(
            random_rotations,
            random_translations,
            batch=batch,
        )

        return random_cdr_frames

    def _update_cdr(
        self,
        scores: Tuple[torch.Tensor, torch.Tensor],
        current_cdr: OrientationFrames,
        time_step: torch.Tensor,
        noise_scale: float,
        self_cond: bool,
    ) -> OrientationFrames:
        """
        Updates the CDR backbone conformation by performing a reverse diffusion step,
        returning the updated CDR OrientationFrames object.
        """
        rot_scores, trans_scores = scores
        int_time_step = time_step.item()

        new_cdr_positions = self._translation_reverse_process(
            current_cdr.translations,
            trans_scores,
            int_time_step,
        )

        new_cdr_rotations = self._rotation_reverse_process(
            current_cdr.rotations,
            rot_scores,
            int_time_step,
        )

        updated_cdr = OrientationFrames(
            new_cdr_rotations,
            new_cdr_positions,
            batch=current_cdr.batch,
            ptr=current_cdr.ptr,
        )

        return updated_cdr

    def calculate_loss(
        self, batch: CDRFramesBatch, time_step: torch.Tensor
    ) -> PairTensor:
        """
        Runs a forward pass and calculates MSE loss for rotational and translational scores.
        The returned loss tensors contain the MSE loss for each individual score, stored
        within a rank 1 tensor.
        """
        names, epitope, cdr = batch
        (
            scores,
            graph,
            self_cond_graph,
        ) = self.sample_forward(cdr, epitope, time_step)

        rot_scores, trans_scores = scores

        if self_cond_graph is not None:
            self_cond_scores = self(self_cond_graph, time_step + 1)
            graph = self.self_condition(graph, self_cond_scores)

        # predict scores and get losses
        pred_rot_scores, pred_trans_scores = self(graph, time_step)

        rot_score_errors = self._loss_fn(pred_rot_scores, rot_scores)
        trans_score_errors = self._loss_fn(pred_trans_scores, trans_scores)

        rot_score_mse = torch.sum(rot_score_errors, dim=-1)
        trans_score_mse = torch.mean(trans_score_errors, dim=-1)

        if self._weight_loss_by_norm:
            rot_score_mse /= self._mean_rot_score_norms[time_step.item()] ** 2

        return rot_score_mse, trans_score_mse

    def training_step(self, batch: CDRFramesBatch) -> torch.Tensor:
        """
        Predicts 2 score vectors (rotation/translation) for each residue
        in the batch and computes the loss between the predicted and
        ground truth scores.
        """
        time_step = self.sample_time_step()

        rot_score_mse, trans_score_mse = self.calculate_loss(batch, time_step)

        mean_rot_loss = torch.mean(rot_score_mse)
        mean_trans_loss = torch.mean(trans_score_mse)

        self.log("rotation_train_loss", mean_rot_loss.item())
        self.log("translation_train_loss", mean_trans_loss.item())
        self.log("time_step", float(time_step.item()))

        return mean_rot_loss + mean_trans_loss

    def on_validation_epoch_end(self) -> None:
        """
        Calculates the mean rotational and translational score losses
        on the whole validation set.
        """
        rot_score_mse, trans_score_mse = map(torch.cat, zip(*self._val_outputs))

        mean_rot_loss = torch.mean(rot_score_mse)
        mean_trans_loss = torch.mean(trans_score_mse)

        self.log("rotation_val_loss", mean_rot_loss.item())
        self.log("translation_val_loss", mean_trans_loss.item())
        self.log("validation_loss", (mean_rot_loss + mean_trans_loss).item())

        self._val_outputs.clear()

    def on_test_epoch_end(self) -> None:
        """
        Calculates the mean rotational and translational score losses
        on the whole test set.
        """
        rot_score_mse, trans_score_mse = map(torch.cat, zip(*self._test_outputs))

        mean_rot_loss = torch.mean(rot_score_mse)
        mean_trans_loss = torch.mean(trans_score_mse)

        self.log("rotation_test_loss", mean_rot_loss.item())
        self.log("translation_test_loss", mean_trans_loss.item())
        self.log("test_loss", (mean_rot_loss + mean_trans_loss).item())

    def generate(
        self,
        batch: CDRFramesBatch,
        noise_scale: float = 0.2,
        self_cond: bool = True,
        return_intermediates: bool = False,
        n: int = 1,
    ) -> Union[
        Tuple[Structure, List[OrientationFrames]],
        Tuple[Structure, List[List[OrientationFrames]]],
    ]:
        """
        Initialises the CDR with random rotations and translations and
        performs the reverse diffusion process over CDR frames (rotations and translations_
        to generate a CDR backbone conformation.

        :param batch: 3-tuple containing the names of the CDRs in the batch,
            the epitope structure, and the CDR orientation frames.
        :param noise_scale: Noise scale for the reverse diffusion process.
        :param self_cond: Whether to use self-conditioning information.
        :param return_intermediates: Whether to return a list of intermediate CDRs.
            :param n: Number of CDRs to generate for each epitope.
        :returns: 2-tuple containing the epitope Structure and the n generated CDR OrientationFrames
            objects. If return_intermediates is True, returns a list of 2-tuples containing the
            epitope Structure and the generated CDR OrientationFrames objects at each time step.
        """

        old_rot_noise_scale = self._rotation_reverse_process.noise_scale
        self._rotation_reverse_process.noise_scale = noise_scale

        output = super().generate(
            batch, noise_scale, self_cond, return_intermediates, n
        )

        self._rotation_reverse_process.noise_scale = old_rot_noise_scale

        return output
