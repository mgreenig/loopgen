"""
This is a sub-package for code used to build specific deep learning model.
"""
from typing import Type, Literal, Optional, Tuple, Union, Dict, Set

import logging as lg

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.seed import isolate_rng

from .settings import ModelSettings, TrainSettings
from .network import GVPR3ScorePredictor, GVPSE3ScorePredictor
from .datamodule import CDRFrameDataModule, CDRCoordinateDataModule
from .model import CDRFrameDiffusionModel, CDRCoordinateDiffusionModel
from .types import ParamDictionary

from ..data import ReceptorLigandDataset, load_splits_file


def load_trained_model(
    model_class: Union[Type[CDRCoordinateDiffusionModel], Type[CDRFrameDiffusionModel]],
    checkpoint_path: str,
    settings_path: str,
    dataset_path: Optional[str] = None,
    splits_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    strict: bool = False,
) -> Tuple[LightningDataModule, LightningModule]:
    """
    Loads a trained model and its datamodule from a checkpoint and settings file.

    :param model_class: The model class that was trained.
    :param checkpoint_path: Path to the checkpoint file.
    :param settings_path: Path to the settings YAML file.
    :param dataset_path: The path to the HDF5 file that was used for training - this is optional, but if
        provided, the dataset will be passed to the returned datamodule.
    :param splits_path: Dictionary containing the train/test/validation splits as string names
        of the instances in the dataset, stored under the respective keys "train", "test", and "validation".
    :param device: The device to load the model on.
    :param strict: Whether to load the model in strict mode, i.e. throw an error if the parameters
        in the checkpoint file do not match the parameters in the model class.
    :returns: The trained model and its associated datamodule.
    """
    if dataset_path is not None:
        dataset = ReceptorLigandDataset.from_hdf5_file(dataset_path, device=device)
    else:
        dataset = None

    if splits_path is not None and dataset is not None:
        splits = load_splits_file(splits_path, dataset)
    else:
        splits = None

    settings = TrainSettings.from_yaml(settings_path)
    param_dict = settings.distribute_model_params(model_class)
    datamodule = model_class.datamodule_class(
        dataset=dataset,
        splits=splits,
        **param_dict[model_class.datamodule_class.__name__],
    )
    example_batch = datamodule.generate_example()
    network = model_class.network_class(
        example_batch, **param_dict[model_class.network_class.__name__]
    )
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        network=network,
        map_location=device,
        strict=strict,
        **param_dict[model_class.__name__],
    )
    return datamodule, model


def setup_model(
    dataset: ReceptorLigandDataset,
    splits: Dict[str, Set[str]],
    param_dict: ParamDictionary,
    model_class: Union[Type[CDRCoordinateDiffusionModel], Type[CDRFrameDiffusionModel]],
    accelerator: Literal["cpu", "gpu"],
    checkpoint_path: Optional[str] = None,
) -> Tuple[LightningDataModule, LightningModule]:
    """
    Using a dataset and a param dictionary storing arguments
    for the model's datamodule class, network class, and the model class itself,
    generate a datamodule and model instance.

    :param dataset: The dataset to be used for training/evaluation.
    :param splits: Dictionary containing the train/test/validation splits as string names
        of the instances in the dataset, stored under the respective keys "train", "test", and "validation".
    :param param_dict: A dictionary of parameters for the model.
    :param model_class: The model class to be used.
    :param accelerator: The accelerator (device) to be used.
    :param checkpoint_path: A path to a PyTorch checkpoint file containing model weights.
    :returns: A tuple of the datamodule and model instances.
    """

    datamodule = model_class.datamodule_class(
        dataset,
        splits,
        **param_dict[model_class.datamodule_class.__name__],
    )

    with isolate_rng():
        example_batch = datamodule.generate_example()

        network = model_class.network_class(
            example_batch, **param_dict[model_class.network_class.__name__]
        )

        if checkpoint_path is None:
            model = model_class(network=network, **param_dict[model_class.__name__])
        else:
            lg.info(f"Loading model from checkpoint {checkpoint_path}...")
            model = model_class.load_from_checkpoint(
                checkpoint_path,
                network=network,
                **param_dict[model_class.__name__],
            )

    return datamodule, model
