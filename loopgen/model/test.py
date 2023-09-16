""" Contains a generic evaluate() function that can be used across all model. """

from typing import Type, Literal, Optional, Dict, Set
import logging as lg
import os
import argparse

import h5py
import pandas as pd
import pytorch_lightning as pl

from . import setup_model
from .settings import get_mlflow_logger, ModelSettings
from .. import ReceptorLigandDataset, get_device


def add_test_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds command line arguments for testing to a parser.
    """

    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the HDF5 data file to be used for training/testing.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML file containing the settings.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Optional path to a JSON file containing the names of instances stored under "
        "the keys 'train', 'validation', and 'test'. If not provided, the entire dataset "
        "will be used for testing.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a PyTorch checkpoint file containing model weights.",
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Directory in which results (predictions) will be saved.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Name of the MLFlow run under which the current run data will be saved.",
    )
    parser.add_argument(
        "-t", "--test", action="store_true", help="Whether to run as a test run."
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "gpu"],
        help="Which device to use (defaults to None, in which case the "
        "GPU is used if available, and if not the CPU is used).",
    )


def test(
    dataset: ReceptorLigandDataset,
    splits: Dict[str, Set[str]],
    settings: ModelSettings,
    model_class: Type[pl.LightningModule],
    out_dir: str,
    checkpoint: Optional[str],
    run_name: Optional[str],
    accelerator: Literal["cpu", "gpu"],
    test: bool = False,
) -> None:
    """
    Evaluates a model specified by a settings YAML file.

    :param dataset: The dataset to use for training.
    :param splits: Dictionary containing the train/test/validation splits as string names
        of the instances in the dataset, stored under the respective keys "train", "test", and "validation".
    :param settings: The settings for the model.
    :param model_class: The model class to use.
    :param out_dir: The directory in which to save the results.
    :param checkpoint: The path to a PyTorch checkpoint file containing model weights.
    :param run_name: The name of the MLFlow run under which the current run data will be saved.
    :param accelerator: Which device to use
        (defaults to None, in which case the GPU is used if available, and if not the CPU is used).
    :param test: Whether to run as a test run.
    """

    pl.seed_everything(123, workers=True)

    param_dict = settings.distribute_model_params(model_class)

    if "test_results_filepath" not in param_dict[model_class.__name__]:
        param_dict[model_class.__name__]["test_results_filepath"] = os.path.join(
            out_dir, "test_results.csv"
        )

    lg.basicConfig(format="%(asctime)s %(levelname)-8s: %(message)s")

    if not os.path.exists(out_dir):
        lg.info(f"Specified output directory {out_dir} does not exist, creating...")
        os.mkdir(out_dir)

    datamodule, model = setup_model(
        dataset, param_dict, model_class, accelerator, checkpoint, splits
    )

    exp_name = settings.experiment_name + " test" if test else settings.experiment_name

    mlflow_logger = get_mlflow_logger(exp_name, run_name, settings)
    trainer_args = settings.distribute_params(pl.Trainer)["Trainer"]
    trainer_args["fast_dev_run"] = test

    if "max_epochs" not in trainer_args:
        trainer_args["max_epochs"] = 1
    if "accelerator" not in trainer_args:
        trainer_args["accelerator"] = accelerator
    if "enable_progress_bar" not in trainer_args:
        trainer_args["enable_progress_bar"] = False

    trainer = pl.Trainer(
        logger=mlflow_logger,
        **trainer_args,
    )

    trainer.test(model, datamodule=datamodule)


def test_from_args(
    args: argparse.Namespace, model_class: Type[pl.LightningModule]
) -> None:
    """Runs `test()` using command line arguments."""
    device, accelerator = get_device(args.device)

    with h5py.File(args.data_path) as hdf5_file:
        dataset = ReceptorLigandDataset.from_hdf5_file(hdf5_file, device=device)
        settings = ModelSettings.from_yaml(args.settings)

        if args.metadata is not None:
            metadata = pd.read_csv(args.metadata)
        else:
            metadata = None

        test(
            dataset,
            metadata,
            settings,
            model_class,
            args.out_dir,
            args.checkpoint,
            args.run_name,
            accelerator,
            args.test,
        )
