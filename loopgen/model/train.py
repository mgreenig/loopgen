"""
Contains a generic train() function that can be used across all model.
"""

from typing import Type, Literal, Optional, Dict, Set
import logging as lg
import os
import argparse
import sys

import h5py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .settings import TrainSettings, get_mlflow_logger
from . import setup_model

from .. import ReceptorLigandDataset, get_device
from ..data import load_splits_file


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds command line arguments for training to a parser.
    """

    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the HDF5 data file to be used for training/testing.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        required=True,
        help="Path to a JSON file containing the names of instances stored under "
        "the keys 'train', 'validation', and 'test'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML file containing the settings.",
    )
    parser.add_argument(
        "-e",
        "--n_epochs",
        type=int,
        required=True,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a PyTorch checkpoint file containing model weights.",
    )
    parser.add_argument(
        "--restore_full_state",
        action="store_true",
        help="Whether to restore the full training state from the provided checkpoint. Only"
        "applicable if --checkpoint is passed.",
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
        "-t", "--test", action="store_true", help="Whether to run as a test run"
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "gpu"],
        help="Which device to use (defaults to None, in which case the "
        "GPU is used if available, and if not the CPU is used).",
    )


def train(
    dataset: ReceptorLigandDataset,
    splits: Dict[str, Set[str]],
    settings: TrainSettings,
    model_class: Type[pl.LightningModule],
    out_dir: str,
    num_epochs: int,
    checkpoint: Optional[str],
    restore_full_state: bool,
    accelerator: Literal["cpu", "gpu"],
    test: bool = False,
) -> None:
    """
    Trains a model specified by a settings YAML file.

    :param dataset: The dataset to use for training.
    :param splits: Dictionary containing the train/test/validation splits as string names
        of the instances in the dataset, stored under the respective keys "train", "test", and "validation".
    :param settings: The settings for the model.
    :param model_class: The model class to use.
    :param out_dir: The directory in which to save the results.
    :param num_epochs: The number of epochs to train for.
    :param checkpoint: The path to a PyTorch checkpoint file containing model weights.
    :param restore_full_state: Whether to restore the full state of the Trainer (including
        optimizer state, schedulers, etc.) from the provided checkpoint. Only applicable if
        checkpoint is passed.
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
    run_name = settings.run_name + " test" if test else settings.run_name
    mlflow_logger = get_mlflow_logger(exp_name, run_name, settings)

    checkpoint_path = os.path.join(out_dir, f"{settings.checkpoint_outfile}.ckpt")

    i = 1
    while os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(
            out_dir, f"{settings.checkpoint_outfile}-v{i}.ckpt"
        )
        i += 1

    checkpoint_args = settings.distribute_params(ModelCheckpoint)["ModelCheckpoint"]

    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        monitor=model.checkpoint_metric,
        filename=checkpoint_path,
        **checkpoint_args,
    )

    trainer_args = settings.distribute_params(pl.Trainer)["Trainer"]
    trainer_args["fast_dev_run"] = test

    if "accelerator" not in trainer_args:
        trainer_args["accelerator"] = accelerator
    if "enable_progress_bar" not in trainer_args:
        trainer_args["enable_progress_bar"] = False

    trainer = pl.Trainer(
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        **trainer_args,
    )

    try:
        if restore_full_state:
            trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)
        else:
            trainer.fit(model, datamodule=datamodule)
    except RuntimeError as e:
        lg.error(
            f"RuntimeError: {e} Saving model checkpoint at error to: {checkpoint_path}"
        )
        trainer.save_checkpoint(checkpoint_path)
        sys.exit(1)

    model = model_class.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        network=model.network,
        **param_dict[model_class.__name__],
    )

    trainer.test(model, datamodule=datamodule)


def train_from_args(
    args: argparse.Namespace, model_class: Type[pl.LightningModule]
) -> None:
    """Runs `train()` using command line arguments."""
    device, accelerator = get_device(args.device)

    with h5py.File(args.data_path) as hdf5_file:
        dataset = ReceptorLigandDataset.from_hdf5_file(hdf5_file, device=device)
        splits = load_splits_file(args.splits, dataset)
        settings = TrainSettings.from_yaml(args.config)

        n_epochs = args.n_epochs
        out_dir = args.out_dir
        checkpoint = args.checkpoint
        restore_full_state = args.restore_full_state
        test = args.test

        if hasattr(settings, "out_dir"):
            if args.out_dir != ".":
                lg.warning(
                    "Out directory found in settings file. Command line will supercede settings.",
                )
            else:
                out_dir = settings.out_dir
        if hasattr(settings, "run_name"):
            if args.run_name:
                lg.warning(
                    "Run name also found in settings file. Command line will supercede settings.",
                )
                settings.run_name = args.run_name
        if hasattr(settings, "checkpoint"):
            if args.checkpoint:
                lg.warning(
                    "Checkpoint found in settings file. "
                    "Command line will supercede settings and load from checkpoint.",
                )
            else:
                checkpoint = settings.checkpoint
        if hasattr(settings, "test"):
            if args.test:
                lg.warning(
                    "Test status found in settings file. Command line will supercede settings.",
                )
            else:
                test = settings.test

        train(
            dataset,
            splits,
            settings,
            model_class,
            out_dir,
            n_epochs,
            checkpoint,
            restore_full_state,
            accelerator,
            test,
        )
