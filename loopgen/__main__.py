"""
Command line interface for loopgen.
"""

import argparse
import warnings

from functools import partial

from .model import CDRCoordinateDiffusionModel, CDRFrameDiffusionModel
from .model.train import add_train_args, train_from_args
from .model.test import add_test_args, test_from_args


USAGE = """
%(prog)s <model> <command> [options]

LoopGen: De novo design of peptide CDR binding loops with SE(3) diffusion models.

Currently, loopgen supports two models:
    - frames: Diffusion over the SE(3), i.e. a 3D rotation and translation for each residue.
    - coords: Diffusion over R3, i.e. a 3D translation for each residue.

Each model supports two commands:
    - train: Trains a new or saved model on a dataset in HDF5 format.
    - evaluate: Evaluates a saved model on a dataset in HDF5 format.
"""


def main():
    """
    Runs the loopgen program based on the provided commands.
    """

    # filter pytorch lightning warning about number of workers in dataloader
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage=USAGE,
    )

    subparser = parser.add_subparsers(title="commands")

    frames_parser = subparser.add_parser(
        "frames",
        description="Loop diffusion over SE(3), modelling a 3D rotation (orientation) "
        "and translation (CA coordinate) for each residue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    frames_subparser = frames_parser.add_subparsers(title="commands")

    coords_parser = subparser.add_parser(
        "coords",
        description="Loop diffusion over R3, modelling a 3D translation (CA coordinate) for each residue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    coords_subparser = coords_parser.add_subparsers(title="commands")

    for subp, model in zip(
        [frames_subparser, coords_subparser],
        [CDRFrameDiffusionModel, CDRCoordinateDiffusionModel],
    ):
        # Adds the parsers for `train` and `evaluate` to the parser for each model
        train_parser = subp.add_parser(
            "train",
            description=f"Train a model specified by a config YAML file.",
            usage="train [options]",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        add_train_args(train_parser)

        train_parser.set_defaults(func=partial(train_from_args, model_class=model))

        test_parser = subp.add_parser(
            "test",
            description=f"Test a model specified by a config YAML file.",
            usage="test [options]",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        add_test_args(test_parser)

        test_parser.set_defaults(func=partial(test_from_args, model_class=model))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
