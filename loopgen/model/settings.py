"""
Defines the required settings needed to specify a deep learning model.
"""

from __future__ import annotations
from typing import Optional, Callable, Dict, Any, Type, Literal
from datetime import date

import yaml

from inspect import signature, Parameter
from collections import defaultdict
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import MLFlowLogger

from .types import ParamDictionary
from ..graph import EdgeIndexMethods


class ModelSettings:
    """
    Base class for an object specifying the settings for a deep learning model.
    At the moment these are loaded from YAML files but other format parsers can be
    added in the future.
    """

    def __init__(
        self,
        experiment_name: str = "Experiment",
        run_name: str = "run",
        steps_per_log: int = 100,
        **kwargs: Any,  # additional parameters specific to particular models
    ):
        self.experiment_name = experiment_name
        self.steps_per_log = steps_per_log

        # run_name is protected so that we can define a setter method in child classes
        # that modifies other attributes if the run name is changed
        self._run_name = run_name

        for attr, value in kwargs.items():
            setattr(self, attr, value)

        self._params = kwargs
        self._param_names = set(self._params)

    @classmethod
    def from_yaml(cls, path: str):
        """Loads the settings from a YAML file."""
        with open(path) as file:
            settings = yaml.safe_load(file)
        cls._check_settings(settings)
        return cls(**settings)

    @property
    def run_name(self) -> str:
        """The name of the current run."""
        return self._run_name

    @run_name.setter
    def run_name(self, value: str):
        """Sets the run name."""
        self._run_name = value

    def distribute_params(self, *callables: Callable) -> ParamDictionary:
        """
        For input callables, distribute the parameters in the `params`
        attribute by searching for their name (key) in each callable's signature.

        Returns a dictionary mapping each callable name to a dictionary of key word
        arguments to be passed to that callable.
        """
        cl_param_dict = defaultdict(dict)
        for cl in callables:
            cl_signature = signature(cl)
            cl_params = set(cl_signature.parameters)
            cl_params_in_settings = self._param_names.intersection(cl_params)
            cl_provided_params = {}
            for p in cl_params_in_settings:
                cl_provided_params[p] = self._params[p]

            cl_param_dict[cl.__name__] = cl_provided_params

        return cl_param_dict

    def distribute_model_params(
        self, model_class: Type[LightningModule]
    ) -> ParamDictionary:
        """
        Distributes the params for a DL model between the model itself,
        the model's datamodule class, and the model's network class.
        All classes (model, datamodule, network) have parameters
        distributed to the constructor.
        """

        param_dict = self.distribute_params(
            model_class,
            model_class.datamodule_class,
            model_class.network_class,
        )

        # add edge index methods to the datamodule signature
        datamodule_params = signature(model_class.datamodule_class).parameters
        if "edge_method" in datamodule_params:
            datamodule_name = model_class.datamodule_class.__name__
            if "edge_method" in param_dict[datamodule_name]:
                edge_index_method = param_dict[datamodule_name]["edge_method"]
                edge_index_class = EdgeIndexMethods[edge_index_method].value
            elif datamodule_params["edge_method"].default != Parameter.empty:
                default_method = datamodule_params["edge_method"].default
                edge_index_class = EdgeIndexMethods[default_method].value
            else:
                raise ValueError(
                    "Edge method not provided and no default method found."
                )

            edge_index_param_dict = self.distribute_params(edge_index_class)
            param_dict[datamodule_name]["edge_kwargs"] = edge_index_param_dict[
                edge_index_class.__name__
            ]

        return param_dict

    @classmethod
    def _check_settings(cls, settings: Dict[str, Any]) -> None:
        """
        Checks a settings dictionary and raises an error if any of
        the provided settings are missing or invalid.
        """

        # check if all requires arguments provided
        constructor_sig = signature(cls.__init__)
        required_slots = {
            param_name
            for param_name, param in constructor_sig.parameters.items()
            if param.default == Parameter.empty and param_name not in {"kwargs", "self"}
        }

        if not required_slots.issubset(settings):
            raise ValueError(
                f"One or more of the required fields: {required_slots} was not found in the input."
            )


class TrainSettings(ModelSettings):

    """
    Extends ModelSettings with additional parameters specific for training.
    """

    def __init__(
        self,
        experiment_name: str = "Experiment",
        run_name: str = "run",
        steps_per_log: int = 100,
        checkpoint_outfile: Optional[str] = None,
        checkpoint_metric: str = "validation_loss",
        checkpoint_mode: Literal["min", "max"] = "min",
        save_top_k: int = 1,
        **kwargs: Any,  # additional parameters specific to particular models, saved under self._params
    ):
        super().__init__(experiment_name, run_name, steps_per_log, **kwargs)

        self.checkpoint_metric = checkpoint_metric

        self._checkpoint_outfile_provided = checkpoint_outfile is not None

        if self._checkpoint_outfile_provided:
            self.checkpoint_outfile = checkpoint_outfile
        else:
            self.checkpoint_outfile = (
                f"{self._run_name}-{date.today()}-"
                f"{{epoch:02d}}-{{{checkpoint_metric}:.2f}}"
            )

        self.checkpoint_mode = checkpoint_mode
        self.save_top_k = save_top_k

    @ModelSettings.run_name.setter
    def run_name(self, value: str):
        """
        Sets the run name and changes the checkpoint outfile name
        accordingly if it was not provided to the constructor.
        """
        self._run_name = value
        if not self._checkpoint_outfile_provided:
            self.checkpoint_outfile = (
                f"{self._run_name}-{date.today()}-"
                f"{{epoch:02d}}-{{{self.checkpoint_metric}:.2f}}"
            )

def get_mlflow_logger(
    exp_name: str, run_name: str, settings: ModelSettings
) -> MLFlowLogger:
    """
    Returns an MLFlow logger object from pytorch lightning with
    the settings logged.
    """
    mlflow_logger = MLFlowLogger(experiment_name=exp_name, run_name=run_name)
    mlflow_logger.log_hyperparams(
        {param: value for param, value in settings.__dict__.items() if param[0] != "_"}
    )

    return mlflow_logger
