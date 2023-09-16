""" Tests the dataset class."""

import pytest
import numpy as np
import os
import torch
from loopgen import ReceptorLigandDataset, Structure
from loopgen.data import BaseDataModule


@pytest.fixture
def data_path() -> str:
    """Returns the path to the test data set."""
    return os.path.join(os.path.dirname(__file__), "data/cdrs_20.hdf5")


@pytest.fixture
def dataset(data_path) -> ReceptorLigandDataset:
    """Returns the actual data set object for the test data set."""
    return ReceptorLigandDataset.from_hdf5_file(data_path)


@pytest.fixture
def datamodule(dataset) -> BaseDataModule:
    """Returns a non-abstract version of the BaseCDRDataModule class."""
    datamodule_class = BaseDataModule
    datamodule_class.__abstractmethods__ = set()
    return datamodule_class(dataset)


@pytest.fixture
def datamodule_with_cdr_types(dataset) -> BaseDataModule:
    """Returns a non-abstract version of the BaseCDRDataModule class with cdr types."""
    datamodule_class = BaseDataModule
    datamodule_class.__abstractmethods__ = set()
    cdr_types = {
        pair["name"]: str(int(i % 2 == 0))
        for i, pair in enumerate(dataset.structure_pairs)
    }
    cdr_type_weights = {"0": 1, "1": 3}
    return datamodule_class(
        dataset,
        cdr_type_dict=cdr_types,
        cdr_type_weights=cdr_type_weights,
        name_to_id_fn=lambda x: x,
    )


class TestReceptorLigandDataset:
    def test_from_hdf5_file(self, data_path):
        dataset = ReceptorLigandDataset.from_hdf5_file(data_path)

        for pair_dict in dataset.structure_pairs:
            assert "name" in pair_dict, "Structure pair should have a name"
            assert "antigen" in pair_dict, "Structure pair should have an antigen"
            assert "cdr" in pair_dict, "Structure pair should have a cdr"

    def test_len(self, dataset):
        assert len(dataset) == len(
            dataset.structure_pairs
        ), "Length of dataset should be same as length of the list of structure pairs"

    def test_getitem(self, dataset):
        for i in range(20):
            name, epitope, cdr = dataset[i]
            assert isinstance(name, str), "Name should be a string"
            assert isinstance(epitope, Structure), "Epitope should be a Structure"
            assert isinstance(cdr, Structure), "CDR should be a Structure"

            assert dataset.structure_pairs[i]["name"] == name, "Name should match"

            assert np.allclose(
                dataset.structure_pairs[i]["antigen"]["N_coords"][:],
                epitope.N_coords.cpu().numpy(),
            ), "N coords of returned antigen should match those in dataset file"

            assert np.allclose(
                dataset.structure_pairs[i]["antigen"]["CA_coords"][:],
                epitope.CA_coords.cpu().numpy(),
            ), "CA coords of returned antigen should match those in dataset file"

            assert np.allclose(
                dataset.structure_pairs[i]["antigen"]["C_coords"][:],
                epitope.C_coords.cpu().numpy(),
            ), "C coords of returned antigen should match those in dataset file"

            assert np.allclose(
                dataset.structure_pairs[i]["antigen"]["CB_coords"][:],
                epitope.CB_coords.cpu().numpy(),
            ), "CB coords of returned antigen should match those in dataset file"

            assert np.allclose(
                dataset.structure_pairs[i]["antigen"]["sequence"][:],
                epitope.sequence.cpu().numpy(),
            ), "Sequence of returned antigen should match those in dataset file"

            assert np.allclose(
                dataset.structure_pairs[i]["cdr"]["N_coords"][:],
                cdr.N_coords.cpu().numpy(),
            ), "N coords of returned cdr should match those in dataset file"

            assert np.allclose(
                dataset.structure_pairs[i]["cdr"]["CA_coords"][:],
                cdr.CA_coords.cpu().numpy(),
            ), "CA coords of returned cdr should match those in dataset file"

            assert np.allclose(
                dataset.structure_pairs[i]["cdr"]["C_coords"][:],
                cdr.C_coords.cpu().numpy(),
            ), "C coords of returned cdr should match those in dataset file"

            assert np.allclose(
                dataset.structure_pairs[i]["cdr"]["CB_coords"][:],
                cdr.CB_coords.cpu().numpy(),
            ), "CB coords of returned cdr should match that in dataset file"

            assert np.allclose(
                dataset.structure_pairs[i]["cdr"]["sequence"][:],
                cdr.sequence.cpu().numpy(),
            ), "Sequence of returned cdr should match those in dataset file"

    def test_train_test_split(self, dataset):
        """
        Tests the train_test_split() function, which performs a random train/test
        split of the dataset.
        """
        train_p = 0.8
        train_1, test_1 = dataset.train_test_split(train_prop=train_p, random_state=123)
        train_2, test_2 = dataset.train_test_split(train_prop=train_p, random_state=123)

        assert len(train_1) == int(
            len(dataset) * train_p
        ), "Train set should be train_p * dataset set size"

        assert (
            train_1.structure_pairs == train_2.structure_pairs
        ), "Train sets should be identical when using same random state"
        assert (
            test_1.structure_pairs == test_2.structure_pairs
        ), "Test sets should be identical when using same random state"

        train_names = [pair["name"] for pair in train_1.structure_pairs]
        test_names = [pair["name"] for pair in test_1.structure_pairs]

        assert (
            len(set(train_names).intersection(set(test_names))) == 0
        ), "Train and test sets should not have any names in common"


class TestBaseCDRDataModule:
    """
    Tests the base class for CDR data modules.
    This is actually an abstract class but we instantiate it here
    to test some of its general functionality.
    """

    def test_setup(self, datamodule):
        """
        Tests the setup() function, which is called by the pytorch lightning trainer
        before training begins.
        """
        # check if train/test/val datasets are None before setup()
        assert (
            datamodule.train_dataset is None
        ), "Train dataset should be None before setup()"
        assert (
            datamodule.test_dataset is None
        ), "Test dataset should be None before setup()"
        assert (
            datamodule.validation_dataset is None
        ), "Val dataset should be None before setup()"

        datamodule.setup("fit")

        assert (
            datamodule.train_dataset is not None
        ), "Train dataset should not be None after setup()"
        assert (
            datamodule.test_dataset is not None
        ), "Test dataset should not be None after setup()"
        assert (
            datamodule.validation_dataset is not None
        ), "Val dataset should not be None after setup()"

        assert len(datamodule.train_dataset) == int(
            datamodule._train_prop * len(datamodule.dataset)
        ), "Train dataset should be train_prop * dataset size"
        assert len(datamodule.test_dataset) == int(
            datamodule._test_prop * len(datamodule.dataset)
        ), "Test dataset should be test_prop * dataset size"
        assert len(datamodule.validation_dataset) == int(
            datamodule._val_prop * len(datamodule.dataset)
        ), "Val dataset should be val_prop * dataset size"

        train_names = [
            pair["name"] for pair in datamodule.train_dataset.structure_pairs
        ]
        test_names = [pair["name"] for pair in datamodule.test_dataset.structure_pairs]
        val_names = [
            pair["name"] for pair in datamodule.validation_dataset.structure_pairs
        ]

        assert (
            len(set(train_names).intersection(set(test_names))) == 0
        ), "Train and test sets should not have any names in common"
        assert (
            len(set(train_names).intersection(set(val_names))) == 0
        ), "Train and val sets should not have any names in common"
        assert (
            len(set(test_names).intersection(set(val_names))) == 0
        ), "Test and val sets should not have any names in common"

        train_pdb_ids = [datamodule._name_to_pdb_id_fn(name) for name in train_names]
        test_pdb_ids = [datamodule._name_to_pdb_id_fn(name) for name in test_names]
        val_pdb_ids = [datamodule._name_to_pdb_id_fn(name) for name in val_names]

        assert (
            len(set(train_pdb_ids).intersection(set(test_pdb_ids))) == 0
        ), "Train and test sets should not have any PDB IDs in common"
        assert (
            len(set(train_pdb_ids).intersection(set(val_pdb_ids))) == 0
        ), "Train and val sets should not have any PDB IDs in common"
        assert (
            len(set(test_pdb_ids).intersection(set(val_pdb_ids))) == 0
        ), "Test and val sets should not have any PDB IDs in common"

    def test_get_cdr_sampler(self, datamodule, datamodule_with_cdr_types):
        weighted_sampler = datamodule_with_cdr_types._get_cdr_sampler(
            datamodule_with_cdr_types.dataset
        )
        """ 
        Tests the _get_cdr_sampler() function, a protected
        method that returns a WeightedRandomSampler if CDR types and their 
        corresponding sampling weights are provided to the datamodule, and None otherwise.
        """

        assert isinstance(
            weighted_sampler, torch.utils.data.WeightedRandomSampler
        ), "CDR sampler should be a WeightedRandomSampler when CDR types are provided"

        sampler = datamodule._get_cdr_sampler(datamodule.dataset)

        assert (
            sampler is None
        ), "CDR sampler should be None when CDR types are not provided"
