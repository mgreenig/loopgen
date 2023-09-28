"""
Contains the base dataset class for receptor/ligand structure pairs
and the base datamodule class that organises structure pairs into
train/test/validation splits.
"""

from __future__ import annotations
from typing import (
    Any,
    List,
    Tuple,
    TypedDict,
    Sequence,
    Optional,
    Hashable,
    Union,
    Dict,
    Callable,
    Generator,
    Type,
    Set,
)
from abc import ABC, abstractmethod
import json

import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split as sklearn_split
import numpy as np
import h5py
from h5py import Group as h5pyGroup
from h5py import File as h5pyFile

from .structure import Structure, LinearStructure


class StructureDict(TypedDict):

    """
    Type for a dictionary-like object containing keys for all the relevant information needed to
    create a Structure.
    """

    sequence: np.ndarray  # sequence represented as integers 0-19 in alphabetical order of 3-letter AA codes
    N_coords: np.ndarray
    CA_coords: np.ndarray
    C_coords: np.ndarray
    CB_coords: np.ndarray


def structure_dict_to_structure(
    frag_dict: StructureDict,
    device: torch.device,
    structure_class: Type[Structure] = Structure,
) -> Union[Structure, LinearStructure]:
    """
    Creates a structure object (with the class `structure_class`) from a
    StructureDict object.
    """

    return structure_class(
        N_coords=torch.as_tensor(
            frag_dict["N_coords"][:], dtype=torch.float32, device=device
        ),
        CA_coords=torch.as_tensor(
            frag_dict["CA_coords"][:], dtype=torch.float32, device=device
        ),
        C_coords=torch.as_tensor(
            frag_dict["C_coords"][:], dtype=torch.float32, device=device
        ),
        CB_coords=torch.as_tensor(
            frag_dict["CB_coords"][:], dtype=torch.float32, device=device
        ),
        sequence=torch.as_tensor(
            frag_dict["sequence"][:], dtype=torch.int64, device=device
        ),
    )


class ReceptorLigandPair(TypedDict):
    """Type for a dictionary-like object containing a pair of examples with a name."""

    receptor: StructureDict
    ligand: StructureDict
    name: str


def hdf5_group_generator(
    group: Union[h5pyGroup, h5pyFile],
    predicate: Callable[[Union[h5pyGroup, h5pyFile]], bool],
) -> Generator:
    """
    Recursively traverses through an HDF5 structure via a
    depth-first search, evaluating a predicate at each group.
    When a group that yields a value of True is
    identified, yields the group.
    """
    if isinstance(group, h5py.Dataset):
        return

    if predicate(group) is True:
        yield group
    else:
        for key in group:
            yield from hdf5_group_generator(group[key], predicate)


class ReceptorLigandDataset(Dataset):

    """
    Class for storing a dataset of receptor/ligand structure pairs.

    The class stores a list of receptor/ligand fragment pairs,
    where each pair is a mapping-like ReceptorLigandPair object
    that contains StructureDict for the receptor and ligand,
    which contains keys for sequence (stored as integers),
    and N, CA, C, and CB coordinates.

    Most commonly, instances of this class are initialised via the `from_hdf5_file()` class method.

    Indexing and instance of the class returns a 3-tuple, the first element of
    which is the name of the complex, the second element of which is the
    receptor Structure, and the final element of which is the ligand Structure.
    """

    def __init__(
        self,
        structure_pairs: List[ReceptorLigandPair],
        device: torch.device,
        receptor_structure_class: Type[Structure] = Structure,
        ligand_structure_class: Type[Structure] = Structure,
    ):
        self._structure_pairs = structure_pairs
        self._structure_pairs_by_name = {pair["name"]: pair for pair in structure_pairs}
        self.device = device

        self.receptor_structure_class = receptor_structure_class
        self.ligand_structure_class = ligand_structure_class

    @property
    def structure_pairs(self) -> List[ReceptorLigandPair]:
        """The underlying receptor/ligand structure pairs."""
        return self._structure_pairs

    @property
    def structure_pairs_by_name(self) -> Dict[str, ReceptorLigandPair]:
        """The underlying receptor/ligand structure pairs, stored in a dictionary indexed by name."""
        return self._structure_pairs_by_name

    def __len__(self) -> int:
        return len(self._structure_pairs)

    def __contains__(self, name: str) -> bool:
        """Checks whether a structure pair with the given name exists in the dataset."""
        return name in self._structure_pairs_by_name

    def pair_to_structures(
        self, frag_pair: ReceptorLigandPair
    ) -> Tuple[str, Structure, LinearStructure]:
        """
        Converts a fragment pair into a 3-tuple, the first element of which is the name
        of the fragment pair, the second element of which is the antigen Structure, and
        the third element of which is the CDR Structure.
        """

        name = frag_pair["name"]

        receptor_dict = frag_pair["receptor"]
        receptor_structure = structure_dict_to_structure(
            receptor_dict, self.device, self.receptor_structure_class
        )

        ligand_dict = frag_pair["ligand"]
        ligand_structure = structure_dict_to_structure(
            ligand_dict, self.device, self.ligand_structure_class
        )

        return name, receptor_structure, ligand_structure

    def __getitem__(self, idx: int) -> Tuple[str, Structure, LinearStructure]:
        frag_pair = self._structure_pairs[idx]
        return self.pair_to_structures(frag_pair)

    def __repr__(self):
        return f"{self.__class__.__name__}(len={len(self)})"

    def sample(self, num: int = 1) -> List[Tuple[str, Structure, LinearStructure]]:
        """Samples some number of receptor/ligand complexes from the dataset."""
        samples = np.random.choice(len(self._structure_pairs), num)
        return [self[idx] for idx in samples]

    @classmethod
    def from_hdf5_file(
        cls,
        hdf5_file: Union[h5py.File, str],
        device: torch.device = torch.device("cpu"),
        receptor_key: str = "receptor",
        ligand_key: str = "ligand",
        receptor_structure_class: Type[Structure] = Structure,
        ligand_structure_class: Type[Structure] = Structure,
    ):
        """
        Takes in a hdf5 file containing keys for fragment pairs stored
        under keys for proteins, and simply returns a class instance initialised with all the pairs.

        First searches for the receptor key in groups in file - once a group is found
        with the receptor key, the group is checked for the ligand key. If the ligand key
        is not in the group, the group is descended until the ligand key is found.
        If ligand CDR keys are found within subgroups of the receptor group, they
        are all added to the dataset.

        :param hdf5_file: The hdf5 file containing the fragment pairs, passed either as a string filepath
            or as an h5py.File object.
        :param device: The device on which to store the structures.
        :param receptor_key: The key under which receptor data is stored. The keys "N_coords", "CA_coords",
            "C_coords", "CB_coords", and "sequence" must be present under this key. (default: "receptor")
        :param ligand_key: The key under which ligand data are stored. The keys "N_coords", "CA_coords",
            "C_coords", "CB_coords", and "sequence" must be present under this key. (default: "ligand")
        :param receptor_structure_class: The class to be used to represent a receptor structure. This can
            be any subclass of Structure, and can be changed after initialisation. (default: Structure)
        :param ligand_structure_class: The class to be used to represent a ligand structure. This can
            be any subclass of Structure, and can be changed after initialisation. (default: Structure)
        :returns: Dataset with receptor/ligand pairs read from the file.
        """

        if isinstance(hdf5_file, str):
            hdf5_file = h5py.File(hdf5_file)

        structure_pairs = []
        group_has_receptor = lambda grp: receptor_key in grp
        group_has_ligand = lambda grp: ligand_key in grp

        # loop through groups with the receptor key
        for receptor_group in hdf5_group_generator(hdf5_file, group_has_receptor):
            # if ligand in the receptor group, make a receptor/ligand pair
            if ligand_key in receptor_group:
                pair_dict = ReceptorLigandPair(
                    name=receptor_group.name,
                    receptor=receptor_group[receptor_key],
                    ligand=receptor_group[ligand_key],
                )
                structure_pairs.append(pair_dict)
            # otherwise look for the CDR key in the lower levels of the group
            else:
                for ligand_group in hdf5_group_generator(
                    receptor_group, group_has_ligand
                ):
                    pair_dict = ReceptorLigandPair(
                        name=ligand_group.name,
                        receptor=receptor_group[receptor_key],
                        ligand=ligand_group[ligand_key],
                    )
                    structure_pairs.append(pair_dict)

        return cls(
            structure_pairs, device, receptor_structure_class, ligand_structure_class
        )

    def subset_by_name(self, names: Set[str]):
        """
        Returns a new dataset with the subset of structure pairs whose names appear
        in the input set of names.

        :param names: A set of names of structure pairs to keep. These are checked with the string
            stored under the key "name" in each ReceptorLigandPair object in structure_pairs.
        :returns: A new ReceptorLigandDataset object with the subset of structure pairs.
        """

        name_set = set(names)
        kept_pairs = []
        for pair in self._structure_pairs:
            if pair["name"] in name_set:
                kept_pairs.append(pair)

        return self.__class__(kept_pairs, self.device)

    def train_test_split(
        self,
        train_prop: float = 0.8,
        random_state: int = 123,
        by: Optional[Sequence[Hashable]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[ReceptorLigandDataset, ReceptorLigandDataset]:
        """
        Performs a train/test split with a proportion
        `train_prop` samples kept in the training set, and
        the remainder allocated to the test set. Users can
        specify the `by` argument to provide a sequence
        of labels - the same length as `self.fragment_pairs` -
        which should be used to determine the train/test split,
        so that no label appears in both the test and the train set.
        If `by` is specified, `train_prop` will refer to the proportion
        of **unique** labels allocated to the train set.
        """

        if train_prop == 1.0:
            train_frags = self._structure_pairs
            test_frags = []
        elif train_prop == 0.0:
            train_frags = []
            test_frags = self._structure_pairs
        # split by labels if provided
        elif by is not None:
            if len(by) != len(self):
                raise ValueError(
                    f"Length of by ({len(by)}) does not match length of dataset {len(self)}."
                )

            labels = np.unique(by)
            train_labels, test_labels = sklearn_split(
                labels, train_size=train_prop, random_state=random_state
            )
            train_label_set = set(train_labels)
            test_label_set = set(test_labels)

            train_frags = []
            test_frags = []
            for lab, frag_pair in zip(by, self._structure_pairs):
                if lab in train_label_set:
                    train_frags.append(frag_pair)
                elif lab in test_label_set:
                    test_frags.append(frag_pair)

        # otherwise just split the list of fragment pairs
        else:
            train_frags, test_frags = sklearn_split(
                self._structure_pairs,
                train_size=train_prop,
                random_state=random_state,
            )

        # make class instances for both the train and test datasets
        train_dataset = self.__class__(train_frags, device=self.device, *args, **kwargs)
        test_dataset = self.__class__(test_frags, device=self.device, *args, **kwargs)

        return train_dataset, test_dataset


def load_splits_file(
    filepath: str, dataset: ReceptorLigandDataset
) -> Dict[str, Set[str]]:
    """
    Validates a JSON file containing a dictionary of train/test/validation splits against a dataset,
    ensuring that all the required keys are provided, that names do not appear in multiple sets, and
    that all names provided in the file are present in the dataset. Raises an error if these conditions are not met.

    Returns a dictionary of the splits (taken from the file) if all conditions are met,
    with the lists of names converted into sets.

    :param filepath: Filepath to a JSON file containing a dictionary of train/test/validation splits.
        The file must have the key "train", and optionally keys "test" and "validation" for the test/validation sets.
    :param dataset: The dataset against which to validate the splits.
    :returns: A dictionary of the splits (sets of names) if all conditions are met.
    """
    split_dict = json.load(open(filepath, "r"))

    if "train" not in split_dict:
        raise KeyError("JSON file must contain a key 'train' for the training set.")

    if not isinstance(split_dict["train"], list):
        raise TypeError(
            "The item stored under the key 'train' must be a list of names."
        )

    train_names = set(split_dict["train"])
    for name in train_names:
        if name not in dataset:
            raise ValueError(
                f"Name {name} in training set not found in dataset {dataset}."
            )

    if "test" in split_dict:
        test_names = set(split_dict["test"])
        if not isinstance(split_dict["test"], list):
            raise TypeError(
                "The item stored under the key 'test' must be a list of names."
            )
        for name in split_dict["test"]:
            if name not in dataset:
                raise ValueError(
                    f"Name {name} in test set not found in dataset {dataset}."
                )
            if name in train_names:
                raise ValueError(
                    f"Name {name} appears in both the training and test sets."
                )
    else:
        test_names = set()

    if "validation" in split_dict:
        validation_names = set(split_dict["validation"])
        if not isinstance(split_dict["validation"], list):
            raise TypeError(
                "The item stored under the key 'validation' must be a list of names."
            )
        for name in split_dict["validation"]:
            if name not in dataset:
                raise ValueError(
                    f"Name {name} in test set not found in dataset {dataset}."
                )
            if name in train_names:
                raise ValueError(
                    f"Name {name} appears in both the training and validation sets."
                )
            if name in test_names:
                raise ValueError(
                    f"Name {name} appears in both the test and validation sets."
                )
    else:
        validation_names = set()

    split_dict = {
        "train": train_names,
        "test": test_names,
        "validation": validation_names,
    }
    return split_dict


def load_generated_structures(
    hdf5_filepath: str, device: torch.device("cpu")
) -> List[Tuple[str, Structure, LinearStructure, Tuple[LinearStructure, ...]]]:
    """
    Loads generated structures from an HDF5 file. The HDF5 file should be the output
    of a generation run (i.e. running loopgen <model> generate [args]),
    and should contain the following keys stored under every structure pair
    group:
        - receptor
        - ligand
        - generated_<i> (for however many generated structures were generated)
    """

    file = h5py.File(hdf5_filepath)
    all_structures = []
    for group in hdf5_group_generator(
        file, lambda g: "receptor" in g and "ligand" in g
    ):
        receptor = group["receptor"]
        receptor_structure = structure_dict_to_structure(receptor, device)

        ligand = group["ligand"]
        ligand_structure = structure_dict_to_structure(ligand, device, LinearStructure)

        generated = tuple(group[key] for key in group if key.startswith("generated_"))
        generated_structures = tuple(
            structure_dict_to_structure(g, device, LinearStructure) for g in generated
        )

        all_structures.append(
            (group.name, receptor_structure, ligand_structure, generated_structures)
        )

    return all_structures
