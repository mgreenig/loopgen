"""
Generates new structures for some input epitopes using a LoopGen model.

If input is a PDB file, treats each model as a separate epitope and generates CDR loop structures for each.
These generated structures are saved as different models in an output PDB file.

If input is an HDF5 file, generates CDR loop structures for each epitope in the file (see README.md for 
file format details). The generated structures are then stored in an output HDF5 file with the same
format as the input file, with keys of the form "generated_{i}" (i labels which generated structure is present)
added to each structure pair group (at the same level as the keys "receptor" and "ligand").

Generated structures are saved as different models in an output PDB file.
"""

from typing import List, Tuple, Union, Type
import logging as lg
import os
import argparse
from pathlib import Path

import torch
import numpy as np
import h5py
import pytorch_lightning as pl
import pandas as pd
from Bio.PDB import PDBParser

from . import setup_model
from .settings import ModelSettings
from .datamodule import CDRFrameDataModule, CDRCoordinateDataModule
from .model import CDRFrameDiffusionModel, CDRCoordinateDiffusionModel
from .utils import permute_cdrs, permute_epitopes, translate_cdrs_away
from .metrics import (
    get_rmsd,
    get_clash_loss,
    get_epitope_cdr_clashes,
    get_bond_length_loss,
    get_bond_angle_loss,
    mean_pairwise_rmsd,
    pca,
)
from .types import CDRFramesBatch
from ..structure import Structure, LinearStructure
from ..data import (
    ReceptorLigandDataset,
    ReceptorLigandPair,
    StructureDict,
    load_splits_file,
)
from ..utils import get_device


def add_generate_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds command line arguments for generation to a parser.
    """

    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the HDF5 data file or a PDB file containing epitope data to be used for generation.",
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
        help="Path to a JSON file containing the names of instances stored under "
        "the keys 'train', 'validation', and 'test'. Only used if an HDF5 file is given as a data set. "
        "If provided, structures will only be generated for the test set. "
        "If not provided, structures will be generated "
        "for all instances in the input HDF5 file.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a PyTorch checkpoint file containing model weights.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of structures to generate for each input epitope.",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=10,
        help="Length of generated CDRs (only used if input file is a PDB file).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.2,
        help="Scale of the noise used in the reverse process. Higher values "
        "generate more diverse samples at the cost of lower quality. "
        "We have found that only values <0.5 generate valid structures.",
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Directory in which results (predictions) will be saved.",
    )
    parser.add_argument(
        "--permute_epitopes",
        action="store_true",
        help="Whether to permute epitopes during the generation. This randomly swaps epitopes between "
        "complexes, aligning the principal components of the swapped epitope to the original epitope.",
    )
    parser.add_argument(
        "--scramble_epitopes",
        action="store_true",
        help="Whether to scramble each epitope's sequence, randomly swapping residue identities.",
    )
    parser.add_argument(
        "--translate_cdrs",
        action="store_true",
        help="Whether to translate each CDR 20 angstroms away from the epitope.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "gpu"],
        help="Which device to use (defaults to None, in which case the "
        "GPU is used if available, and if not the CPU is used).",
    )


def _gen_structure_evaluation(
    all_instances: List[Tuple[str, Structure, LinearStructure]],
    gen_structs: List[Tuple[LinearStructure]],
    gen_structs_epitopes_permuted: List[Tuple[LinearStructure]],
    gen_structs_cdrs_translated: List[Tuple[LinearStructure]],
    gen_structs_epitopes_scrambled: List[Tuple[LinearStructure]],
) -> pd.DataFrame:
    """Evaluates generated structures."""
    all_instances_cdr_permuted = permute_cdrs(all_instances)

    mean_rmsds = []
    rmsd_stds = []
    mean_perm_cdr_rmsds = []

    first_pc_similarities = []
    first_pc_sim_stds = []
    perm_first_pc_similarities = []

    pca_loading_similarities = []
    pca_loading_sim_stds = []
    perm_pca_loading_similarities = []

    true_min_distance_to_epitope = []
    true_center_distance_to_epitope = []
    min_distances_to_epitope = []
    center_distances_to_epitope = []

    total_gen_structs = []
    bond_angle_viol_count = []
    bond_length_viol_count = []
    clash_viol_count = []
    epitope_clash_viol_count = []
    any_viol_count = []

    WT_mean_pairwise_rmsds = []
    epitope_permuted_mean_pairwise_rmsds = []
    epitope_scrambled_mean_pairwise_rmsds = []
    cdr_translated_mean_pairwise_rmsds = []

    for i in range(len(all_instances)):
        name, epitope, cdr = all_instances[i]

        gen_cdrs = gen_structs[i]
        epitope = epitope.to(gen_cdrs[0].CA_coords.device)
        cdr = cdr.to(gen_cdrs[0].CA_coords.device)

        total_gen_structs.append(len(gen_cdrs))

        _, _, rand_cdr = all_instances_cdr_permuted[i]
        rand_cdr = rand_cdr.to(gen_cdrs[0].CA_coords.device)

        # comparisons to ground truth
        mean_rmsd = np.mean(
            [get_rmsd(cdr.CA_coords, g.CA_coords).item() for g in gen_cdrs]
        )
        mean_rmsds.append(mean_rmsd)

        rmsd_std = np.std(
            [get_rmsd(cdr.CA_coords, g.CA_coords).item() for g in gen_cdrs]
        )
        rmsd_stds.append(rmsd_std)

        mean_perm_cdr_rmsd = np.mean(
            [get_rmsd(rand_cdr.CA_coords, g.CA_coords).item() for g in gen_cdrs]
        )
        mean_perm_cdr_rmsds.append(mean_perm_cdr_rmsd)

        cdr_CA_pcs, pca_loadings = pca(cdr.CA_coords)
        gen_CA = torch.stack([g.CA_coords for g in gen_cdrs])
        gen_cdr_CA_pcs, gen_pca_loadings = pca(gen_CA)
        perm_cdr_CA_pcs, perm_pca_loadings = pca(rand_cdr.CA_coords)

        cdr_first_pc = cdr_CA_pcs[..., 0].unsqueeze(0).expand(len(gen_cdrs), -1)
        perm_first_pc = perm_cdr_CA_pcs[..., 0].unsqueeze(0).expand(len(gen_cdrs), -1)
        gen_cdr_first_pcs = gen_cdr_CA_pcs[..., 0]

        first_pc_sim = torch.abs(
            torch.nn.functional.cosine_similarity(
                cdr_first_pc, gen_cdr_first_pcs, dim=-1
            )
        )
        perm_first_pc_sim = torch.abs(
            torch.nn.functional.cosine_similarity(
                perm_first_pc, gen_cdr_first_pcs, dim=-1
            )
        )

        first_pc_similarities.append(first_pc_sim.mean().item())
        first_pc_sim_stds.append(first_pc_sim.std().item())
        perm_first_pc_similarities.append(perm_first_pc_sim.mean().item())

        cdr_pc_loadings = torch.abs(
            pca_loadings[..., 0].unsqueeze(0).expand(len(gen_cdrs), -1)
        )
        perm_pc_loadings = torch.abs(
            perm_pca_loadings[..., 0].unsqueeze(0).expand(len(gen_cdrs), -1)
        )

        pc_loading_sim = torch.nn.functional.cosine_similarity(
            cdr_pc_loadings, gen_pca_loadings, dim=-1
        )
        perm_pc_loading_sim = torch.nn.functional.cosine_similarity(
            perm_pc_loadings, gen_pca_loadings, dim=-1
        )

        pca_loading_similarities.append(pc_loading_sim.mean().item())
        pca_loading_sim_stds.append(pc_loading_sim.std().item())
        perm_pca_loading_similarities.append(perm_pc_loading_sim.mean().item())

        # Structural violation metrics
        bond_angle_viols = 0
        bond_length_viols = 0
        clash_viols = 0
        epitope_clash_viols = 0
        any_viols = 0

        for g in gen_cdrs:
            bond_angle_viols += torch.any(get_bond_angle_loss(g) > 0).long().item()
            bond_length_viols += torch.any(get_bond_length_loss(g) > 0).long().item()
            clash_viols += torch.any(get_clash_loss(g) > 0).long().item()
            epitope_clash_viols += (
                torch.any(get_epitope_cdr_clashes(g, epitope) > 0).long().item()
            )
            any_viols += (
                torch.any(
                    (get_bond_angle_loss(g) > 0)
                    | (get_bond_length_loss(g) > 0)
                    | (get_clash_loss(g) > 0)
                    | (get_epitope_cdr_clashes(g, epitope) > 0)
                )
                .long()
                .item()
            )

        bond_angle_viol_count.append(bond_angle_viols)
        bond_length_viol_count.append(bond_length_viols)
        clash_viol_count.append(clash_viols)
        epitope_clash_viol_count.append(epitope_clash_viols)
        any_viol_count.append(any_viols)

        # comparisons to between transformation conditions
        WT_CA_coords = [g.CA_coords for g in gen_cdrs]
        epitope_permuted_CA_coords = [
            g.CA_coords for g in gen_structs_epitopes_permuted[i]
        ]
        epitope_scrambled_CA_coords = [
            g.CA_coords for g in gen_structs_epitopes_scrambled[i]
        ]
        cdr_translated_CA_coords = [g.CA_coords for g in gen_structs_cdrs_translated[i]]

        WT_mean_pairwise_rmsd = mean_pairwise_rmsd(WT_CA_coords, WT_CA_coords).item()
        epitope_permuted_mean_pairwise_rmsd = mean_pairwise_rmsd(
            WT_CA_coords, epitope_permuted_CA_coords
        ).item()
        epitope_scrambled_mean_pairwise_rmsd = mean_pairwise_rmsd(
            WT_CA_coords, epitope_scrambled_CA_coords
        ).item()
        cdr_translated_mean_pairwise_rmsd = mean_pairwise_rmsd(
            WT_CA_coords, cdr_translated_CA_coords
        ).item()

        WT_mean_pairwise_rmsds.append(WT_mean_pairwise_rmsd)
        epitope_permuted_mean_pairwise_rmsds.append(epitope_permuted_mean_pairwise_rmsd)
        epitope_scrambled_mean_pairwise_rmsds.append(
            epitope_scrambled_mean_pairwise_rmsd
        )
        cdr_translated_mean_pairwise_rmsds.append(cdr_translated_mean_pairwise_rmsd)

    df = pd.DataFrame(
        {
            "name": [name for name, _, _ in all_instances],
            "mean_rmsd": mean_rmsds,
            "rmsd_std": rmsd_stds,
            "mean_perm_cdr_rmsd": mean_perm_cdr_rmsds,
            "first_pc_sim": first_pc_similarities,
            "first_pc_sim_std": first_pc_sim_stds,
            "perm_first_pc_sim": perm_first_pc_similarities,
            "pca_loading_sim": pca_loading_similarities,
            "pca_loading_sim_std": pca_loading_sim_stds,
            "perm_pca_loading_sim": perm_pca_loading_similarities,
            "bond_angle_viol_count": bond_angle_viol_count,
            "bond_length_viol_count": bond_length_viol_count,
            "clash_viol_count": clash_viol_count,
            "epitope_clash_viol_count": epitope_clash_viol_count,
            "any_viol_count": any_viol_count,
            "total_gen_structures": total_gen_structs,
            "WT_mean_pairwise_rmsd": WT_mean_pairwise_rmsds,
            "epitope_permuted_mean_pairwise_rmsd": epitope_permuted_mean_pairwise_rmsds,
            "epitope_scrambled_mean_pairwise_rmsd": epitope_scrambled_mean_pairwise_rmsds,
            "cdr_translated_mean_pairwise_rmsd": cdr_translated_mean_pairwise_rmsds,
        }
    )

    return df


def generate(
    datamodule: Union[CDRFrameDataModule, CDRCoordinateDataModule],
    model: Union[CDRFrameDiffusionModel, CDRCoordinateDiffusionModel],
    n: int,
    use_epitope_permutation: bool,
    use_epitope_scrambling: bool,
    use_cdr_translation: bool,
    noise_scale: float,
    seed: int,
) -> Union[List[Tuple[str, Structure, LinearStructure, Tuple[LinearStructure, ...]]]]:
    """
    Generates samples from a model and runs an evaluation on them (if run_evaluation is True)

    :param datamodule: The datamodule for collating data.
    :param model: The model
    :param n: Number of CDR loops to generate for each epitope.
    :param use_epitope_permutation: Whether to permute epitopes in the dataset, swapping each CDR's
        epitope with a random one and aligning the new epitope to the old epitope's principal components.
    :param use_epitope_scrambling: Whether to scramble the epitope's sequence.
    :param use_cdr_translation: Whether to translate each CDR away from the epitope.
    :param noise_scale: Scale of the noise used in the reverse process. Higher values
        generate more diverse samples at the cost of lower quality.
    :param seed: Random seed.
    :returns: A list of tuples containing the name of the complex, the epitope, the ground truth CDR,
        and a tuple of generated CDR structures.
    """

    def gen_batch(batch: CDRFramesBatch) -> List[Tuple[LinearStructure, ...]]:
        """Generation for a single batch, returning a list of generated CDR structures for each batch element."""

        with torch.no_grad():
            output = model.generate(batch, noise_scale=noise_scale, n=n)
        output_structures = [
            LinearStructure.from_frames(f).detach().to(torch.device("cpu"))
            for f in output[-1]
        ]
        output_structures_split = [s.split() for s in output_structures]

        gen_cdr_structures = list(map(tuple, zip(*output_structures_split)))
        return gen_cdr_structures

    pl.seed_everything(123, workers=True)

    datamodule.setup("generate")

    all_instances = [
        datamodule.test_dataset[i] for i in range(len(datamodule.test_dataset))
    ]

    if use_epitope_permutation:
        all_instances = permute_epitopes(all_instances)

    if use_epitope_scrambling:
        all_instances = [
            (name, ep.scramble_sequence(), cdr) for name, ep, cdr in all_instances
        ]

    if use_cdr_translation:
        all_instances = translate_cdrs_away(all_instances)

    outputs = []
    for i in range(0, len(datamodule.test_dataset), datamodule.batch_size):
        start = i
        end = min(i + datamodule.batch_size, len(datamodule.test_dataset))
        batch_instances = all_instances[start:end]

        batch = datamodule.collate(batch_instances)
        names, epitope, _ = batch

        epitopes = epitope.split()
        cdrs = [cdr.center(return_centre=False) for _, _, cdr in batch_instances]

        gen_cdr_structures = gen_batch(batch)

        for name, ep, cdr, pred_cdrs in zip(names, epitopes, cdrs, gen_cdr_structures):
            outputs.append(
                (
                    name,
                    ep,
                    cdr,
                    pred_cdrs,
                )
            )

    return outputs


def generate_from_args(
    args: argparse.Namespace, model_class: Type[pl.LightningModule]
) -> None:
    """Runs `generate()` using command line arguments."""
    device, accelerator = get_device(args.device)

    settings = ModelSettings.from_yaml(args.config)

    is_pdb = args.data_path.endswith(".pdb")

    if is_pdb and (
        args.permute_epitopes or args.scramble_epitopes or args.translate_cdrs
    ):
        lg.warning(
            "Epitope permutation/scrambling and CDR translating is not available for PDB input, "
            "switch to HDF5 format instead."
        )

    lg.basicConfig(format="%(asctime)s %(levelname)-8s: %(message)s")

    if not os.path.exists(args.out_dir):
        lg.info(
            f"Specified output directory {args.out_dir} does not exist, creating..."
        )
        os.mkdir(args.out_dir)

    if is_pdb:
        pdb_id = Path(args.data_path).stem
        parser = PDBParser()
        structure = parser.get_structure(pdb_id, args.data_path)

        structure_pairs = []
        for i, model in enumerate(structure.get_models(), start=1):
            residues = list(model.get_residues())

            epitope_structure = Structure.from_pdb_residues(residues)
            epitope_dict = StructureDict(
                N_coords=epitope_structure.N_coords.cpu().numpy(),
                CA_coords=epitope_structure.CA_coords.cpu().numpy(),
                C_coords=epitope_structure.C_coords.cpu().numpy(),
                CB_coords=epitope_structure.CB_coords.cpu().numpy(),
                sequence=epitope_structure.sequence.cpu().numpy(),
            )

            cdr_coords = np.zeros((args.length, 3))
            cdr_sequence = np.zeros((args.length,), dtype=int)
            cdr_dict = StructureDict(
                N_coords=cdr_coords,
                CA_coords=cdr_coords,
                C_coords=cdr_coords,
                CB_coords=cdr_coords,
                sequence=cdr_sequence,
            )

            structure_pair = ReceptorLigandPair(
                name=f"{pdb_id}_{i}", receptor=epitope_dict, ligand=cdr_dict
            )

            structure_pairs.append(structure_pair)

        dataset = ReceptorLigandDataset(structure_pairs, device=device)
        splits = None
        hdf5_file = None
        out_hdf5_file = None

    else:
        hdf5_file = h5py.File(args.data_path)
        dataset = ReceptorLigandDataset.from_hdf5_file(args.data_path, device=device)
        splits = load_splits_file(args.splits, dataset)

        base_filepath = f"{Path(args.data_path).stem}_generated"

        if args.permute_epitopes:
            base_filepath = f"{base_filepath}_permuted"

        if args.scramble_epitopes:
            base_filepath = f"{base_filepath}_scrambled"

        if args.translate_cdrs:
            base_filepath = f"{base_filepath}_translated"

        out_hdf5_filepath = os.path.join(args.out_dir, f"{base_filepath}.hdf5")
        i = 1
        while os.path.exists(out_hdf5_filepath):
            out_hdf5_filepath = f"{base_filepath}_v{i}.hdf5"
            i += 1
        out_hdf5_file = h5py.File(out_hdf5_filepath, "w")

    param_dict = settings.distribute_model_params(model_class)

    datamodule, model = setup_model(
        dataset, splits, param_dict, model_class, args.checkpoint
    )

    model = model.to(device)
    model.eval()

    outputs = generate(
        datamodule,
        model,
        args.n,
        args.permute_epitopes,
        args.scramble_epitopes,
        args.translate_cdrs,
        args.noise_scale,
        args.seed,
    )

    for name, epitope, cdr, pred_cdrs in outputs:
        if is_pdb:
            pred_cdr_batch = LinearStructure.combine(pred_cdrs)
            pred_cdr_batch.write_to_pdb(
                os.path.join(
                    args.out_dir, f"{name.lstrip('/').replace('/', '_')}_generated.pdb"
                )
            )
        else:
            group = out_hdf5_file.create_group(name)
            epitope.write_to_hdf5(group.create_group("receptor"))
            cdr.write_to_hdf5(group.create_group("ligand"))
            for i, pred_cdr in enumerate(pred_cdrs, start=1):
                pred_cdr_group = out_hdf5_file[name].create_group(f"generated_{i}")
                pred_cdr.write_to_hdf5(pred_cdr_group)

    if not is_pdb:
        out_hdf5_file.close()
        hdf5_file.close()
