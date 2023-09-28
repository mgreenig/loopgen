from typing import OrderedDict, Optional, Tuple, List

import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
import torch
import seaborn as sns

from sklearn.metrics import confusion_matrix

from .structure import Structure, LinearStructure, AminoAcid3
from .utils import combine_coords, get_covalent_bonds


def _extend_contig(covalent_bond_matrix: torch.Tensor, indices: list):
    """
    Extends a list of residue indices if any residues are covalently bonded to the
    residue indexed by the final element of `indices`.
    """
    next_residue_bonds = covalent_bond_matrix[:, indices[-1]] == 1.0
    if torch.any(next_residue_bonds):
        next_index = torch.nonzero(next_residue_bonds).squeeze(-1)[0].item()
        indices.append(next_index)
        return _extend_contig(covalent_bond_matrix, indices)
    else:
        return indices


def get_contiguous_regions(
    covalent_bond_matrix: torch.Tensor,
) -> OrderedDict[int, list[int]]:
    """
    Gets an integer-indexed dictionary of lists of indices corresponding
    to covalently-bonded atoms, obtained from an input binary covalent bond matrix.
    The output dictionary has the form {region_num: region_residue_indices}.
    """
    contiguous_regions = collections.OrderedDict()
    N_bonded_indices, C_bonded_indices = torch.nonzero(
        covalent_bond_matrix, as_tuple=True
    )
    current_N_index, current_C_index = (
        N_bonded_indices[0].item(),
        C_bonded_indices[0].item(),
    )
    region_num = 0
    while (
        current_N_index < covalent_bond_matrix.shape[-2] - 1
        and current_C_index < covalent_bond_matrix.shape[-1]
    ):
        contig = _extend_contig(
            covalent_bond_matrix, indices=[current_C_index, current_N_index]
        )
        contiguous_regions[region_num] = contig
        region_num += 1
        current_N_index = N_bonded_indices[N_bonded_indices > contig[-1]]
        current_C_index = C_bonded_indices[C_bonded_indices > contig[-1]]
        if len(current_N_index) > 0 and len(current_C_index) > 0:
            current_N_index = current_N_index[0].item()
            current_C_index = current_C_index[0].item()
        else:
            break

    return contiguous_regions


def plot_structure(
    structure: Structure,
    figure: Optional[go.Figure] = None,
    backbone_colour: str = "black",
    sidechain_colour: str = "yellow",
):
    """Generates a 3-D plot of a Structure object."""
    if isinstance(structure, LinearStructure):
        contiguous_regions = {0: list(range(len(structure)))}
    else:
        covalent_bond_matrix = get_covalent_bonds(
            structure.N_coords.unsqueeze(-2), structure.C_coords.unsqueeze(-3)
        )
        contiguous_regions = get_contiguous_regions(covalent_bond_matrix)

    residue_nums_by_contig = {}
    all_contig_nums = []
    for contig, residues in contiguous_regions.items():
        all_contig_nums.extend([contig] * len(residues))
        for res_num in residues:
            residue_nums_by_contig[res_num] = contig

    all_coords = pd.DataFrame(
        combine_coords(
            structure.N_coords,
            structure.CA_coords,
            structure.C_coords,
            structure.CB_coords,
        )
        .cpu()
        .detach()
        .numpy(),
        columns=["x", "y", "z"],
    )

    all_coords["contig"] = sorted(
        np.repeat(all_contig_nums, 4), key=residue_nums_by_contig.get
    )

    backbone_coords = pd.DataFrame(
        combine_coords(structure.N_coords, structure.CA_coords, structure.C_coords)
        .cpu()
        .detach()
        .numpy(),
        columns=["x", "y", "z"],
    )

    backbone_coords["contig"] = sorted(
        np.repeat(all_contig_nums, 3), key=residue_nums_by_contig.get
    )

    sidechain_coords = pd.DataFrame(
        combine_coords(structure.CA_coords, structure.CB_coords).cpu().detach().numpy(),
        columns=["x", "y", "z"],
    )
    sidechain_coords["contig"] = sorted(
        np.repeat(all_contig_nums, 2), key=residue_nums_by_contig.get
    )

    atom_scatter_plots = [
        go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            marker=dict(color=backbone_colour),
            mode="markers",
            name=frag_type,
        )
        for frag_type, df in all_coords.groupby("contig")
    ]

    backbone_line_plots = [
        go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            line=dict(color=backbone_colour),
            mode="lines",
            showlegend=False,
        )
        for frag_type, df in backbone_coords.groupby("contig")
    ]

    sidechain_line_plots = [
        go.Scatter3d(
            x=sidechain_coords.iloc[(i - 1) : (i + 1)]["x"],
            y=sidechain_coords.iloc[(i - 1) : (i + 1)]["y"],
            z=sidechain_coords.iloc[(i - 1) : (i + 1)]["z"],
            line=dict(color=sidechain_colour),
            mode="lines",
            showlegend=False,
        )
        for i in range(1, len(sidechain_coords), 2)
    ]

    if figure is not None:
        figure.add_traces(
            data=atom_scatter_plots + backbone_line_plots + sidechain_line_plots
        )
    else:
        figure = go.Figure(
            data=atom_scatter_plots + backbone_line_plots + sidechain_line_plots
        )
        figure.update_layout(height=600, width=1000, autosize=False)

    return figure


def create_color_map(
    minval: float,
    maxval: float,
    colors=["0.8", "0"],
    masked_vals_color=None,
    return_sm=False,
    set_under=None,
    set_over=None,
) -> matplotlib.colors.LinearSegmentedColormap:
    """Creates a colorblind-friendly color map for plotting."""
    mymap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mycolors", colors, N=256
    )
    if set_under is None:
        set_under = colors[0]
    if set_over is None:
        set_over = colors[-1]

    mymap.set_under(set_under)
    mymap.set_over(set_over)

    if masked_vals_color is not None:
        mymap.set_bad(masked_vals_color)

    if return_sm:
        sm = plt.cm.ScalarMappable(
            cmap=mymap, norm=plt.Normalize(vmin=minval, vmax=maxval)
        )
        sm._A = []
        return mymap, sm

    return mymap


def plot_color_map_simple(colormap, npoints: int = 256) -> None:
    if isinstance(colormap, str):
        colormap = matplotlib.cm.get_cmap(colormap)

    f = plt.figure(figsize=(2, 8))
    ax = f.gca()
    ax.scatter(
        [0] * npoints,
        np.linspace(0, 1, npoints),
        color=colormap(np.linspace(0, 1, npoints)),
    )
    ax.set_xticklabels([])
    plt.show(block=False)


readable_map_for_chars = create_color_map(
    1, 0, colors=["white", "Gold", "SpringGreen", "DeepSkyBlue"]
)
readable_map_for_chars_centered = create_color_map(
    1, 0, colors=["Gold", "Moccasin", "white", "SpringGreen", "DeepSkyBlue"]
)
readable_map_for_chars_vmin = create_color_map(
    0,
    1,
    colors=[
        "AntiqueWhite",
        "Bisque",
        "Gold",
        "GoldenRod",
        "GreenYellow",
        "SpringGreen",
        "LightSkyBlue",
        "DeepSkyBlue",
    ],
    set_under="white",
    set_over="blue",
)


def aa_conf_mat_heatmap(
    true_labs: pd.Series,
    pred_labs: pd.Series,
    normalize: str = "pred",
    cmap: Optional[str] = None,
    annot: bool = True,
    threshold: float = 0.10,
    figsize: Tuple[int, int] = (22, 16),
    xlab: str = "Predicted residues",
    ylab: str = "True residues",
):
    """
    Plots a confusion matrix heatmap for a set of true labels/predicted amino acids.
    Expects amino acids encoded as integers 0-19, sorted in alphabetical order of 3 letter codes.
    Used to visualise the performance of sequence prediction models.
    """

    labs = np.arange(20)

    conf_mat = confusion_matrix(true_labs, pred_labs, labels=labs, normalize=normalize)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 5], width_ratios=[1.5, 4, 0.25])
    dummy1_ax = fig.add_subplot(gs[0, 0])
    bar1_ax = fig.add_subplot(gs[0, 1])
    dummy2_ax = fig.add_subplot(gs[0, 2])
    bar2_ax = fig.add_subplot(gs[1, 0])
    hm_ax = fig.add_subplot(gs[1, 1], sharex=bar1_ax, sharey=bar2_ax)
    cmap_ax = fig.add_subplot(gs[1, 2])

    if normalize == "true":
        cbar_title = "Row\nproportion\n"
    elif normalize == "pred":
        cbar_title = "Column\nproportion\n"
    else:
        cbar_title = ""

    cmap_ax.set_title(cbar_title, fontsize=18)
    cmap_ax.tick_params(labelsize=16)

    dummy1_ax.axis("off")
    dummy2_ax.axis("off")

    if normalize is not None:
        vmin = 0
        vmax = 1
    else:
        vmin = None
        vmax = None

    if cmap is None:
        cmap = readable_map_for_chars

    sns.heatmap(
        conf_mat,
        xticklabels=AminoAcid3.__members__,
        yticklabels=AminoAcid3.__members__,
        ax=hm_ax,
        cmap=cmap,
        annot=annot,
        annot_kws={"fontsize": 16},
        cbar_ax=cmap_ax,
        linewidths=2,
        vmin=vmin,
        vmax=vmax,
    )
    hm_ax.tick_params(axis="both", labelrotation=0, left=True)

    hm_ax.xaxis.tick_top()
    hm_ax.xaxis.set_label_position("top")
    for t in hm_ax.texts:
        if float(t.get_text()) >= threshold:
            t.set_text(t.get_text())
        else:
            t.set_text("")

    hm_ax.tick_params(labelsize=14)
    bar1_ax.tick_params(labelsize=14)
    bar2_ax.tick_params(labelsize=14)

    true_lab_counts = true_labs.value_counts(normalize=True).sort_index()
    true_lab_counts = true_lab_counts.reindex(labs, fill_value=0.0)
    y_tick_pos = [i + 0.5 for i in range(20)]
    bar2_ax.barh(
        y=y_tick_pos, width=true_lab_counts.values, align="center", edgecolor="black"
    )
    bar2_ax.set_xlim((0, max(true_lab_counts) + 0.05))
    bar2_ax.invert_xaxis()
    bar2_ax.set_xlabel("\nFrequency", fontsize=20)
    bar2_ax.set_ylabel(f"{ylab}\n", fontsize=20)
    bar2_ax.tick_params(axis="y", left=True)

    pred_lab_counts = pred_labs.value_counts(normalize=True).sort_index()
    pred_lab_counts = pred_lab_counts.reindex(labs, fill_value=0.0)
    x_tick_pos = [i + 0.5 for i in range(20)]
    bar1_ax.bar(
        x=x_tick_pos, height=pred_lab_counts.values, align="center", edgecolor="black"
    )
    bar1_ax.xaxis.tick_top()
    bar1_ax.xaxis.set_label_position("top")
    bar1_ax.set_ylabel("Frequency\n", fontsize=20)
    bar1_ax.set_xlabel(f"{xlab}\n", fontsize=20)

    return fig


def ramachandran_plot(
    structures: List[LinearStructure], ax: Optional[plt.Axes] = None, **kwargs
):
    """
    Makes a ramachandran plot using a list of structures. If `ax` is not specified,
    a new figure is created. Additional keyword arguments are passed to `plt.plot`.

    :param structures: A list of structures to plot.
    :param ax: The matplotlib axes on which to plot.
    :param kwargs: Additional keyword arguments to pass to `plt.plot`.
    :return: The matplotlib axes on which the plot was made.
    """

    phi_arr = []
    omega_arr = []
    psi_arr = []

    for structure in structures:
        angles = structure.get_backbone_dihedrals()
        phi = angles[1:, 0]
        omega = angles[:-1, 1]
        psi = angles[:-1, 2]

        phi_arr.extend(phi.numpy().tolist())
        omega_arr.extend(omega.numpy().tolist())
        psi_arr.extend(psi.numpy().tolist())

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.plot(phi_arr, psi_arr, ".", **kwargs)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)

    ax.set_aspect("equal")

    ax.axhline(color="black", linewidth=0.8)  # Add horizontal axis
    ax.axvline(color="black", linewidth=0.8)  # Add vertical axis

    ax.grid(True, linestyle="--", alpha=0.2)  # Add grid lines
    return ax
