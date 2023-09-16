""" 
Tests the StructureData and ComplexData classes (and derivatives),
which represent a single structure and a complex of two structures, respectively,
as graphs. These classes inherit from the PyTorch Geometric Data and HeteroData classes
respectively.
"""

import pytest
import os
import torch

from loopgen import Structure, LinearStructure, ReceptorLigandDataset
from loopgen.graph import (
    StructureData,
    ScalarFeatureStructureData,
    VectorFeatureStructureData,
    ScalarFeatureComplexData,
    VectorFeatureComplexData,
)


@pytest.fixture
def datapoints() -> list[tuple[tuple[str], Structure, LinearStructure]]:
    """Returns a list of datapoints from the dataset."""
    dataset = ReceptorLigandDataset.from_hdf5_file(
        os.path.join(os.path.dirname(__file__), "data/cdrs_20.hdf5")
    )
    return [dataset[i] for i in range(len(dataset))]


class TestStructureData:
    """Tests the StructureData class, which represents a single structure."""

    def test_from_structure(
        self, datapoints: list[tuple[tuple[str], Structure, LinearStructure]]
    ):
        """Test if a StructureData object can be created from the CDRs in the dataset."""
        for _, _, cdr in datapoints:
            scalar_feature_graph = ScalarFeatureStructureData.from_structure(cdr)
            vector_feature_graph = VectorFeatureStructureData.from_structure(cdr)

            assert (
                scalar_feature_graph.num_nodes
                == vector_feature_graph.num_nodes
                == cdr.num_residues
            ), "Number of nodes does not match number of CDR residues"

            assert hasattr(
                vector_feature_graph, "vector_x"
            ), "Vector node features not found for CDR graph"
            assert hasattr(
                vector_feature_graph, "vector_edge_attr"
            ), "Vector node features not found for CDR graph"

        cdr_batch = Structure.combine([cdr for _, _, cdr in datapoints])
        scalar_feature_graph_batch = ScalarFeatureStructureData.from_structure(
            cdr_batch
        )
        vector_feature_graph_batch = VectorFeatureStructureData.from_structure(
            cdr_batch
        )

        # check if batch information has transferred correctly
        assert torch.allclose(
            scalar_feature_graph_batch.batch, cdr_batch.batch
        ), "Batch assignments does not match their values in the CDR Structure"
        assert torch.allclose(
            vector_feature_graph_batch.batch, cdr_batch.batch
        ), "Batch assignments does not match their values in the CDR Structure"

        assert torch.allclose(
            scalar_feature_graph_batch.ptr, cdr_batch.ptr
        ), "Ptr tensor does not match that of the CDR Structure"
        assert torch.allclose(
            scalar_feature_graph_batch.ptr, cdr_batch.ptr
        ), "Ptr tensor does not match that of the CDR Structure"

    def test_update_structure(
        self, datapoints: list[tuple[tuple[str], Structure, LinearStructure]]
    ):
        """
        Tests the update_structure() method, which updates the features of the
        StructureData object using a new structure.
        """
        _, _, update_cdr = datapoints[-1]
        update_cdr_scalar_feature_graph = ScalarFeatureStructureData.from_structure(
            update_cdr
        )
        update_cdr_vector_feature_graph = VectorFeatureStructureData.from_structure(
            update_cdr
        )

        for _, _, cdr in datapoints[:-1]:
            scalar_feature_graph = ScalarFeatureStructureData.from_structure(cdr)
            scalar_feature_graph.update_structure(update_cdr)

            vector_feature_graph = VectorFeatureStructureData.from_structure(cdr)
            vector_feature_graph.update_structure(update_cdr)

            assert (
                scalar_feature_graph.num_nodes
                == vector_feature_graph.num_nodes
                == update_cdr.num_residues
            ), "Number of nodes does not match number of CDR residues"

            assert torch.allclose(
                scalar_feature_graph.x, update_cdr_scalar_feature_graph.x
            ), "Scalar node features do not match target values after updating"

            assert torch.allclose(
                scalar_feature_graph.edge_index,
                update_cdr_scalar_feature_graph.edge_index,
            ), "Edge indices not match target values after updating"

            assert torch.allclose(
                scalar_feature_graph.edge_attr,
                update_cdr_scalar_feature_graph.edge_attr,
            ), "Edge features do not match target values after updating"

            assert torch.allclose(
                scalar_feature_graph.orientations,
                update_cdr_scalar_feature_graph.orientations,
            ), "Orientations do not match target values after updating"

            assert torch.allclose(
                scalar_feature_graph.pos, update_cdr_scalar_feature_graph.pos
            ), "Positions do not match target values after updating"

            assert torch.allclose(
                scalar_feature_graph.sequence, update_cdr_scalar_feature_graph.sequence
            ), "Sequences do not match target values after updating"

            assert torch.allclose(
                vector_feature_graph.vector_x, update_cdr_vector_feature_graph.vector_x
            ), "Vector node features do not match target values after updating"

            assert torch.allclose(
                vector_feature_graph.vector_edge_attr,
                update_cdr_vector_feature_graph.vector_edge_attr,
            ), "Vector edge features do not match target values after updating"


class TestComplexData:
    """Tests the ComplexData classes, which represents a receptor/ligand complex."""

    def test_from_structures(
        self, datapoints: list[tuple[tuple[str], Structure, LinearStructure]]
    ):
        """Test if a ComplexData object can be created from the epitopes/CDRs in the dataset."""
        for _, epitope, cdr in datapoints:
            scalar_feature_graph = ScalarFeatureComplexData.from_structures(
                epitope, cdr
            )
            vector_feature_graph = VectorFeatureComplexData.from_structures(
                epitope, cdr
            )

            assert (
                scalar_feature_graph["ligand"].num_nodes
                == vector_feature_graph["ligand"].num_nodes
                == cdr.num_residues
            ), "Number of nodes does not match number of CDR residues"

            assert (
                scalar_feature_graph["receptor"].num_nodes
                == vector_feature_graph["receptor"].num_nodes
                == epitope.num_residues
            ), "Number of nodes does not match number of epitope residues"

            assert (
                len(vector_feature_graph.edge_types) == 4
            ), "Should be 4 edge types for each possible pairing of receptor/ligand"

            assert hasattr(
                vector_feature_graph["ligand"], "vector_x"
            ), "Vector node features not found for CDR graph"
            assert hasattr(
                vector_feature_graph["ligand", "ligand"], "vector_edge_attr"
            ), "Vector node features not found for CDR graph"

            assert hasattr(
                vector_feature_graph["receptor"], "vector_x"
            ), "Vector node features not found for epitope graph"
            assert hasattr(
                vector_feature_graph["receptor", "receptor"], "vector_edge_attr"
            ), "Vector node features not found for epitope graph"

        cdr_batch = LinearStructure.combine([cdr for _, _, cdr in datapoints])
        epitope_batch = LinearStructure.combine(
            [epitope for _, epitope, _ in datapoints]
        )

        scalar_feature_graph_batch = ScalarFeatureComplexData.from_structures(
            epitope_batch, cdr_batch
        )
        vector_feature_graph_batch = VectorFeatureComplexData.from_structures(
            epitope_batch, cdr_batch
        )

        # check if batch information has transferred correctly
        assert torch.allclose(
            scalar_feature_graph_batch["ligand"].batch, cdr_batch.batch
        ), "Batch assignments does not match their values in the CDR Structure"
        assert torch.allclose(
            vector_feature_graph_batch["ligand"].batch, cdr_batch.batch
        ), "Batch assignments does not match their values in the CDR Structure"
        assert torch.allclose(
            scalar_feature_graph_batch["receptor"].batch, epitope_batch.batch
        ), "Batch assignments does not match their values in the epitope Structure"
        assert torch.allclose(
            vector_feature_graph_batch["receptor"].batch, epitope_batch.batch
        ), "Batch assignments does not match their values in the epitope Structure"

        assert torch.allclose(
            scalar_feature_graph_batch["ligand"].ptr, cdr_batch.ptr
        ), "Ptr tensor does not match that of the CDR Structure"
        assert torch.allclose(
            scalar_feature_graph_batch["ligand"].ptr, cdr_batch.ptr
        ), "Ptr tensor does not match that of the CDR Structure"
        assert torch.allclose(
            scalar_feature_graph_batch["receptor"].ptr, epitope_batch.ptr
        ), "Ptr tensor does not match that of the epitope Structure"
        assert torch.allclose(
            vector_feature_graph_batch["receptor"].ptr, epitope_batch.ptr
        ), "Ptr tensor does not match that of the epitope Structure"

    def test_update_structure(
        self, datapoints: list[tuple[tuple[str], Structure, LinearStructure]]
    ):
        """
        Tests the update_structure() method, which updates the graph features
        after replacing one of the structures - either the receptor or ligand -
        with a new one.
        """
        _, update_epitope, update_cdr = datapoints[-1]
        update_cdr_scalar_feature_graph = ScalarFeatureComplexData.from_structures(
            update_epitope, update_cdr
        )
        update_cdr_vector_feature_graph = VectorFeatureComplexData.from_structures(
            update_epitope, update_cdr
        )
        update_conformer_graph = VectorFeatureComplexData.from_structures(
            update_epitope, update_cdr.orientation_frames
        )

        for _, epitope, cdr in datapoints[:-1]:
            scalar_feature_graph = ScalarFeatureComplexData.from_structures(
                epitope, cdr
            )
            # update the CDR in the scalar feature graph
            scalar_feature_graph.update_structure(update_cdr, key="ligand")

            assert torch.allclose(
                update_cdr_scalar_feature_graph["ligand", "ligand"].edge_index,
                scalar_feature_graph["ligand", "ligand"].edge_index,
            ), "Edge index not updated after updating CDR structure"

            assert torch.allclose(
                update_cdr_scalar_feature_graph["ligand"].x,
                scalar_feature_graph["ligand"].x,
            ), "Scalar node features not updated after updating CDR structure"

            # update the epitope in the scalar feature graph
            scalar_feature_graph.update_structure(update_epitope, key="receptor")

            assert torch.allclose(
                update_cdr_scalar_feature_graph["ligand", "receptor"].edge_index,
                scalar_feature_graph["ligand", "receptor"].edge_index,
            ), "Edge index not updated after updating epitope structure"

            assert torch.allclose(
                update_cdr_scalar_feature_graph["receptor"].x,
                scalar_feature_graph["receptor"].x,
            ), "Scalar node features not updated after updating epitope structure"

            vector_feature_graph = VectorFeatureComplexData.from_structures(
                epitope, cdr
            )

            # update the CDR in the vector feature graph
            vector_feature_graph.update_structure(update_cdr, key="ligand")

            assert torch.allclose(
                update_cdr_vector_feature_graph["ligand", "ligand"].edge_index,
                vector_feature_graph["ligand", "ligand"].edge_index,
            ), "Edge index not updated after updating CDR structure"

            assert torch.allclose(
                update_cdr_vector_feature_graph["ligand"].x,
                vector_feature_graph["ligand"].x,
            ), "Scalar node features not updated after updating CDR structure"

            assert torch.allclose(
                update_cdr_vector_feature_graph["ligand"].vector_x,
                vector_feature_graph["ligand"].vector_x,
            ), "Vector node features not updated after updating CDR structure"

            # update the epitope in the vector feature graph
            vector_feature_graph.update_structure(update_epitope, key="receptor")

            assert torch.allclose(
                update_cdr_vector_feature_graph["ligand", "receptor"].edge_index,
                vector_feature_graph["ligand", "receptor"].edge_index,
            ), "Edge index not updated after updating epitope structure"

            assert torch.allclose(
                update_cdr_vector_feature_graph["ligand", "receptor"].edge_attr,
                vector_feature_graph["ligand", "receptor"].edge_attr,
            ), "Edge features not updated after updating epitope structure"

            assert torch.allclose(
                update_cdr_vector_feature_graph["ligand", "receptor"].vector_edge_attr,
                vector_feature_graph["ligand", "receptor"].vector_edge_attr,
            ), "Vector edge features not updated after updating epitope structure"

            assert torch.allclose(
                update_cdr_vector_feature_graph["receptor"].x,
                vector_feature_graph["receptor"].x,
            ), "Scalar node features not updated after updating epitope structure"

            assert torch.allclose(
                update_cdr_vector_feature_graph["receptor"].vector_x,
                vector_feature_graph["receptor"].vector_x,
            ), "Vector node features not updated after updating epitope structure"

            conformer_graph = VectorFeatureComplexData.from_structures(
                epitope, cdr.orientation_frames
            )

            # update the CDR in the conformer graph
            conformer_graph.update_structure(
                update_cdr.orientation_frames, key="ligand"
            )

            assert torch.allclose(
                update_conformer_graph["ligand", "ligand"].edge_index,
                conformer_graph["ligand", "ligand"].edge_index,
            ), "Edge index not updated after updating CDR structure"

            assert torch.allclose(
                update_conformer_graph["ligand"].x,
                conformer_graph["ligand"].x,
            ), "Scalar node features not updated after updating CDR structure"

            assert torch.allclose(
                update_conformer_graph["ligand"].vector_x,
                conformer_graph["ligand"].vector_x,
            ), "Vector node features not updated after updating CDR structure"

            # update the epitope in the conformer graph
            conformer_graph.update_structure(update_epitope, key="receptor")

            assert torch.allclose(
                update_conformer_graph["ligand", "receptor"].edge_index,
                conformer_graph["ligand", "receptor"].edge_index,
            ), "Edge index not updated after updating epitope structure"

            assert torch.allclose(
                update_conformer_graph["ligand", "receptor"].edge_attr,
                conformer_graph["ligand", "receptor"].edge_attr,
            ), "Edge features not updated after updating epitope structure"

            assert torch.allclose(
                update_conformer_graph["ligand", "receptor"].vector_edge_attr,
                conformer_graph["ligand", "receptor"].vector_edge_attr,
            ), "Vector edge features not updated after updating epitope structure"

            assert torch.allclose(
                update_conformer_graph["receptor"].x,
                conformer_graph["receptor"].x,
            ), "Scalar node features not updated after updating epitope structure"

            assert torch.allclose(
                update_conformer_graph["receptor"].vector_x,
                conformer_graph["receptor"].vector_x,
            ), "Vector node features not updated after updating epitope structure"

    def test_to_homogeneous(
        self, datapoints: list[tuple[tuple[str], Structure, LinearStructure]]
    ):
        """
        Tests the to_homogeneous() method, which converts a ComplexData object - a heterogeneous graph
        of a receptor/ligand complex - to a StructureData object.
        """
        for _, epitope, cdr in datapoints:
            scalar_feature_graph = ScalarFeatureComplexData.from_structures(
                epitope, cdr
            )
            vector_feature_graph = VectorFeatureComplexData.from_structures(
                epitope, cdr
            )
            homogeneous_scalar_feature_graph = scalar_feature_graph.to_homogeneous()
            homogeneous_vector_feature_graph = vector_feature_graph.to_homogeneous()

            for orig_graph, hom_graph in zip(
                [scalar_feature_graph, vector_feature_graph],
                [homogeneous_scalar_feature_graph, homogeneous_vector_feature_graph],
            ):
                assert isinstance(
                    hom_graph, StructureData
                ), "Homogeneous graph is not of type StructureData"

                assert (
                    hom_graph.num_nodes == orig_graph.num_nodes
                ), "Number of nodes in homogeneous graph does not match that of the original graph"

                assert torch.allclose(
                    hom_graph.x[hom_graph.node_type == 0],
                    orig_graph["receptor"].x,
                ), "Receptor features not saved correctly in homogeneous graph"

                assert torch.allclose(
                    hom_graph.x[hom_graph.node_type == 1],
                    orig_graph["ligand"].x,
                ), "Ligand features not saved correctly in homogeneous graph"

                assert torch.allclose(
                    hom_graph.edge_attr[hom_graph.edge_type == 0],
                    orig_graph["receptor", "receptor"].edge_attr,
                ), "Receptor-receptor edge features not saved correctly in homogeneous graph"

                assert torch.allclose(
                    hom_graph.edge_attr[hom_graph.edge_type == 1],
                    orig_graph["receptor", "ligand"].edge_attr,
                ), "Receptor-ligand edge features not saved correctly in homogeneous graph"

                assert torch.allclose(
                    hom_graph.edge_attr[hom_graph.edge_type == 2],
                    orig_graph["ligand", "receptor"].edge_attr,
                ), "Ligand-receptor edge features not saved correctly in homogeneous graph"

                assert torch.allclose(
                    hom_graph.edge_attr[hom_graph.edge_type == 3],
                    orig_graph["ligand", "ligand"].edge_attr,
                ), "Ligand-ligand edge features not saved correctly in homogeneous graph"
