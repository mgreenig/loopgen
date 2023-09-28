"""
Types used across many or all model.
"""

from typing import Any, Dict, DefaultDict, Union, Tuple, Optional, Sequence

import torch

from ..structure import Structure, OrientationFrames
from ..graph import (
    ComplexData,
    StructureData,
    VectorFeatureComplexData,
    VectorFeatureStructureData,
    ScalarFeatureComplexData,
    ScalarFeatureStructureData,
)

# all the possible types of CDR graph (either in complex or as a solo structure)
ProteinGraph = Union[ComplexData, StructureData]
VectorFeatureGraph = Union[VectorFeatureComplexData, VectorFeatureStructureData]
ScalarFeatureGraph = Union[ScalarFeatureComplexData, ScalarFeatureStructureData]

# dictionary for storing parameters for different classes
ParamDictionary = DefaultDict[str, Dict[str, Any]]

# The output of the CDRFrameDataModule collate() (used for training), consisting of:
# 1. tuple of IDs, one associated with each CDR in the batch
# 2. epitope structure
# 3. ground truth CDR orientation frames
CDRFramesBatch = Tuple[Tuple[str], Structure, OrientationFrames]

Score = Union[torch.Tensor, Sequence[torch.Tensor]]

# The output of a forward process, consisting of
# 1. tensor(s) of scores
# 2. the noised graph sampled from q(x_t | x_0)
# 3. an optional self-conditioning noised graph sampled from q(x_{t+1} | x_t)
ForwardProcessOutput = Tuple[
    Score,
    VectorFeatureGraph,
    Optional[VectorFeatureGraph],
]
