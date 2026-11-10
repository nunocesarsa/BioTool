"""Model registry. To add a model: write an adapter (see base.py) and register it here."""

from .birdcode import BirdCODE
from .birdnet import BirdNET
from .dfs import DeepForestSound
from .indices import AcousticIndices
from .perch import PerchV2
from .yamnet import YAMNet

DETECTORS = {cls.key: cls for cls in (BirdNET, PerchV2, YAMNet, DeepForestSound, BirdCODE)}
INDICES = AcousticIndices

ALL_KEYS = list(DETECTORS) + [AcousticIndices.key]


def build(key: str, **kwargs):
    if key == AcousticIndices.key:
        return AcousticIndices()
    return DETECTORS[key](**kwargs)
