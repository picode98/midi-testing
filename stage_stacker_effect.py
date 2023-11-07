from abc import ABC
from typing import Any, Optional
import dataclasses

from pygame import midi


from midi_utils import ControlChangeMessage

@dataclasses.dataclass
class StageParamInfo:
    name: str
    dtype: str
    initial_value: Any
    min: Optional[float] = None
    max: Optional[float] = None

class SampleStage(ABC):
    def __init__(self, magnitude: float):
        self.magnitude = magnitude
        self.parameters_updated = False

    def get_parameters(self):
        return [StageParamInfo('magnitude', 'float', self.magnitude, 0.0, 1.0)]

    def __call__(self, input, num_frames: int, sample_rate: int):
        raise NotImplementedError()
    
    def on_control_change(self, instrument: midi.Input, event: ControlChangeMessage):
        pass