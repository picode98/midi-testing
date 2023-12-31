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
    display_name: Optional[str] = None

    def __init__(self, magnitude: float):
        self.magnitude = magnitude
        self.parameters_updated = False

    def get_parameters(self):
        return [StageParamInfo('magnitude', 'float', self.magnitude, 0.0, 1.0)]
    
    def get_parameter(self, name):
        if name in [param.name for param in self.get_parameters()]:
            return getattr(self, name)
        else:
            raise AttributeError(name)
        
    def set_parameter(self, name, value):
        if name in [param.name for param in self.get_parameters()]:
            return setattr(self, name, value)
        else:
            raise AttributeError(name)

    def __call__(self, input, num_frames: int, sample_rate: int):
        raise NotImplementedError()
    
    def on_control_change(self, instrument: midi.Input, event: ControlChangeMessage):
        pass