import dataclasses
from abc import ABC
from enum import StrEnum
import pathlib
from typing import Type, Optional, Dict, List, Tuple

import numpy as np
import dataclasses_json

@dataclasses.dataclass
class EffectSettings:
    setting_name: str
    data_type: Type
    range_min: Optional[float] = None
    range_max: Optional[float] = None

class SampleEffect(ABC):
    def __init__(self):
        self.sensitivity = 1.0

    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        raise NotImplementedError()

    def get_settings(self) -> Dict[str, EffectSettings]:
        return {'sensitivity': EffectSettings('Sensitivity', float, 0.0, None)}
    
@dataclasses.dataclass
class LoopedRegion(dataclasses_json.DataClassJsonMixin):
    start: Optional[float]
    end: Optional[float]
    loop_duration: Optional[float]
    sub_loops: List['LoopedRegion']

    def deep_copy(self):
        return dataclasses.replace(self, sub_loops=[loop.deep_copy() for loop in self.sub_loops])

@dataclasses.dataclass
class SnapshotExternalWAVFile(dataclasses_json.DataClassJsonMixin):
    path: str
    values: Optional[np.ndarray] = dataclasses.field(default=None, metadata=dataclasses_json.config(exclude=lambda _: True))

    def save(self, project_folder: pathlib.Path):
        assert self.values is not None
        from utils import write_wav_file

        try:
            with (project_folder / self.path).open('xb') as created_data_file:
                write_wav_file(created_data_file, self.values)
        except FileExistsError:
            pass

    def load(self, project_folder: pathlib.Path):
        from utils import read_wav_file
        self.values, _ = read_wav_file(str(project_folder / self.path))

    def deep_copy(self):
        return dataclasses.replace(self, values=(self.values.copy() if self.values is not None else None))

@dataclasses.dataclass
class Modifier(dataclasses_json.DataClassJsonMixin):
    value_buffer: Optional[SnapshotExternalWAVFile] = None

    def apply(self, sample: np.ndarray, sample_rate: int):
        raise NotImplementedError()

    def save(self, project_folder: pathlib.Path):
        if self.value_buffer is not None:
            self.value_buffer.save(project_folder)

    def load(self, project_folder: pathlib.Path):
        if self.value_buffer is not None:
            self.value_buffer.load(project_folder)

    def deep_copy(self):
        return dataclasses.replace(self, value_buffer=(self.value_buffer.deep_copy() if self.value_buffer is not None else None))

    @classmethod
    def from_dict_polymorphic(cls, values: Dict, class_map: Dict[str, Type['Modifier']]) -> 'Modifier':
        return class_map[values['type_id']].from_dict(values)

    def to_dict_polymorphic(self) -> Dict:
        return {**self.to_dict(), 'type_id': type(self).__name__}

class FrequencyDomainModifier(Modifier):
    pass

class TimeDomainModifier(Modifier):
    pass

@dataclasses.dataclass
class SampleEditorSnapshot(dataclasses_json.DataClassJsonMixin):
    sample_file: SnapshotExternalWAVFile
    num_wavelengths: int
    loop_tree: LoopedRegion
    description: str
    fd_modifiers: List[FrequencyDomainModifier]
    td_modifiers: List[TimeDomainModifier]

    def save(self, project_folder: pathlib.Path):
        self.sample_file.save(project_folder)
        for modifier in self.fd_modifiers:
            modifier.save(project_folder)
        for modifier in self.td_modifiers:
            modifier.save(project_folder)

    def load(self, project_folder: pathlib.Path):
        self.sample_file.load(project_folder)
        for modifier in self.fd_modifiers:
            modifier.load(project_folder)
        for modifier in self.td_modifiers:
            modifier.load(project_folder)

    @classmethod
    def from_dict_polymorphic(cls, values: Dict, fd_modifier_class_map: Dict[str, Type[FrequencyDomainModifier]], td_modifier_class_map: Dict[str, Type[TimeDomainModifier]]) -> 'SampleEditorSnapshot':
        result = cls.from_dict({**values, 'fd_modifiers': [], 'td_modifiers': []})
        result.fd_modifiers = [FrequencyDomainModifier.from_dict_polymorphic(mod_data, fd_modifier_class_map) for mod_data in values['fd_modifiers']]
        result.td_modifiers = [TimeDomainModifier.from_dict_polymorphic(mod_data, td_modifier_class_map) for mod_data in values['td_modifiers']]

        return result

    def to_dict_polymorphic(self):    
            return {**self.to_dict(), 'fd_modifiers': [modifier.to_dict_polymorphic() for modifier in self.fd_modifiers], 'td_modifiers': [modifier.to_dict_polymorphic() for modifier in self.td_modifiers]}

# @dataclasses.dataclass
# class SampleData:
#     sample_values: np.ndarray
#     num_wavelengths: int
#     loop_region_tree: LoopedRegion
#     modifiers: List[Modifier]