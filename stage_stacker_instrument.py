from abc import ABC
from math import pi
from copy import deepcopy
import dataclasses
import time
from typing import Any, Dict, List, Type, Tuple, Callable

import numpy as np
from pygame import midi

from midi_utils import ControlChangeMessage, CustomSynth, KeyOffMessage, KeyOnMessage, get_piano_key_frequency
from utils import CustomOsc, Fadeable, linear_wave
from stage_stacker_effect import SampleStage, StageParamInfo
from stage_stacker_ui import StageStackerWebviewUI

class PeriodicGeneratorStage(SampleStage):
    def __init__(self):
        super().__init__(1.0)
        self.phase = 0
        self.osc_time = 0

    def periodic_waveform(self, time_vector, phase_vector):
        raise NotImplementedError()

    def __call__(self, input: Tuple[np.ndarray, np.ndarray], num_frames: int, sample_rate: int) -> Any:
        time_vector = np.linspace(self.osc_time, self.osc_time + num_frames / sample_rate, num_frames)
        freq_vec, amp_vec = input

        phase_vec = 2 * pi * np.cumsum(freq_vec) / sample_rate + self.phase
        phase_vec = phase_vec % (2 * pi)
        frame_array = amp_vec * self.periodic_waveform(time_vector, phase_vec)

        self.osc_time += num_frames / sample_rate
        self.phase = phase_vec[-1]

        return frame_array

class SineStage(PeriodicGeneratorStage):
    # def __init__(self):
    #     self.phase = 0.0

    def periodic_waveform(self, time_vector, phase_vector):
        return np.sin(phase_vector)
    # def __call__(self, input: Tuple[np.ndarray, np.ndarray], num_frames: int, sample_rate: int) -> Any:
    #     phases = 2.0 * pi * np.cumsum(np.pad(input[0], pad_width=(1, 0), constant_values=0)) / sample_rate + self.phase
    #     result = np.sin(phases[:-1]) * input[1]
    #     self.phase = phases[-1] % (2.0 * pi)
    #     return result

class SawtoothStage(PeriodicGeneratorStage):
    def periodic_waveform(self, time_vector, phase_vector):
        return ((phase_vector + pi) % (2.0 * pi)) / pi - 1.0
    
class ExpressionGeneratorStage(PeriodicGeneratorStage):
    def __init__(self, expression: str):
        super().__init__()
        self.expression = expression

    def get_parameters(self):
        return super().get_parameters() + [StageParamInfo('expression', 'string', self.expression)]

    @property
    def expression(self):
        return self._expression
    
    @expression.setter
    def expression(self, new_expression: str):
        self._expression = new_expression
        self._compiled_expr = compile(self._expression, '<effect expression>', mode='eval')

    def periodic_waveform(self, time_vector, phase_vector):
        return self.magnitude * eval(self._compiled_expr, {'x': phase_vector / (2.0 * pi), 't': time_vector, 'sin': lambda x: np.sin(x * 2.0 * pi),
                                                           'cos': lambda x: np.cos(x * 2.0 * pi), 'exp': np.exp,
                                                           'abs': np.absolute,
                                                           'lin': lambda x: linear_wave(x * 2.0 * pi)})

class VibratoStage(SampleStage):
    def __init__(self, magnitude: float):
        super().__init__(magnitude)
        self.frequency = 10.0
        self.phase = 0.0

    def get_parameters(self):
        return super().get_parameters() + [StageParamInfo('frequency', 'float', self.frequency, 0.0, 20.0)]

    def __call__(self, input: List[Tuple[np.ndarray, np.ndarray]], num_frames: int, sample_rate: int) -> Any:
        delta = 2.0 * pi * num_frames * self.frequency / sample_rate
        freq_deltas = self.magnitude * 0.04 * np.sin(np.linspace(self.phase, self.phase + delta, num_frames, endpoint=False)) + 1.0
        self.phase = (self.phase + delta) % (2.0 * pi)
        return [(freq_values * freq_deltas, amp_values) for freq_values, amp_values in input]


# class MixerStage(SampleStage):
#     def __init__(self, components: List[SampleStage]):
#         self.components = components

#     def __call__(self, input: List, num_frames: int, sample_rate: int) -> Any:
#         assert len(input) == len(self.components)
#         return sum(stage(elem, num_frames, sample_rate) for elem, stage in zip(input, self.components))

class MixerStage(SampleStage):
    def __init__(self, new_component_factory: Callable[[], SampleStage]):
        super().__init__(1.0)
        self.components = []
        self.new_component_factory = new_component_factory
        self.component_params = dict()
        self.gains = []

    def get_parameters(self):
        return super().get_parameters() + [StageParamInfo(f'gain_{i + 1}', 'float', (self.gains[i] if i < len(self.gains) else 1.0), 0.0, 1.0) for i, component in enumerate(self.components)] \
            + ([dataclasses.replace(param, name='component_' + param.name) for param in self.components[0].get_parameters()] if len(self.components) > 0 else [])
    
    def get_parameter(self, name: str) -> Any:
        if name.startswith('gain_'):
            gain_num = int(name.removeprefix('gain_'))
            if len(self.gains) < gain_num:
                return 1.0
            else:
                return self.gains[gain_num - 1]
        elif name.startswith('component_'):
            return self.component_params[name.removeprefix('component_')]
        else:
            return super().get_parameter(name)

    def set_parameter(self, name: str, value: Any) -> None:
        if name.startswith('gain_'):
            gain_num = int(name.removeprefix('gain_'))
            if len(self.gains) < gain_num:
                self.gains += [1.0] * (gain_num - len(self.gains))
            self.gains[gain_num - 1] = value
        elif name.startswith('component_'):
            self.component_params[name.removeprefix('component_')] = value
            for component in self.components:
                component.set_parameter(name.removeprefix('component_'), value)
        else:
            super().set_parameter(name, value)

    def __call__(self, input: List, num_frames: int, sample_rate: int):
        if len(input) < len(self.components):
            self.components = self.components[:len(input)]
            self.parameters_updated = True
        elif len(input) > len(self.components):
            new_components = [self.new_component_factory() for _ in range(len(input) - len(self.components))]
            for component in new_components:
                for name, value in self.component_params.items():
                    component.set_parameter(name, value)
            self.components += new_components
            self.parameters_updated = True

        assert len(input) == len(self.components)
        return sum((self.gains[i] if i < len(self.gains) else 1.0) * stage(elem, num_frames, sample_rate) for i, (elem, stage) in enumerate(zip(input, self.components)))
    

class HarmonicsStage(SampleStage):
    def __call__(self, input: List[Tuple[np.ndarray, np.ndarray]], num_frames: int, sample_rate: int):
        result = []
        for component_freq, component_amp in input:
            result += [(component_freq, component_amp), (component_freq / 2.0, component_amp * self.magnitude / 2.0),
                       (component_freq / 4.0, component_amp * self.magnitude / 3.0), (component_freq / 8.0, component_amp * self.magnitude / 6.0),
                       (component_freq * 2.0, component_amp * self.magnitude / 10.0)]
            
        return result

class BassHarmonicsStage(SampleStage):
    def __call__(self, input: List[Tuple[np.ndarray, np.ndarray]], num_frames: int, sample_rate: int):
        result = []
        for component_freq, component_amp in input:
            result += [(component_freq, component_amp), (component_freq / 8.0, component_amp * self.magnitude),
                       (component_freq / 16.0, component_amp * self.magnitude / 1.5), (component_freq / 32.0, component_amp * self.magnitude / 3.0),
                       (component_freq / 64.0, component_amp * self.magnitude / 6.0)]
            
        return result
    
class SimpleDistortEffect(SampleStage):
    def __call__(self, input: np.ndarray, num_frames: int, sample_rate: int):
        max_threshold = np.max(np.absolute(input))
        clipped = np.clip(input, -max_threshold * (1.0 - self.magnitude / 2.0), max_threshold * (1.0 - self.magnitude / 2.0))
        return clipped / (1.0 - self.magnitude / 2.0)
    
class AMEffect(SampleStage):
    def __init__(self, magnitude: float):
        super().__init__(magnitude)
        self.phase = 0.0

    def __call__(self, input: np.ndarray, num_frames: int, sample_rate: int):
        delta = 2.0 * pi * num_frames * (self.magnitude * 20.0) / sample_rate
        amplitudes = np.cos(np.linspace(self.phase, self.phase + delta, num_frames, endpoint=False))
        self.phase = (self.phase + delta) % (2.0 * pi)
        return input * amplitudes
    
class FrequencyDomainWahWahEffect(SampleStage):
    def __init__(self, magnitude: float):
        super().__init__(magnitude)
        self.phase = 0.0
        self.dist_spread, self.dist_center_min, self.dist_center_max = 1.0, 1.0, 12.0

    def get_parameters(self):
        return super().get_parameters() + [StageParamInfo('dist_spread', 'float', self.dist_spread, 0.5, 5.0),
                                           StageParamInfo('dist_center_min', 'float', self.dist_center_min, 1.0, 12.0),
                                           StageParamInfo('dist_center_max', 'float', self.dist_center_max, 1.0, 12.0)]

    def __call__(self, input: List[Tuple[np.ndarray, np.ndarray]], num_frames: int, sample_rate: int):
        delta = 2.0 * pi * num_frames * (self.magnitude * 20.0) / sample_rate
        dist_centers = self.dist_center_min + (self.dist_center_max - self.dist_center_min) \
            * (np.sin(np.linspace(self.phase, self.phase + delta, num_frames, endpoint=False)) + 1.0) / 2.0
        log_freqs = np.stack([np.log2(freqs) for freqs, _ in input], axis=1)
        # center_freqs, freq_stddevs = np.average(log_freqs, axis=1), np.std(log_freqs, axis=1)
        normalized_log_freqs = (log_freqs - np.reshape(dist_centers, (num_frames, 1))) / self.dist_spread
        amp_multipliers = 1.0 / (normalized_log_freqs ** 2 + 1.0)
        # amp_multipliers /= np.reshape(np.sum(amp_multipliers, axis=1), (num_frames, 1))
        amp_mat = np.stack([amps for _, amps in input], axis=1)
        scaled_amp_mat = amp_mat * amp_multipliers
        result_amps = scaled_amp_mat * np.reshape(np.sum(amp_mat, axis=1) / np.sum(scaled_amp_mat, axis=1), (num_frames, 1))
        self.phase = (self.phase + delta) % (2.0 * pi)
        return [(freqs, result_amps[:, i]) for i, (freqs, _) in enumerate(input)]

@Fadeable
class StageStackOsc(CustomOsc):
    def __init__(self, frequency: float, amplitude: float, freq_domain_stages: List[SampleStage],
                 time_domain_generator: SampleStage, time_domain_stages: List[SampleStage]):
        super().__init__()
        self.freq_domain_stages = freq_domain_stages
        self.time_domain_generator = time_domain_generator
        self.time_domain_stages = time_domain_stages
        self.frequency = frequency
        self.amplitude = amplitude

    def play_frames(self, num_frames: int):
        curr_result = [(np.full((num_frames,), self.frequency), np.full((num_frames,), self.amplitude))]

        for stage in self.freq_domain_stages + [self.time_domain_generator] + self.time_domain_stages:
            curr_result = stage(curr_result, num_frames, self.sample_rate)

        assert isinstance(curr_result, np.ndarray)
        return curr_result
    
class StageStackerSynth(CustomSynth):
    def __init__(self, freq_domain_key_map: Dict[int, Type[SampleStage]], time_domain_key_map: Dict[int, Type[SampleStage]]):
        super().__init__()
        self.freq_domain_key_map = freq_domain_key_map
        self.time_domain_key_map = time_domain_key_map

        self.template_osc = StageStackOsc(0.0, 0.0, [], MixerStage(lambda: ExpressionGeneratorStage(expression='sin(x)')), [])
        # self.freq_domain_stages = []
        # self.time_domain_generator = MixerStage(lambda: SineStage()) # MixerStage([SineStage() for _ in range(4)])
        # self.time_domain_stages = []

        self.ui = StageStackerWebviewUI()
        self.ui.add_effect(self.template_osc.time_domain_generator, 'generator')

        for effect in freq_domain_key_map.values():
            self.ui.register_effect(effect, 'freq-domain')
        for effect in time_domain_key_map.values():
            self.ui.register_effect(effect, 'time-domain')
        # self.generator_stack = [HarmonicsStage(), VibratoStage(0.015), MixerStage([SineStage() for _ in range(4)])]

    # def _iter_current_and_templates(self):
    #     for osc in self.iter_oscs():
    #         yield from osc.stages

    #     yield from self.freq_domain_stages + self.time_domain_stages

    def update_output(self):
        ui_events = list(self.ui.get_events())

        for osc in [self.template_osc] + list(self.iter_oscs()):
            location_map = {'freq-domain': osc.freq_domain_stages, 'generator': [osc.time_domain_generator], 'time-domain': osc.time_domain_stages}
            for ui_event in ui_events:
                if ui_event[0] == 'reorder':
                    reorder_list = location_map[ui_event[1]]
                    reorder_list.insert(ui_event[3], reorder_list.pop(ui_event[2]))
                elif ui_event[0] == 'set_param':
                    effect = location_map[ui_event[1]][ui_event[2]]
                    effect.set_parameter(ui_event[3], ui_event[4])
                elif ui_event[0] == 'add_effect':
                    types = {'freq-domain': self.freq_domain_key_map.values(), 'time-domain': self.time_domain_key_map.values()}[ui_event[1]]
                    effect_class = next(effect_type for effect_type in types if effect_type.__name__ == ui_event[2])
                    new_effect = effect_class(magnitude=1.0)
                    location_map[ui_event[1]].append(new_effect)
                    if osc == self.template_osc:
                        self.ui.add_effect(new_effect, ui_event[1])

        for osc in [self.template_osc] + list(self.iter_oscs()):
            location_map = {'freq-domain': osc.freq_domain_stages, 'generator': [osc.time_domain_generator], 'time-domain': osc.time_domain_stages}
            for location, stage_list in location_map.items():
                for i, stage in enumerate(stage_list):
                    if stage.parameters_updated:
                        self.ui.update_effect(location, i, stage)
                        stage.parameters_updated = False

        return super().update_output()

    def on_key_on(self, instrument: midi.Input, event: KeyOnMessage, oscs: List[CustomOsc]):
        if event.key_num in self.freq_domain_key_map:
            new_stage = self.freq_domain_key_map[event.key_num](event.velocity)
            for osc in [self.template_osc] + list(self.iter_oscs()):
                osc.freq_domain_stages.append(new_stage)
            self.ui.add_effect(new_stage, 'freq-domain')
        elif event.key_num in self.time_domain_key_map:
            new_stage = self.time_domain_key_map[event.key_num](event.velocity)
            for osc in [self.template_osc] + list(self.iter_oscs()):
                osc.time_domain_stages.append(new_stage)
            self.ui.add_effect(new_stage, 'time-domain')
        elif event.key_num == 88 + 20:
            for osc in [self.template_osc] + list(self.iter_oscs()):
                osc.freq_domain_stages.clear()
                osc.time_domain_stages.clear()
            self.ui.clear_effects('freq-domain')
            self.ui.clear_effects('time-domain')
        else:
            note_frequency = get_piano_key_frequency(event.key_num)
            new_osc = deepcopy(self.template_osc)
            new_osc.frequency = note_frequency
            new_osc.amplitude = event.velocity / 10.0
            oscs.append(new_osc)

    def on_key_off(self, instrument: midi.Input, event: KeyOffMessage, oscs: List[CustomOsc]):
        for osc in oscs:
            osc.fade_rate = 5.0

    def on_control_change(self, instrument: midi.Input, event: ControlChangeMessage):
        for osc in [self.template_osc] + list(self.iter_oscs()):
            for stage in osc.stages:
                if event.control_change_type == ControlChangeMessage.CONTROL_CHANGE_TYPE.SWING_CHANGE:
                    stage.magnitude = event.data_byte_2 / 127.0

        # for stage in self.freq_domain_stages + self.time_domain_stages:
        #     if event.control_change_type == ControlChangeMessage.CONTROL_CHANGE_TYPE.SWING_CHANGE:
        #         stage.magnitude = event.data_byte_2 / 127.0

if __name__ == '__main__':
    # import sounddevice as sd
    # s = sd.OutputStream(samplerate=44100)
    # s.start()
    # s.write(np.zeros((10000, 2), dtype=np.float32))
    # s.write(np.zeros((10000, 2), dtype=np.float32))
    # s.stop()

    test_sine1 = SineStage()
    test_freq_amp1 = (np.full((1000,), 400.0), np.full((1000,), 1.0))
    output1 = test_sine1(test_freq_amp1, 1000, 44100)

    test_sine2 = SineStage()
    test_freq_amp2 = (np.full((500,), 400.0), np.full((500,), 1.0))
    output2 = test_sine2(test_freq_amp2, 500, 44100)
    output3 = test_sine2(test_freq_amp2, 500, 44100)

    assert np.all(np.abs(output1 - np.concatenate((output2, output3), axis=0)) < 1e-6)

    synth = StageStackerSynth({82 + 20: BassHarmonicsStage, 83 + 20: FrequencyDomainWahWahEffect, 85 + 20: HarmonicsStage, 87 + 20: VibratoStage}, {84 + 20: AMEffect, 86 + 20: SimpleDistortEffect})
    i = 0
    print(synth.output_stream.latency)
    while True:
        # if i % 100 == 0:
        #     print(synth.output_stream.latency)
        i += 1
        synth.update_output()

    # import sounddevice
    # test_out = sounddevice.OutputStream(samplerate=44100)
    # test_out.latency