from abc import ABC
from math import pi
from copy import deepcopy
from typing import Any, Dict, List, Type, Tuple, Callable

import numpy as np
from pygame import midi

from midi_utils import CustomSynth, KeyOffMessage, KeyOnMessage, get_piano_key_frequency
from utils import CustomOsc, Fadeable

class SampleStage(ABC):
    def __init__(self, magnitude: float):
        self.magnitude = magnitude

    def __call__(self, input, num_frames: int, sample_rate: int) -> Any:
        raise NotImplementedError()
    
class SineStage(SampleStage):
    def __init__(self):
        self.phase = 0.0

    def __call__(self, input: Tuple[np.ndarray, np.ndarray], num_frames: int, sample_rate: int) -> Any:
        phases = 2.0 * pi * np.cumsum(np.pad(input[0], pad_width=(1, 0), constant_values=0)) / sample_rate + self.phase
        result = np.sin(phases[:-1]) * input[1]
        self.phase = phases[-1] % (2.0 * pi)
        return result

class VibratoStage(SampleStage):
    def __init__(self, magnitude: float):
        super().__init__(magnitude)
        self.frequency = 10.0
        self.phase = 0.0

    def __call__(self, input: List[Tuple[np.ndarray, np.ndarray]], num_frames: int, sample_rate: int) -> Any:
        delta = 2.0 * pi * num_frames * self.frequency / sample_rate
        freq_deltas = self.magnitude * 0.02 * np.sin(np.linspace(self.phase, self.phase + delta, num_frames, endpoint=False)) + 1.0
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
        self.components = []
        self.new_component_factory = new_component_factory

    def __call__(self, input: List, num_frames: int, sample_rate: int):
        if len(input) < len(self.components):
            self.components = self.components[:len(input)]
        elif len(input) > len(self.components):
            self.components += [self.new_component_factory() for _ in range(len(input) - len(self.components))]

        assert len(input) == len(self.components)
        return sum(stage(elem, num_frames, sample_rate) for elem, stage in zip(input, self.components))
    

class HarmonicsStage(SampleStage):
    def __call__(self, input: List[Tuple[np.ndarray, np.ndarray]], num_frames: int, sample_rate: int):
        result = []
        for component_freq, component_amp in input:
            result += [(component_freq, component_amp), (component_freq / 2.0, component_amp / 2.0),
                       (component_freq / 4.0, component_amp / 3.0), (component_freq * 2.0, component_amp / 10.0)]
            
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
        delta = 2.0 * pi * num_frames * (1.0 / (1.0 - self.magnitude * 0.95)) / sample_rate
        amplitudes = np.cos(np.linspace(self.phase, self.phase + delta, num_frames, endpoint=False))
        self.phase = (self.phase + delta) % (2.0 * pi)
        return input * amplitudes
    
class FrequencyDomainWahWahEffect(SampleStage):
    def __init__(self, magnitude: float):
        super().__init__(magnitude)
        self.phase = 0.0
        self.dist_center_min, self.dist_center_max = 1.0, 12.0

    def __call__(self, input: List[Tuple[np.ndarray, np.ndarray]], num_frames: int, sample_rate: int):
        delta = 2.0 * pi * num_frames * (1.0 / (1.0 - self.magnitude * 0.95)) / sample_rate
        dist_centers = self.dist_center_min + (self.dist_center_max - self.dist_center_min) \
            * (np.sin(np.linspace(self.phase, self.phase + delta, num_frames, endpoint=False)) + 1.0) / 2.0
        log_freqs = np.stack([np.log2(freqs) for freqs, _ in input], axis=1)
        # center_freqs, freq_stddevs = np.average(log_freqs, axis=1), np.std(log_freqs, axis=1)
        normalized_log_freqs = (log_freqs - np.reshape(dist_centers, (num_frames, 1))) / 1.0
        amp_multipliers = 1.0 / (normalized_log_freqs ** 2 + 1.0)
        # amp_multipliers /= np.reshape(np.sum(amp_multipliers, axis=1), (num_frames, 1))
        amp_mat = np.stack([amps for _, amps in input], axis=1)
        scaled_amp_mat = amp_mat * amp_multipliers
        result_amps = scaled_amp_mat * np.reshape(np.sum(amp_mat, axis=1) / np.sum(scaled_amp_mat, axis=1), (num_frames, 1))
        self.phase = (self.phase + delta) % (2.0 * pi)
        return [(freqs, result_amps[:, i]) for i, (freqs, _) in enumerate(input)]

@Fadeable
class StageStackOsc(CustomOsc):
    def __init__(self, frequency: float, amplitude: float, stages: List[SampleStage]):
        super().__init__()
        self.stages = stages
        self.frequency = frequency
        self.amplitude = amplitude

    def play_frames(self, num_frames: int):
        curr_result = [(np.full((num_frames,), self.frequency), np.full((num_frames,), self.amplitude))]

        for stage in self.stages:
            curr_result = stage(curr_result, num_frames, self.sample_rate)

        assert isinstance(curr_result, np.ndarray)
        return curr_result
    
class StageStackerSynth(CustomSynth):
    def __init__(self, freq_domain_key_map: Dict[int, Type[SampleStage]], time_domain_key_map: Dict[int, Type[SampleStage]]):
        super().__init__()
        self.freq_domain_key_map = freq_domain_key_map
        self.time_domain_key_map = time_domain_key_map

        self.freq_domain_stages = []
        self.time_domain_generator = MixerStage(lambda: SineStage()) # MixerStage([SineStage() for _ in range(4)])
        self.time_domain_stages = []
        # self.generator_stack = [HarmonicsStage(), VibratoStage(0.015), MixerStage([SineStage() for _ in range(4)])]

    def on_key_on(self, instrument: midi.Input, event: KeyOnMessage, oscs: List[CustomOsc]):
        if event.key_num in self.freq_domain_key_map:
            self.freq_domain_stages.append(self.freq_domain_key_map[event.key_num](event.velocity))
        elif event.key_num in self.time_domain_key_map:
            self.time_domain_stages.append(self.time_domain_key_map[event.key_num](event.velocity))
        elif event.key_num == 88 + 20:
            self.freq_domain_stages.clear()
            self.time_domain_stages.clear()
        else:
            note_frequency = get_piano_key_frequency(event.key_num)
            oscs.append(StageStackOsc(note_frequency, event.velocity / 10.0, deepcopy(self.freq_domain_stages + [self.time_domain_generator] + self.time_domain_stages)))

    def on_key_off(self, instrument: midi.Input, event: KeyOffMessage, oscs: List[CustomOsc]):
        for osc in oscs:
            osc.fade_rate = 5.0

test_sine1 = SineStage()
test_freq_amp1 = (np.full((1000,), 400.0), np.full((1000,), 1.0))
output1 = test_sine1(test_freq_amp1, 1000, 44100)

test_sine2 = SineStage()
test_freq_amp2 = (np.full((500,), 400.0), np.full((500,), 1.0))
output2 = test_sine2(test_freq_amp2, 500, 44100)
output3 = test_sine2(test_freq_amp2, 500, 44100)

assert np.all(np.abs(output1 - np.concatenate((output2, output3), axis=0)) < 1e-6)

synth = StageStackerSynth({83 + 20: FrequencyDomainWahWahEffect, 85 + 20: HarmonicsStage, 87 + 20: VibratoStage}, {84 + 20: AMEffect, 86 + 20: SimpleDistortEffect})
while True:
    synth.update_output()