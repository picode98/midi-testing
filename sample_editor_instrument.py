from math import pi
from abc import ABC
from collections import OrderedDict
import queue
import multiprocessing as mp
from typing import List

import numpy as np
from pygame import midi
import resampy

from midi_utils import *
from midi_utils import KeyOffMessage, KeyOnMessage, np
from utils import CustomOsc, Fadeable

@Fadeable
class SampleRepeatingOsc(CustomOsc):
    def __init__(self, sample_buf: np.ndarray, num_wavelengths: int, frequency: float, amplitude: float, sample_rate=44100):
        super().__init__(sample_rate)
        self.sample_buf = sample_buf
        self.sample_offset = 0.0
        self.frequency = frequency
        self.sample_frequency = sample_rate * num_wavelengths / self.sample_buf.shape[0]
        self.amplitude = amplitude
    
    def play_frames(self, num_frames: int):
        buf_max_index = self.sample_buf.shape[0] - 1
        frame_indices = (np.arange(num_frames) * (self.frequency / self.sample_frequency) + self.sample_offset) % buf_max_index
        # print(frame_indices)
        assert frame_indices.shape[0] == num_frames
        self.sample_offset = (self.sample_offset + num_frames * (self.frequency / self.sample_frequency)) % buf_max_index
        resampled_buf = resampy.resample_nu(self.sample_buf, 1.0, frame_indices)
        return np.transpose(np.tile(self.amplitude * resampled_buf, (2, 1)))

class SampleEffect(ABC):
    def apply_step(self, sample: np.ndarray, magnitude: float):
        raise NotImplementedError()

class PowerFadeEffect(SampleEffect):
    def apply_step(self, sample: np.ndarray, magnitude: float):
        sample[:] = (np.absolute(sample) ** (1.0 + magnitude / 1000.0)) * np.sign(sample)

class PowerStrengthenEffect(SampleEffect):
    def apply_step(self, sample: np.ndarray, magnitude: float):
        sample[:] = (np.absolute(sample) ** (1.0 - magnitude / 1000.0)) * np.sign(sample)

class SimpleDistortEffect(SampleEffect):
    def apply_step(self, sample: np.ndarray, magnitude: float):
        max_threshold = np.max(sample)
        min_threshold = np.min(sample)
        sample *= (1.0 + magnitude / 1000.0)
        sample[sample >= max_threshold] = max_threshold
        sample[sample <= min_threshold] = min_threshold

class HalfWavelengthEffect(SampleEffect):
    def apply_step(self, sample: np.ndarray, magnitude: float):
        doubled = resampy.resample(sample, 1, 2)
        sample += doubled[:len(sample)] * magnitude / 100.0
        sample += doubled[len(sample):] * magnitude / 100.0

class SmoothingEffect(SampleEffect):
    def apply_step(self, sample: np.ndarray, magnitude: float):
        num_frames = 5
        smoothing_term = sample * magnitude / (100.0 * num_frames)
        sample *= 1.0 - magnitude / 100.0
        idx_range = np.arange(sample.shape[0])
        for i in range(1, num_frames + 1):
            sample += smoothing_term[(idx_range + i * 5) % sample.shape[0]]

class QuantizeEffect(SampleEffect):
    def apply_step(self, sample: np.ndarray, magnitude: float):
        factor = 20.0
        rounded_values = np.round(sample * factor) / factor
        sample -= (sample - rounded_values) * (magnitude / 100.0)

class SampleEditorSynth(CustomSynth):
    def __init__(self, num_wavelengths: int, effect_map: Dict[int, SampleEffect], sample_size: int = 4410):
        super().__init__()
        self.master_sample = np.sin(np.linspace(0.0, num_wavelengths * (2.0 * pi), sample_size))
        self.master_sample_wavelengths = num_wavelengths
        self.effect_map = effect_map
        self.current_effects: OrderedDict[int, Tuple[SampleEffect, float]] = OrderedDict()

    def on_key_on(self, instrument: midi.Input, event: KeyOnMessage, oscs: List[CustomOsc]):
        if event.key_num in self.effect_map:
            self.current_effects[event.key_num] = (self.effect_map[event.key_num], event.velocity / 10.0)
        else:
            note_frequency = get_piano_key_frequency(event.key_num)
            oscs += [SampleRepeatingOsc(self.master_sample, self.master_sample_wavelengths, note_frequency, event.velocity / 10.0)]

    def on_key_off(self, instrument: midi.Input, event: KeyOffMessage, oscs: List[CustomOsc]):
        if event.key_num in self.current_effects:
            del self.current_effects[event.key_num]
        else:
            for osc in oscs:
                osc.fade_rate = 5.0

    def update_output(self):
        for effect, magnitude in self.current_effects.values():
            effect.apply_step(self.master_sample, magnitude)

        max_val = np.max(np.abs(self.master_sample))
        if max_val > 1.0:
            self.master_sample /= max_val

        super().update_output()

synth = SampleEditorSynth(num_wavelengths=80, effect_map={83 + 20: QuantizeEffect(), 84 + 20: HalfWavelengthEffect(), 85 + 20: PowerFadeEffect(), 86 + 20: PowerStrengthenEffect(),
                                                         87 + 20: SimpleDistortEffect(), 88 + 20: SmoothingEffect()}, sample_size=88200)

def visualization_worker(wave_queue: mp.Queue):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    wave_graph = plt.figure()
    wave_axes = None
    wave_artist = None
    def draw_wave(_):
        nonlocal wave_axes, wave_artist
        new_wave = None
        try:
            while True:
                new_wave = wave_queue.get_nowait()
        except queue.Empty:
            pass

        if new_wave is not None:
            if wave_axes is None:
                wave_axes = wave_graph.gca()
                wave_artist = wave_axes.plot(np.arange(new_wave.shape[0]), new_wave)[0]

            wave_artist.set_ydata(new_wave)

        return wave_artist

    wave_anim = FuncAnimation(wave_graph, draw_wave)
    wave_anim.resume()
    plt.show(block=True)

if __name__ == '__main__':
    wave_queue = mp.Queue()
    vis_process = mp.Process(target=visualization_worker, args=(wave_queue,))
    vis_process.start()
    i = 0
    while True:
        synth.update_output()

        if i % 1000 == 0:
            wave_queue.put(synth.master_sample)

        i += 1


# osc1 = SampleRepeatingOsc(np.sin(np.linspace(0.0, 1 * (2.0 * pi), 4410)), 1, 449.7)
# buf1 = osc1.play_frames(20)
# osc2 = SampleRepeatingOsc(np.sin(np.linspace(0.0, 1 * (2.0 * pi), 4410)), 1, 449.7)
# buf2 = osc2.play_frames(10)
# buf3 = osc2.play_frames(10)
# print(buf1 - np.concatenate([buf2, buf3]))
# pass
