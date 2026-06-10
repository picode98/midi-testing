from math import pi
from abc import ABC
from collections import OrderedDict
import pathlib
import json
from typing import List

import numpy as np
from pygame import midi
# import resampy

from midi_utils import *
from midi_utils import KeyOffMessage, KeyOnMessage, np
from utils import CustomOsc, Fadeable, read_wav_file, write_wav_file

import sample_editor_native_ui

def wrap_waveform(sample: np.ndarray):
    int_parts = np.floor(sample)
    result = sample - int_parts
    reflect_mask = (int_parts % 2 == 1)
    result[reflect_mask] = 1 - result[reflect_mask]

    return result

def triangle_waveform(num_wavelengths: float, num_samples: int):
    return wrap_waveform(np.linspace(0.5, 2.0 * num_wavelengths + 0.5, num_samples, endpoint=False)) * 2.0 - 1.0

def sawtooth_waveform(num_wavelengths: float, num_samples: int):
    return np.fmod(np.linspace(1.0, 2.0 * num_wavelengths + 1.0, num_samples, endpoint=False), 2.0) - 1.0

def sine_waveform(num_wavelengths: float, num_samples: int):
    return np.sin(np.linspace(0.0, 2.0 * np.pi * num_wavelengths, num_samples, endpoint=False))

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
        resampled_buf = np.interp(frame_indices, np.arange(self.sample_buf.shape[0]), self.sample_buf)
        return np.transpose(np.tile(self.amplitude * resampled_buf, (2, 1)))

class SampleEffect(ABC):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        raise NotImplementedError()

class TremoloEffect(SampleEffect):
    def __init__(self, wavelength: float):
        self.wavelength = wavelength

    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        multipliers = np.sin(2.0 * np.pi * np.arange(len(sample_slice)) / self.wavelength) * (0.002 * magnitude) + (1.0 - (0.002 * magnitude))
        sample_slice *= multipliers

class PowerFadeEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        sample_slice[:] = (np.absolute(sample_slice) ** (1.0 + magnitude / 100.0)) * np.sign(sample_slice)
        print(sample_slice[:10])

class PowerStrengthenEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        sample_slice[:] = (np.absolute(sample_slice) ** (1.0 - magnitude / 100.0)) * np.sign(sample_slice)
        print(sample_slice[:10])

class SimpleDistortEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        max_threshold = np.max(sample_slice)
        min_threshold = np.min(sample_slice)
        sample_slice *= (1.0 + magnitude / 1000.0)
        sample_slice[sample_slice >= max_threshold] = max_threshold
        sample_slice[sample_slice <= min_threshold] = min_threshold

class FractionalWavelengthEffect(SampleEffect):
    def __init__(self, multiple: int):
        self.multiple = multiple

    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        interpolated = np.interp(np.linspace(0, len(sample_slice), len(sample_slice) * self.multiple, endpoint=False), np.arange(len(sample_slice)), sample_slice)
        for i in range(0, len(interpolated), len(sample_slice)):
            sample_slice += interpolated[i:i + len(sample_slice)] * magnitude / 100.0

class SmoothingEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        num_frames = 5
        smoothing_term = sample_slice * magnitude / (100.0 * num_frames)
        sample_slice *= 1.0 - magnitude / 100.0
        idx_range = np.arange(sample_slice.shape[0])
        for i in range(1, num_frames + 1):
            sample_slice += smoothing_term[(idx_range + i * 5) % sample_slice.shape[0]]

class QuantizeEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        factor = 20.0
        rounded_values = np.round(sample_slice * factor) / factor
        sample_slice -= (sample_slice - rounded_values) * (magnitude / 100.0)

class NoiseEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        sample_slice += (magnitude / 100.0) * (np.random.random(len(sample_slice)) - 0.5)
        sample_slice[:] = np.clip(sample_slice, -1.0, 1.0)

class SilenceEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        sample_slice[:] = np.maximum(np.abs(sample_slice) - (magnitude / 100.0), 0.0) * np.sign(sample_slice)

class NonDistortingAmplifyEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        max_magnitude = np.max(np.abs(sample_slice))
        if max_magnitude > 0.0:
            sample_slice *= min(1.0 + magnitude / 100.0, 1.0 / max_magnitude)

class MixWithStaticSampleEffect(SampleEffect):
    def __init__(self, mix_sample: np.ndarray):
        self.mix_sample = mix_sample

    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, magnitude: float):
        sample_slice[:] = sample_slice * (1.0 - magnitude / 100.0) + self.mix_sample[slice_offset:slice_offset + len(sample_slice)] * (magnitude / 100.0)

class SampleEditorProjectManager:
    def __init__(self):
        self.project_folder = None
        self.snapshots = []

    def append_snapshot(self, sample_data: np.ndarray):
        self.snapshots.append((f'{len(self.snapshots) + 1:05d}.wav', sample_data.copy()))

    def write_project(self):
        self.project_folder.mkdir(exist_ok=True)

        proj_data_obj = {'snapshots': [rel_path for rel_path, sample_data in self.snapshots]}
        with (self.project_folder / 'project.json').open('w') as proj_file:
            json.dump(proj_data_obj, proj_file)

        for rel_path, sample_data in self.snapshots:
            try:
                with (self.project_folder / rel_path).open('xb') as created_data_file:
                    write_wav_file(created_data_file, sample_data)
            except FileExistsError:
                pass

    def set_project_folder(self, new_path: pathlib.Path):
        self.project_folder = new_path

        try:
            with (new_path / 'project.json').open('r') as proj_file:
                proj_data = json.load(proj_file)
        except FileNotFoundError:
            self.write_project()
            return
        
        self.snapshots: List[Tuple[str, np.ndarray]] = []
        for sample_data_path in proj_data['snapshots']:
            frames, sample_rate = read_wav_file(str(self.project_folder / sample_data_path))
            self.snapshots.append((sample_data_path, frames))





class SampleEditorSynth(CustomSynth):
    def __init__(self, num_wavelengths: int, bank_switch_key: int, effect_map: List[Dict[int, SampleEffect]], sample_size: int = 4410):
        super().__init__(output=sd.OutputStream(device=0, samplerate=44100, latency=0.05))
        self.master_sample = np.sin(np.linspace(0.0, num_wavelengths * (2.0 * pi), sample_size))
        self.master_sample_wavelengths = num_wavelengths
        self.effect_map = effect_map
        self.applying_effects = False
        self.current_effects: OrderedDict[int, Tuple[SampleEffect, float]] = OrderedDict()
        self.edit_min_sample: int = 0
        self.edit_max_sample: int = len(self.master_sample)
        self.bank_switch_key = bank_switch_key
        self.current_bank_index = 0
        self.effect_feather_size = 500
        self.effect_feather_ramp = np.linspace(0.0, 1.0, self.effect_feather_size, endpoint=False)

    def on_key_on(self, instrument: midi.Input, event: KeyOnMessage, oscs: List[CustomOsc]):
        if event.key_num == self.bank_switch_key:
            self.current_bank_index = (self.current_bank_index + 1) % len(self.effect_map)
            print('Switched to bank ' + str(self.current_bank_index))
        elif event.key_num in self.effect_map[self.current_bank_index]:
            self.current_effects[event.key_num] = (self.effect_map[self.current_bank_index][event.key_num], event.velocity / 10.0)
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
            if self.effect_feather_size is not None and (self.edit_min_sample > 0 or self.edit_max_sample < len(self.master_sample)):
                begin_slice = slice(self.edit_min_sample, self.edit_min_sample + self.effect_feather_size)
                end_slice = slice(self.edit_max_sample - self.effect_feather_size, self.edit_max_sample)
                feather_start_region = self.master_sample[begin_slice].copy()
                feather_stop_region =  self.master_sample[end_slice].copy()

            effect.apply_step(self.master_sample[self.edit_min_sample:self.edit_max_sample], self.edit_min_sample, magnitude)

            if self.effect_feather_size is not None and (self.edit_min_sample > 0 or self.edit_max_sample < len(self.master_sample)):
                self.master_sample[begin_slice] = self.master_sample[begin_slice] * self.effect_feather_ramp + (1.0 - self.effect_feather_ramp) * feather_start_region
                self.master_sample[end_slice] = self.master_sample[end_slice] * self.effect_feather_ramp[::-1] + (1.0 - self.effect_feather_ramp[::-1]) * feather_stop_region

        max_val = np.max(np.abs(self.master_sample))
        if max_val > 1.0:
            self.master_sample /= max_val

        super().update_output()

        is_checkpoint = self.applying_effects and len(self.current_effects) == 0
        self.applying_effects = len(self.current_effects) > 0

        return is_checkpoint

    def set_edit_params(self, min_sample: int, max_sample: int):
        self.edit_min_sample, self.edit_max_sample = max(min(min_sample, len(self.master_sample)), 0), max(min(max_sample, len(self.master_sample)), 0)


# def visualization_worker(wave_queue: mp.Queue, edit_params_queue: mp.Queue):
#     import matplotlib.pyplot as plt
#     from matplotlib.animation import FuncAnimation

#     wave_graph = plt.figure()
#     wave_axes = None
#     wave_artist = None
#     existing_x_bounds = None
#     def draw_wave(_):
#         nonlocal wave_axes, wave_artist, existing_x_bounds
#         new_wave = None
#         try:
#             while True:
#                 new_wave = wave_queue.get_nowait()
#         except queue.Empty:
#             pass

#         if new_wave is not None:
#             if wave_axes is None:
#                 wave_axes = wave_graph.gca()
#                 wave_artist = wave_axes.plot(np.arange(new_wave.shape[0]), new_wave)[0]
#                 new_x_bound_min, new_x_bound_max = wave_axes.get_xbound()
#                 existing_x_bounds = (int(round(new_x_bound_min)), int(round(new_x_bound_max)))

#             wave_artist.set_ydata(new_wave)

#         new_x_bound_min, new_x_bound_max = wave_axes.get_xbound()
#         new_x_bounds = (int(round(new_x_bound_min)), int(round(new_x_bound_max)))
#         if new_x_bounds != existing_x_bounds:
#             existing_x_bounds = new_x_bounds
#             edit_params_queue.put_nowait(existing_x_bounds)

#         return wave_artist

#     wave_anim = FuncAnimation(wave_graph, draw_wave)
#     wave_anim.resume()
#     plt.show(block=True)

if __name__ == '__main__':
    synth = SampleEditorSynth(num_wavelengths=80, bank_switch_key=77 + 20,
                              effect_map=[{78 + 20: SilenceEffect(),
                                           79 + 20: NonDistortingAmplifyEffect(),
                                           80 + 20: MixWithStaticSampleEffect(triangle_waveform(80, 88200)),
                                           81 + 20: MixWithStaticSampleEffect(sawtooth_waveform(80, 88200)),
                                           82 + 20: MixWithStaticSampleEffect(sine_waveform(80, 88200))},
                                          {79 + 20: TremoloEffect(10000.0),
                                           80 + 20: FractionalWavelengthEffect(2),
                                           81 + 20: FractionalWavelengthEffect(3),
                                           82 + 20: FractionalWavelengthEffect(5),
                                           83 + 20: NoiseEffect(), 84 + 20: QuantizeEffect(),
                                           85 + 20: PowerFadeEffect(),
                                           86 + 20: PowerStrengthenEffect(),
                                           87 + 20: SimpleDistortEffect(),
                                           88 + 20: SmoothingEffect()}], sample_size=88200)

    proj_manager = SampleEditorProjectManager()
    # wave_queue = mp.Queue()
    # edit_params_queue = mp.Queue()
    editor_ui = sample_editor_native_ui.SampleEditorNativeUI()
    # vis_process = mp.Process(target=visualization_worker, args=(wave_queue, edit_params_queue))
    # vis_process.start()
    i = 0
    while True:
        is_checkpoint = synth.update_output()

        for event in editor_ui.get_events():
            if event[0] == sample_editor_native_ui.OutgoingMessageType.SET_PROJECT_FOLDER:
                _, new_folder = event
                proj_manager.set_project_folder(new_folder)
                if len(proj_manager.snapshots) > 0:
                    synth.master_sample[:] = proj_manager.snapshots[-1][1]
                editor_ui.load_project_data([sample_data for file_path, sample_data in proj_manager.snapshots])
            elif event[0] == sample_editor_native_ui.OutgoingMessageType.SET_EDIT_WINDOW:
                _, (new_lbound, new_hbound) = event
                print(event)
                synth.set_edit_params(new_lbound, new_hbound)
            elif event[0] == sample_editor_native_ui.OutgoingMessageType.SET_ACTIVE_SNAPSHOT:
                _, snapshot_index = event
                synth.master_sample[:] = proj_manager.snapshots[snapshot_index][1]


        if is_checkpoint:
            proj_manager.append_snapshot(synth.master_sample)
            if proj_manager.project_folder is not None:
                proj_manager.write_project()

        if is_checkpoint or i % 10000 == 0:
            editor_ui.update_current_sample(synth.master_sample, is_checkpoint, '')
            # wave_queue.put(synth.master_sample)

            # new_edit_params = None
            # try:
            #     while True:
            #         new_edit_params = edit_params_queue.get_nowait()
            # except queue.Empty:
            #     pass

            # if new_edit_params is not None:
            #     print(new_edit_params)
            #     synth.set_edit_params(new_edit_params[0], new_edit_params[1])


        i += 1


# osc1 = SampleRepeatingOsc(np.sin(np.linspace(0.0, 1 * (2.0 * pi), 4410)), 1, 449.7)
# buf1 = osc1.play_frames(20)
# osc2 = SampleRepeatingOsc(np.sin(np.linspace(0.0, 1 * (2.0 * pi), 4410)), 1, 449.7)
# buf2 = osc2.play_frames(10)
# buf3 = osc2.play_frames(10)
# print(buf1 - np.concatenate([buf2, buf3]))
# pass
