import dataclasses
from math import pi
from abc import ABC
from collections import OrderedDict
import pathlib
import json
from typing import Type

import numpy as np
from pygame import midi
import dataclasses_json
# import resampy

from sample_editor_effect import EffectSettings, SampleEffect
from midi_utils import *
from midi_utils import KeyOffMessage, KeyOnMessage, np
from utils import CustomOsc, Fadeable, read_wav_file, write_wav_file, sentence_from_list

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

def square_waveform(num_wavelengths: float, num_samples: int, duty_cycle: float):
    sawtooth_wave = sawtooth_waveform(num_wavelengths, num_samples)
    mask = sawtooth_wave > (duty_cycle * 2.0 - 1.0)
    sawtooth_wave[mask] = 1.0
    sawtooth_wave[~mask] = -1.0

    return sawtooth_wave

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

class TremoloEffect(SampleEffect):
    def __init__(self, wavelength: float):
        super().__init__()
        self.wavelength = wavelength

    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        multipliers = np.sin(2.0 * np.pi * np.arange(len(sample_slice)) / (self.wavelength * resolution)) * (0.002 * magnitude) + (1.0 - (0.002 * magnitude))
        sample_slice *= multipliers

    def get_settings(self):
        return {**super().get_settings(), 'wavelength': EffectSettings('Wavelength', float, 0.0, None)}
    
class VibratoEffect(SampleEffect):
    def __init__(self, wavelength: float):
        super().__init__()
        self.wavelength = wavelength

    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        base_indices = np.arange(len(sample_slice))
        sampling_indices = base_indices + (magnitude / 10.0) * np.sin(2.0 * np.pi * base_indices / (self.wavelength * resolution))
        sample_slice[:] = np.interp(sampling_indices, base_indices, sample_slice)

    def get_settings(self):
        return {**super().get_settings(), 'wavelength': EffectSettings('Wavelength', float, 0.0, None)}

class PowerFadeEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        sample_slice[:] = (np.absolute(sample_slice) ** (1.0 + magnitude / 100.0)) * np.sign(sample_slice)
        print(sample_slice[:10])

class PowerStrengthenEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        sample_slice[:] = (np.absolute(sample_slice) ** (1.0 - magnitude / 100.0)) * np.sign(sample_slice)
        print(sample_slice[:10])

class SimpleDistortEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        max_threshold = np.max(sample_slice)
        min_threshold = np.min(sample_slice)
        sample_slice *= (1.0 + magnitude / 1000.0)
        sample_slice[sample_slice >= max_threshold] = max_threshold
        sample_slice[sample_slice <= min_threshold] = min_threshold

class FractionalWavelengthEffect(SampleEffect):
    def __init__(self, multiple: int):
        super().__init__()
        self.multiple = multiple

    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        interpolated = np.interp(np.linspace(0, len(sample_slice), len(sample_slice) * self.multiple, endpoint=False), np.arange(len(sample_slice)), sample_slice)
        for i in range(0, len(interpolated), len(sample_slice)):
            sample_slice += interpolated[i:i + len(sample_slice)] * magnitude / 100.0

    def get_settings(self):
        return {**super().get_settings(), 'multiple': EffectSettings('Wavelength multiple', int, 2, 20)}

class SmoothingEffect(SampleEffect):
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size

    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        smoothing_term = sample_slice * magnitude / (100.0 * self.window_size)
        sample_slice *= 1.0 - magnitude / 100.0
        idx_range = np.arange(sample_slice.shape[0])
        for i in range(1, self.window_size + 1):
            sample_slice += smoothing_term[(idx_range + i * 5) % sample_slice.shape[0]]

    def get_settings(self):
        return {**super().get_settings(), 'window_size': EffectSettings('Window size', int, 0, 100)}

class QuantizeEffect(SampleEffect):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        rounded_values = np.round(sample_slice * self.factor) / self.factor
        sample_slice -= (sample_slice - rounded_values) * (magnitude / 100.0)

    def get_settings(self):
        return {**super().get_settings(), 'factor': EffectSettings('Quantization factor', float, 1.0, None)}

class NoiseEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        sample_slice += (magnitude / 100.0) * (np.random.random(len(sample_slice)) - 0.5)
        sample_slice[:] = np.clip(sample_slice, -1.0, 1.0)

class SilenceEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        sample_slice[:] = np.maximum(np.abs(sample_slice) - (magnitude / 100.0), 0.0) * np.sign(sample_slice)

class NonDistortingAmplifyEffect(SampleEffect):
    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        max_magnitude = np.max(np.abs(sample_slice))
        if max_magnitude > 0.0:
            sample_slice *= min(1.0 + magnitude / 100.0, 1.0 / max_magnitude)

class MixWithStaticSampleEffect(SampleEffect):
    def __init__(self):
        super().__init__()
        self.mix_sample: Optional[np.ndarray] = None
        self.frequency_multiple = 1.0
        self._cached_frequency_multiple = self.frequency_multiple

    def generate_sample(self, length: int, num_wavelengths: float) -> np.ndarray:
        return NotImplemented

    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        if self.mix_sample is None or len(self.mix_sample) < slice_offset + len(sample_slice) or self.frequency_multiple != self._cached_frequency_multiple:
            self.mix_sample = self.generate_sample(slice_offset + len(sample_slice), self.frequency_multiple * (slice_offset + len(sample_slice)) / resolution)
            self._cached_frequency_multiple = self.frequency_multiple
        
        sample_slice[:] = sample_slice * (1.0 - magnitude / 100.0) + self.mix_sample[slice_offset:slice_offset + len(sample_slice)] * (magnitude / 100.0)

    def get_settings(self):
        return {**super().get_settings(), 'frequency_multiple': EffectSettings('Frequency multiple', float, 0.0, None)}

class MixWithTriangleWaveEffect(MixWithStaticSampleEffect):
    def generate_sample(self, length: int, num_wavelengths: float):
        return triangle_waveform(num_wavelengths, length)
    
class MixWithSawtoothWaveEffect(MixWithStaticSampleEffect):
    def generate_sample(self, length: int, num_wavelengths: float):
        return sawtooth_waveform(num_wavelengths, length)
    
class MixWithSineWaveEffect(MixWithStaticSampleEffect):
    def generate_sample(self, length: int, num_wavelengths: float):
        return sine_waveform(num_wavelengths, length)
    
class MixWithSquareWaveEffect(MixWithStaticSampleEffect):
    def __init__(self):
        super().__init__()
        self.duty_cycle = 0.5

    def generate_sample(self, length: int, num_wavelengths: float):
        return square_waveform(num_wavelengths, length, self.duty_cycle)
    
    def get_settings(self):
        return {**super().get_settings(), 'duty_cycle': EffectSettings('Duty cycle', float, 0.0, 1.0)}

@dataclasses.dataclass
class SampleEditorSnapshot(dataclasses_json.DataClassJsonMixin):
    path: str
    num_wavelengths: int
    description: str

@dataclasses.dataclass
class SampleEditorProjectSettings(dataclasses_json.DataClassJsonMixin):
    snapshots: List[SampleEditorSnapshot] = dataclasses.field(default_factory=lambda: [])
    effect_setting_values: Dict[str, Dict[str, int | float | str]] = dataclasses.field(default_factory=lambda: dict())

class SampleEditorProjectManager:
    def __init__(self):
        self.project_folder: Optional[pathlib.Path] = None
        self.settings = SampleEditorProjectSettings()
        self.snapshot_data = []

    def append_snapshot(self, sample_data: np.ndarray, num_wavelengths: int, sample_desc: str):
        new_snapshot_path = f'{len(self.snapshot_data) + 1:05d}.wav'
        self.snapshot_data.append((new_snapshot_path, sample_data.copy()))
        self.settings.snapshots.append(SampleEditorSnapshot(new_snapshot_path, num_wavelengths, sample_desc))

    def write_project(self):
        assert self.project_folder is not None
        self.project_folder.mkdir(exist_ok=True)

        proj_data_obj = self.settings.to_dict()
        with (self.project_folder / 'project.json').open('w') as proj_file:
            json.dump(proj_data_obj, proj_file)

        for rel_path, sample_data in self.snapshot_data:
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
        
        self.settings = SampleEditorProjectSettings.from_dict(proj_data)
        
        self.snapshot_data: List[Tuple[str, np.ndarray]] = []
        for snapshot in self.settings.snapshots:
            frames, sample_rate = read_wav_file(str(self.project_folder / snapshot.path))
            self.snapshot_data.append((snapshot.path, frames))





class SampleEditorSynth(CustomSynth):
    def __init__(self, num_wavelengths: int, bank_switch_key: int, effect_map: List[Dict[int, SampleEffect]], sample_resolution: int = 1000):
        super().__init__(output=sd.OutputStream(device=0, samplerate=44100, latency=0.05))
        self.master_sample = np.sin(np.linspace(0.0, num_wavelengths * (2.0 * pi), num_wavelengths * sample_resolution))
        self.master_sample_wavelengths = num_wavelengths
        self.sample_resolution = sample_resolution
        self.effect_map = effect_map
        self.applying_effects = False
        self.current_effects: OrderedDict[int, Tuple[SampleEffect, float]] = OrderedDict()
        self.effects_since_last_checkpoint: List[SampleEffect] = []
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
            new_effect = self.effect_map[self.current_bank_index][event.key_num]
            self.current_effects[event.key_num] = (new_effect, event.velocity / 10.0)
            self.effects_since_last_checkpoint.append(new_effect)
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

            effect.apply_step(self.master_sample[self.edit_min_sample:self.edit_max_sample], self.edit_min_sample, self.sample_resolution, magnitude * effect.sensitivity)

            if self.effect_feather_size is not None and (self.edit_min_sample > 0 or self.edit_max_sample < len(self.master_sample)):
                self.master_sample[begin_slice] = self.master_sample[begin_slice] * self.effect_feather_ramp + (1.0 - self.effect_feather_ramp) * feather_start_region
                self.master_sample[end_slice] = self.master_sample[end_slice] * self.effect_feather_ramp[::-1] + (1.0 - self.effect_feather_ramp[::-1]) * feather_stop_region

        max_val = np.max(np.abs(self.master_sample))
        if max_val > 1.0:
            self.master_sample /= max_val

        super().update_output()

        is_checkpoint = self.applying_effects and len(self.current_effects) == 0
        self.applying_effects = len(self.current_effects) > 0

        if is_checkpoint:
            effect_list = self.effects_since_last_checkpoint
            self.effects_since_last_checkpoint = []
        else:
            effect_list = None

        return is_checkpoint, effect_list

    def set_edit_params(self, min_sample: int, max_sample: int):
        self.edit_min_sample, self.edit_max_sample = max(min(min_sample, len(self.master_sample)), 0), max(min(max_sample, len(self.master_sample)), 0)
    
    def set_master_sample(self, new_sample: np.ndarray, new_num_wavelengths: int):
        self.master_sample_wavelengths = new_num_wavelengths

        self.master_sample = new_sample
        for osc_list in self.active_oscs.values():
            for osc in osc_list:
                osc.sample_buf = self.master_sample
                if osc.sample_offset >= len(self.master_sample):
                    osc.sample_offset = 0

    def resize_sample(self, new_num_wavelengths: int, add_mode: sample_editor_native_ui.WavelengthAddMode):
        new_sample = np.resize(self.master_sample, (new_num_wavelengths * self.sample_resolution,))

        if len(new_sample) > len(self.master_sample):
            # Looping occurs by default; we only need to considers the other modes
            if add_mode == sample_editor_native_ui.WavelengthAddMode.ADD_SILENCE:
                new_sample[len(self.master_sample):] = 0.0
            elif add_mode == sample_editor_native_ui.WavelengthAddMode.ADD_SINE_WAVE:
                sine_points = np.sin(np.linspace(-2.0 * np.pi * (len(new_sample) - len(self.master_sample)) / self.sample_resolution, 0.0, len(new_sample) - len(self.master_sample)))
                new_sample[len(self.master_sample):] = sine_points

        self.set_master_sample(new_sample, new_num_wavelengths)

            


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
                                           80 + 20: MixWithTriangleWaveEffect(),
                                           81 + 20: MixWithSawtoothWaveEffect(),
                                           82 + 20: MixWithSineWaveEffect(),
                                           83 + 20: MixWithSquareWaveEffect()},
                                          {80 + 20: TremoloEffect(50.0),
                                           81 + 20: VibratoEffect(20.0),
                                           82 + 20: FractionalWavelengthEffect(2),
                                           83 + 20: NoiseEffect(), 84 + 20: QuantizeEffect(20.0),
                                           85 + 20: PowerFadeEffect(),
                                           86 + 20: PowerStrengthenEffect(),
                                           87 + 20: SimpleDistortEffect(),
                                           88 + 20: SmoothingEffect(5)}], sample_resolution=1000)

    proj_manager = SampleEditorProjectManager()
    proj_manager.append_snapshot(synth.master_sample, synth.master_sample_wavelengths, 'Default sine-wave sample')
    # wave_queue = mp.Queue()
    # edit_params_queue = mp.Queue()
    effects_by_type = {type(effect).__name__: effect for bank_map in synth.effect_map for effect in bank_map.values()}
    all_settings_info = {effect_type: effect.get_settings() for effect_type, effect in effects_by_type.items()}
    initial_setting_values = {effect_type: {setting_key: getattr(effects_by_type[effect_type], setting_key) for setting_key in effect_settings.keys()} for effect_type, effect_settings in all_settings_info.items()}
    proj_manager.settings.effect_setting_values = initial_setting_values

    editor_ui = sample_editor_native_ui.SampleEditorNativeUI(all_settings_info, initial_setting_values)
    editor_ui.update_current_sample(synth.master_sample, synth.master_sample_wavelengths, True, 'Default sine-wave sample', '')
    # vis_process = mp.Process(target=visualization_worker, args=(wave_queue, edit_params_queue))
    # vis_process.start()
    i = 0
    while True:
        is_checkpoint, effects_since_last_checkpoint = synth.update_output()
        sample_desc = ''

        for event in editor_ui.get_events():
            if event[0] == sample_editor_native_ui.OutgoingMessageType.SET_PROJECT_FOLDER:
                _, new_folder = event
                proj_manager.set_project_folder(new_folder)
                if len(proj_manager.snapshot_data) > 0:
                    synth.set_master_sample(proj_manager.snapshot_data[-1][1].copy(), proj_manager.settings.snapshots[-1].num_wavelengths)

                for effect_name, effect_settings in proj_manager.settings.effect_setting_values.items():
                    for setting_key, setting_value in effect_settings.items():
                        setattr(effects_by_type[effect_name], setting_key, setting_value)

                editor_ui.load_project_data(new_folder, [(sample_data, metadata.num_wavelengths, metadata.description) for (file_path, sample_data), metadata in zip(proj_manager.snapshot_data, proj_manager.settings.snapshots)], proj_manager.settings.effect_setting_values)
            elif event[0] == sample_editor_native_ui.OutgoingMessageType.SET_EDIT_WINDOW:
                _, (new_lbound, new_hbound) = event
                print(event)
                synth.set_edit_params(new_lbound, new_hbound)
            elif event[0] == sample_editor_native_ui.OutgoingMessageType.SET_ACTIVE_SNAPSHOT:
                _, snapshot_index = event
                synth.set_master_sample(proj_manager.snapshot_data[snapshot_index][1].copy(), proj_manager.settings.snapshots[snapshot_index].num_wavelengths)
            elif event[0] == sample_editor_native_ui.OutgoingMessageType.SET_EFFECT_SETTING_VALUE:
                _, effect_type_name, setting_key, new_value = event
                setattr(effects_by_type[effect_type_name], setting_key, new_value)
                print(setting_key + ' for ' + effect_type_name + ' was changed to ' + str(new_value) + '.')
            elif event[0] == sample_editor_native_ui.OutgoingMessageType.SET_SAMPLE_SIZE:
                _, new_wavelengths, add_mode = event
                sample_desc = f'Resized sample from {synth.master_sample_wavelengths} to {new_wavelengths} wavelengths with mode {add_mode}.'
                synth.resize_sample(new_wavelengths, add_mode)
                is_checkpoint = True
            elif event[0] == sample_editor_native_ui.OutgoingMessageType.RECORD_START:
                _, record_path = event
                synth.start_record(record_path, 2)
                editor_ui.update_recording_status(True, False, 0.0)
            elif event[0] == sample_editor_native_ui.OutgoingMessageType.RECORD_PAUSE:
                synth.pause_record()
                editor_ui.update_recording_status(True, True, synth.samples_recorded / synth.output_stream.samplerate)
            elif event[0] == sample_editor_native_ui.OutgoingMessageType.RECORD_CONTINUE:
                synth.resume_record()
                editor_ui.update_recording_status(True, False, synth.samples_recorded / synth.output_stream.samplerate)
            elif event[0] == sample_editor_native_ui.OutgoingMessageType.RECORD_STOP:
                synth.stop_record()
                editor_ui.update_recording_status(False, False, None)
            elif event[0] == sample_editor_native_ui.OutgoingMessageType.APPLICATION_EXIT:
                exit(0)


        if is_checkpoint:
            if effects_since_last_checkpoint is not None:
                sample_desc = 'Applied ' + sentence_from_list([type(effect).__name__ for effect in effects_since_last_checkpoint])

            proj_manager.append_snapshot(synth.master_sample, synth.master_sample_wavelengths, sample_desc)
            if proj_manager.project_folder is not None:
                proj_manager.write_project()

            editor_ui.update_current_sample(synth.master_sample, synth.master_sample_wavelengths, True, sample_desc, (type(effects_since_last_checkpoint[-1]).__name__ if effects_since_last_checkpoint else ''))
        elif i % 10000 == 0:
            editor_ui.update_current_sample(synth.master_sample, synth.master_sample_wavelengths, False, '', '')

        if i % 1000 == 0 and synth.record_stream is not None:
            editor_ui.update_recording_status(True, synth.record_paused, synth.samples_recorded / synth.output_stream.samplerate)
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
