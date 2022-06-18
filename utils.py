import wave
from abc import ABC

import sounddevice as sd

import numpy as np
import math


def get_note_frequency(octave: int, note: str):
    NOTES = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    return 440 * (2 ** ((octave - 4) + NOTES.index(note) / 12))


def arctan_transition(t: float, deadband=10):
    if t <= -deadband:
        return 0
    elif t >= deadband:
        return 1
    else:
        return (math.atan(t) + math.pi / 2) / math.pi


def fade_pulse_transition(t: float, deadband=10):
    if t <= 0:
        return 1
    elif t >= deadband:
        return 0
    else:
        phase = t - int(t)
        cycle = int(t)
        return (1 - phase) / (2 ** cycle)


def time_shift(buffer: np.ndarray, time_to_shift: float, preserve_length: bool = True, sample_rate: float = 44100):
    shift_buffer = np.zeros((round(abs(time_to_shift) * sample_rate), buffer.shape[1]), buffer.dtype)

    if time_to_shift >= 0:
        if preserve_length:
            return np.concatenate((shift_buffer, buffer[:buffer.shape[0] - shift_buffer.shape[0]]), axis=0)
        else:
            return np.concatenate((shift_buffer, buffer), axis=0)
    else:
        if preserve_length:
            return np.concatenate((buffer[buffer.shape[0] - shift_buffer.shape[0]:], shift_buffer), axis=0)
        else:
            return np.concatenate((buffer, shift_buffer), axis=0)


def write_wav_file(file_name: str, frames: np.ndarray, sample_width: int = 2, sample_rate: int = 44100):
    wave_obj: wave.Wave_write = wave.open(file_name, 'wb')
    wave_obj.setnframes(frames.shape[0])
    wave_obj.setnchannels(frames.shape[1])
    wave_obj.setsampwidth(sample_width)
    wave_obj.setframerate(sample_rate)

    arr_dtype: np.dtype = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[sample_width]

    sample_max = 2 ** (8 * sample_width - 1) - 1
    scaled_samples = (frames * sample_max).clip(min=-sample_max, max=sample_max).astype(arr_dtype)
    byte_array = scaled_samples.tobytes('C')
    wave_obj.writeframes(byte_array)
    wave_obj.close()


class CustomOsc(ABC):
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def play_frames(self, num_frames: int):
        return NotImplemented

    def play(self, length):
        total_frames = int(length * self.sample_rate)
        return self.play_frames(total_frames)

    def is_complete(self) -> bool:
        return False


def Fadeable(cls):
    class FadeableMixin(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fade_coeff: float = 1
            self.fade_rate: float = 0

        def play_frames(self, num_frames: int):
            new_fade_coeff = self.fade_coeff - num_frames * self.fade_rate / self.sample_rate
            fade_mul_array = np.linspace(self.fade_coeff, new_fade_coeff, num_frames)[:, np.newaxis]

            if new_fade_coeff < 0:
                new_fade_coeff = 0
                fade_mul_array[fade_mul_array < 0] = 0

            faded_input = super().play_frames(num_frames) * fade_mul_array

            self.fade_coeff = new_fade_coeff
            return faded_input

        def is_complete(self) -> bool:
            return super().is_complete() or self.fade_coeff == 0

    return FadeableMixin


@Fadeable
class CustomSineOsc(CustomOsc):
    def __init__(self, param_callback, sample_rate=44100):
        super().__init__(sample_rate)
        self.param_callback = param_callback
        self.phase = 0
        self.osc_time = 0

    def play_frames(self, num_frames: int):
        frame_array = np.zeros((num_frames, 2), dtype='d')

        for i in range(num_frames):
            freq, amp = self.param_callback(self.osc_time)
            self.phase += 2 * math.pi * freq / self.sample_rate
            self.osc_time += 1 / self.sample_rate

            frame_array[i][:] = amp * math.sin(self.phase)

        return frame_array


@Fadeable
class CustomSawtoothOsc(CustomOsc):
    def __init__(self, param_callback, leading_edge=False, sample_rate=44100):
        super().__init__(sample_rate)
        self.param_callback = param_callback
        self.leading_edge = leading_edge

    def play_frames(self, num_frames):
        frame_array = np.zeros((num_frames, 2), dtype='d')

        phase = 0
        for i in range(num_frames):
            freq, amp = self.param_callback(i / self.sample_rate)
            phase += freq / self.sample_rate
            phase -= int(phase)

            frame_array[i][:] = amp * ((1 - phase) if self.leading_edge else phase)

        return frame_array
