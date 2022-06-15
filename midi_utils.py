from abc import ABC
from enum import IntEnum
from typing import List, Optional, Iterable, Tuple, Dict

import numpy as np
import sounddevice as sd
from pygame import midi

from utils import CustomOsc


class MIDIMessage:
    class STATUS_MESSAGE_TYPE(IntEnum):
        KEY_OFF = 8
        KEY_ON = 9
        POLYPHONIC_KEY_PRESSURE = 10
        CONTROL_CHANGE = 11
        PROGRAM_CHANGE = 12
        CHANNEL_PRESSURE = 13
        PITCH_WHEEL_CHANGE = 14
        SYSTEM_COMMON_MSG = 15

    def __init__(self, msg_type: STATUS_MESSAGE_TYPE):
        self.msg_type = msg_type


class KeyOnMessage(MIDIMessage):
    def __init__(self, key_num: int, velocity: float):
        super().__init__(self.STATUS_MESSAGE_TYPE.KEY_ON)
        self.key_num = key_num
        self.velocity = velocity


class KeyOffMessage(MIDIMessage):
    def __init__(self, key_num: int, velocity: float):
        super().__init__(self.STATUS_MESSAGE_TYPE.KEY_OFF)
        self.key_num = key_num
        self.velocity = velocity


def decode_message(data_bytes: List[int]) -> MIDIMessage:
    MSG_ENUM = MIDIMessage.STATUS_MESSAGE_TYPE
    msg_type = MSG_ENUM(data_bytes[0] >> 4)
    if msg_type == MSG_ENUM.KEY_ON:
        return KeyOnMessage(data_bytes[1], data_bytes[2] / 127)
    elif msg_type == MSG_ENUM.KEY_OFF:
        return KeyOffMessage(data_bytes[1], data_bytes[2] / 127)
    else:
        return MIDIMessage(msg_type)


def get_piano_key_frequency(key_num: int):
    return 55 * (2 ** ((key_num - 21) / 12))


class CustomSynth(ABC):
    def __init__(self, inputs: Optional[Iterable[midi.Input]] = None, output: Optional[sd.OutputStream] = None,
                 sample_rate: int = 44100):
        midi.init()

        if inputs is None:
            dev_id = midi.get_default_input_id()

            if dev_id == -1:
                raise Exception('Could not find any MIDI input devices.')

            self.instruments = [midi.Input(dev_id)]
        else:
            self.instrument = list(inputs)

        self.output_stream = output or sd.OutputStream(samplerate=sample_rate)
        self.active_oscs: Dict[Tuple[int, int], List[CustomOsc]] = dict()

    def on_key_on(self, instrument: midi.Input, event: MIDIMessage) -> List[CustomOsc]:
        return NotImplemented

    def on_key_off(self, instrument: midi.Input, event: MIDIMessage) -> List[CustomOsc]:
        return []

    def update_output(self):
        for instrument in self.instruments:
            if instrument.poll():
                events = instrument.read(1)
                print(f'Received events: {events}')
                decoded_events = [decode_message(payload) for payload, _ in events]
                for event in decoded_events:
                    if isinstance(event, KeyOnMessage):
                        self.active_oscs[(instrument.device_id, event.key_num)] = self.on_key_on(instrument, event)
                        print(f'Playing key {get_piano_key_frequency(event.key_num)}')
                    elif isinstance(event, KeyOffMessage):
                        self.active_oscs[(instrument.device_id, event.key_num)] = self.on_key_off(instrument, event)

        if not self.output_stream.active:
            self.output_stream.start()

        if self.output_stream.write_available > 0:
            accum_buf = np.zeros(shape=(self.output_stream.write_available, 2), dtype=np.double)
            for osc_list in self.active_oscs.values():
                for osc in osc_list:
                    accum_buf += osc.play_frames(accum_buf.shape[0])
                # print(accum_buf)

            # print(accum_buf[:10])
            self.output_stream.write(accum_buf.clip(min=-1, max=1).astype(np.float32))