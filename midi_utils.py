from abc import ABC
from enum import IntEnum
from typing import List, Optional, Iterable, Tuple, Dict

import numpy as np
import sounddevice as sd
from pygame import midi

from utils import CustomOsc, WavNPWriter


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

    class CONTROL_CHANGE_TYPE(IntEnum):
        TEMPO_CHANGE = 21
        SWING_CHANGE = 22
        GATE_CHANGE = 23
        MUTATE_CHANGE = 24
        DEVIATE_CHANGE = 25
        SUSTAIN_CHANGE = 64

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


class ControlChangeMessage(MIDIMessage):
    def __init__(self, change_type: MIDIMessage.CONTROL_CHANGE_TYPE, data_byte_2: int):
        super().__init__(self.STATUS_MESSAGE_TYPE.CONTROL_CHANGE)
        self.control_change_type = change_type
        self.data_byte_2 = data_byte_2


class SustainStartMessage(ControlChangeMessage):
    pass


class SustainEndMessage(ControlChangeMessage):
    pass


def decode_message(data_bytes: List[int]) -> MIDIMessage:
    MSG_ENUM = MIDIMessage.STATUS_MESSAGE_TYPE
    msg_type = MSG_ENUM(data_bytes[0] >> 4)
    if msg_type == MSG_ENUM.KEY_ON and data_bytes[2] > 0:
        return KeyOnMessage(data_bytes[1], data_bytes[2] / 127)
    elif msg_type == MSG_ENUM.KEY_OFF or (msg_type == MSG_ENUM.KEY_ON and data_bytes[2] == 0):
        return KeyOffMessage(data_bytes[1], data_bytes[2] / 127)
    elif msg_type == MSG_ENUM.CONTROL_CHANGE:
        CHG_TYPE_ENUM = MIDIMessage.CONTROL_CHANGE_TYPE
        change_type = CHG_TYPE_ENUM(data_bytes[1])

        if change_type == CHG_TYPE_ENUM.SUSTAIN_CHANGE:
            return SustainEndMessage() if data_bytes[2] == 0 else SustainStartMessage()
        else:
            return ControlChangeMessage(change_type, data_bytes[2])
    else:
        return MIDIMessage(msg_type)

class VirtualKeyboard(midi.Input):
    def __init__(self):
        self.on_keys = set()
        self.base_key = 30

        self.device_id = -1

    def process_key(self, key: str):
        charmap = {'a': 0, 'w': 1,  's': 2,  'e': 3,  'd': 4, 'f': 5, 't': 6, 'g': 7, 'y': 8,
                            'h': 9, 'u': 10, 'j': 11, 'i': 12, 'k': 13, 'l': 14}
        fixed_key_charmap = {'z': 77 + 20, 'x': 78 + 20, 'c': 79 + 20, 'v': 80 + 20, 'b': 81 + 20, 'n': 82 + 20, 'm': 83 + 20, ',': 84 + 20, '.': 85 + 20, '/': 86 + 20, ';': 87 + 20, '\'': 88 + 20}
            
        if key in charmap or key in fixed_key_charmap:
            MSG_ENUM = MIDIMessage.STATUS_MESSAGE_TYPE
            adjusted_key = self.base_key + charmap[key] if key in charmap else fixed_key_charmap[key]
            if adjusted_key in self.on_keys:
                self.on_keys.remove(adjusted_key)
                return (bytes([MSG_ENUM.KEY_OFF.value << 4, adjusted_key, 127]), None)
                
            else:
                self.on_keys.add(adjusted_key)
                return (bytes([MSG_ENUM.KEY_ON.value << 4, adjusted_key, 127]), None)
        elif key == 'q':
            self.base_key -= 1
        elif key == 'p':
            self.base_key += 1


class WinVirtualKeyboard(VirtualKeyboard):
    def poll(self):
        import msvcrt
        return msvcrt.kbhit()
    
    def read(self, num_events: int):
        import msvcrt

        result = []
        while len(result) < num_events:
            key: bytes = msvcrt.getch()
            if this_msg := self.process_key(key.decode('utf8')):
                result.append(this_msg)

        return result
    
class CursesVirtualKeyboard(VirtualKeyboard):
    def __init__(self):
        super().__init__()

        import curses
        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.unread_keys = []

        import atexit
        atexit.register(lambda: curses.endwin())

    def poll(self):
        if len(self.unread_keys) > 0:
            return True

        import curses
        self.screen.nodelay(True)
        try:
            self.unread_keys.append(self.screen.getkey())
            return True
        except curses.error:
            return False
        
    def read(self, num_events):
        self.screen.nodelay(False)

        result = []
        while len(result) < num_events:
            if len(self.unread_keys) > 0:
                this_key = self.unread_keys.pop(0)
            else:
                this_key = self.screen.getkey()

            if this_msg := self.process_key(this_key):
                result.append(this_msg)

        return result

    

def get_piano_key_frequency(key_num: int):
    return 55 * (2 ** ((key_num - 21) / 12))


class CustomSynth(ABC):
    def __init__(self, inputs: Optional[Iterable[midi.Input]] = None, output: Optional[sd.OutputStream] = None,
                 sample_rate: int = 44100):
        midi.init()

        if inputs is None:
            dev_id = midi.get_default_input_id()

            if dev_id == -1:
                try:
                    import msvcrt
                    self.instruments = [WinVirtualKeyboard()]
                except ImportError:
                    self.instruments = [CursesVirtualKeyboard()]
                # raise Exception('Could not find any MIDI input devices.')
            else:
                self.instruments = [midi.Input(dev_id)]
        else:
            self.instrument = list(inputs)

        self.output_stream = output or sd.OutputStream(samplerate=sample_rate, latency=0.05)
        self.active_oscs: Dict[Tuple[int, int], List[CustomOsc]] = dict()
        self.record_stream: Optional[WavNPWriter] = None
        self.samples_recorded: Optional[int] = None
        self.record_paused: bool = False
        self.recovering_stream = False

    def on_key_on(self, instrument: midi.Input, event: KeyOnMessage, oscs: List[CustomOsc]):
        return NotImplemented

    def on_key_off(self, instrument: midi.Input, event: KeyOffMessage, oscs: List[CustomOsc]):
        oscs.clear()

    def on_control_change(self, instrument: midi.Input, event: ControlChangeMessage):
        pass

    def iter_oscs(self):
        for osc_list in self.active_oscs.values():
            for osc in osc_list:
                yield osc

    def update_output(self):
        for instrument in self.instruments:
            if instrument.poll():
                events = instrument.read(1)
                print(f'Received events: {events}')
                decoded_events = [decode_message(payload) for payload, _ in events]
                for event in decoded_events:
                    if isinstance(event, KeyOnMessage):
                        if not (instrument.device_id, event.key_num) in self.active_oscs:
                            self.active_oscs[(instrument.device_id, event.key_num)] = []

                        self.on_key_on(instrument, event, self.active_oscs[(instrument.device_id, event.key_num)])
                        print(f'Playing key {get_piano_key_frequency(event.key_num)}')
                    elif isinstance(event, KeyOffMessage):
                        self.on_key_off(instrument, event, self.active_oscs[(instrument.device_id, event.key_num)])
                    elif isinstance(event, ControlChangeMessage):
                        self.on_control_change(instrument, event)

        if not self.output_stream.active:
            self.output_stream.start()

        if self.output_stream.write_available > 0:
            accum_buf = np.zeros(shape=(self.output_stream.write_available, 2), dtype=np.double)
            for osc_list in self.active_oscs.values():
                osc_list = [osc for osc in osc_list if not osc.is_complete()]
                for osc in osc_list:
                    result: np.ndarray = osc.play_frames(accum_buf.shape[0])
                    if len(result.shape) == 1:
                        result = np.transpose(np.tile(result, (2, 1)))
                    accum_buf += result
                # print(accum_buf)

            # print(accum_buf[:10])
                    
            # print('foo')
            
            accum_buf = accum_buf.clip(min=-1, max=1)

            try:
                self.output_stream.write(accum_buf.astype(np.float32))
                self.recovering_stream = False
            except sd.PortAudioError as ex:
                if self.recovering_stream:
                    raise
                else:
                    self.recovering_stream = True
                    print('Attempting to recover from stream error...')
                    self.output_stream.stop()
                    # self.output_stream = None
                    self.output_stream = sd.OutputStream(samplerate=self.output_stream.samplerate, latency=self.output_stream.latency)

            if self.record_stream is not None and not self.record_paused:
                self.record_stream.write(accum_buf)
                self.samples_recorded += accum_buf.shape[0]

    def start_record(self, file_path: str, sample_width: int):
        self.record_stream = WavNPWriter(file_path, sample_width, self.output_stream.samplerate)
        self.samples_recorded = 0

    def pause_record(self):
        self.record_paused = True

    def resume_record(self):
        self.record_paused = False

    def stop_record(self):
        if self.record_stream is not None:
            self.record_stream.close()
            self.record_stream = None
            self.samples_recorded = None
        
        self.record_paused = False