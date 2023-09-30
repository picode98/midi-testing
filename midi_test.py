import math

import pygame

from midi_utils import *
from utils import CustomPowerWaveOsc, CustomSineOsc, MouseVelocityInput


# midi.init()
#
# dev_id = midi.get_default_input_id()
#
# if dev_id == -1:
#     raise Exception('Could not find any MIDI input devices.')
#
# instrument = midi.Input(dev_id)
#
# active_notes: Dict[int, CustomSineOsc] = {}
# interval = 0.01


# def output_callback(outdata: np.ndarray, frames: int,
#          time, status: sd.CallbackFlags) -> None:
#     accum_buf = np.zeros(shape=(frames, 2), dtype=np.double)
#     for osc in active_notes.values():
#         accum_buf += osc.play_frames(accum_buf.shape[0])
#         print(accum_buf)
#
#     outdata[:] = accum_buf.clip(min=-1, max=1).astype(np.float32)

class TestSynth(CustomSynth):
    def __init__(self):
        super().__init__()
        self.effect_on = False
        self.mouse_in = MouseVelocityInput()

    def on_key_on(self, instrument: midi.Input, event: KeyOnMessage, oscs: List[CustomOsc]):
        mouse_in = self.mouse_in
        class TestSynthOsc1(CustomSineOsc):
            def get_amplitudes(self, time_vector):
                return event.velocity / 10.0

            def get_frequencies(self, time_vector):
                return get_piano_key_frequency(event.key_num)

        class TestSynthOsc2(CustomSineOsc):
            def get_amplitudes(self, time_vector):
                return event.velocity / 10.0

            def get_frequencies(self, time_vector):
                return get_piano_key_frequency(event.key_num) + (mouse_in.current_vel / 50.0)

        oscs += [TestSynthOsc1(), TestSynthOsc2()]

        # oscs += [CustomPowerWaveOsc(
        #                 lambda t, e=event: (get_piano_key_frequency(e.key_num), e.velocity / 10.0), power=4.0),
        #          CustomPowerWaveOsc(
        #                 lambda t, e=event: (get_piano_key_frequency(e.key_num) + 2.0, e.velocity / 10.0 if self.effect_on else 0.0), power=4.0)]

    def on_key_off(self, instrument: midi.Input, event: MIDIMessage, oscs: List[CustomPowerWaveOsc]):
        for osc in oscs:
            osc.fade_rate = 5.0

    def on_control_change(self, instrument: midi.Input, event: ControlChangeMessage):
        if isinstance(event, SustainStartMessage):
            self.effect_on = True
        elif isinstance(event, SustainEndMessage):
            self.effect_on = False


pygame.init()
pygame.display.set_mode()
synth = TestSynth()

while True:
    pygame.event.pump()
    synth.mouse_in.update()
    synth.update_output()

# with sd.OutputStream(samplerate=44100) as out_stream:
#     while True:
#         if instrument.poll():
#             events = instrument.read(1)
#             print(f'Received events: {events}')
#             decoded_events = [midi_utils.decode_message(payload) for payload, _ in events]
#             for event in decoded_events:
#                 if isinstance(event, midi_utils.KeyOnMessage):
#                     active_notes[event.key_num] = CustomSineOsc(
#                         lambda t, e=event: (midi_utils.get_piano_key_frequency(e.key_num) + 100 * t, e.velocity)
#                     )
#                     print(f'Playing key {midi_utils.get_piano_key_frequency(event.key_num)}')
#                 elif isinstance(event, midi_utils.KeyOffMessage):
#                     del active_notes[event.key_num]
#
#         if out_stream.write_available > 0:
#             accum_buf = np.zeros(shape=(out_stream.write_available, 2), dtype=np.double)
#             for osc in active_notes.values():
#                 accum_buf += osc.play_frames(accum_buf.shape[0])
#                 # print(accum_buf)
#
#             out_stream.write(accum_buf.clip(min=-1, max=1).astype(np.float32))
