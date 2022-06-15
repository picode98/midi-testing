from midi_utils import *
from utils import CustomSineOsc


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
    def on_key_on(self, instrument: midi.Input, event: MIDIMessage) -> List[CustomOsc]:
        return [CustomSineOsc(
                    lambda t, e=event: (get_piano_key_frequency(e.key_num) + 100 * t, e.velocity)
                )]


synth = TestSynth()

while True:
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
