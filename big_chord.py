from math import atan

import numpy as np

from utils import CustomSineOsc, get_note_frequency, arctan_transition, fade_pulse_transition, write_wav_file

s_rate = 44100
frame_buf = np.zeros((5 * s_rate, 2))

for i in range(2, 5):
    this_c_osc = CustomSineOsc(lambda t: (get_note_frequency(i, 'C') * (arctan_transition(20 * (t - 2)) + 1), 0.5 / (t + 1)), s_rate)
    this_f_osc = CustomSineOsc(lambda t: (get_note_frequency(i, 'F') * (arctan_transition(20 * (t - 3.5)) + 1), 0.5 / (t + 1)), s_rate)

    frame_buf += this_c_osc.play(5)
    frame_buf += this_f_osc.play(5)

write_wav_file('test2.wav', frame_buf)

frame_buf2 = np.zeros((5 * s_rate, 2))

for i in range(2, 5):
    this_c_osc = CustomSineOsc(lambda t: (get_note_frequency(i, 'C'), 0.05 + 0.95 * fade_pulse_transition(t * 4, deadband=40)), s_rate)
    this_f_osc = CustomSineOsc(lambda t: (get_note_frequency(i, 'F'), 0.05 + 0.95 * fade_pulse_transition(t * 4, deadband=40)), s_rate)

    frame_buf2 += this_c_osc.play(5)
    frame_buf2 += this_f_osc.play(5)

write_wav_file('test3.wav', frame_buf2)
