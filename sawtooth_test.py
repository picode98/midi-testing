from utils import CustomSawtoothOsc, write_wav_file, time_shift

test1 = CustomSawtoothOsc(lambda t: (400 + 20 * int(t), 0.5)).play(5)
test1_shifted = time_shift(test1, 1 / 2000)
write_wav_file('sawtooth_test1.wav', test1 + test1_shifted)
