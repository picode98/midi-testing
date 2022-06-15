import math

from utils import CustomSineOsc, write_wav_file

if __name__ == '__main__':
    test_sound = CustomSineOsc(lambda time: (400 + 50 * time, abs(math.sin(2 * math.pi * time)))).play(10)
    test_sound2 = CustomSineOsc(lambda time: (400 + 60 * (time - 0.8), abs(math.sin(2.4 * math.pi * (time - 0.8))))).play(10)
    mix = test_sound + test_sound2

    write_wav_file('test.wav', mix)
