#!/usr/bin/env python
import sys
import array
import math
import wave

import matplotlib.pyplot as plt
import numpy
import pywt
from scipy import signal
import click


class BPMDetector:

    def __init__(self):
        pass

    def read_wav(self, filename):
        # open file, get metadata for audio
        try:
            wf = wave.open(filename, "rb")
        except IOError as e:
            print(e)
            return

        # typ = choose_type( wf.getsampwidth() ) # TODO: implement choose_type
        nsamps = wf.getnframes()
        assert nsamps > 0

        fs = wf.getframerate()
        assert fs > 0

        # Read entire file and make into an array
        samps = list(array.array("i", wf.readframes(nsamps)))

        try:
            assert nsamps == len(samps)
        except AssertionError:
            print(nsamps, "not equal to", len(samps))

        return samps, fs

    # print an error when no data can be found
    def no_audio_data(self):
        print("No audio data for sample, skipping...")
        return None, None

    # simple peak detection
    def peak_detect(self, data):
        max_val = numpy.amax(abs(data))
        peak_ndx = numpy.where(data == max_val)
        if len(peak_ndx[0]) == 0:  # if nothing found then the max must be negative
            peak_ndx = numpy.where(data == -max_val)
        return peak_ndx

    def process_data(self, data, fs):
        cA = []
        cD = []
        correl = []
        cD_sum = []
        levels = 4
        max_decimation = 2**(levels - 1)
        min_ndx = math.floor(60.0 / 220 * (fs / max_decimation))
        max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))

        for loop in range(0, levels):
            cD = []
            # 1) DWT
            if loop == 0:
                [cA, cD] = pywt.dwt(data, "db4")
                cD_minlen = len(cD) / max_decimation + 1
                cD_sum = numpy.zeros(math.floor(cD_minlen))
            else:
                [cA, cD] = pywt.dwt(cA, "db4")

            # 2) Filter
            cD = signal.lfilter([0.01], [1 - 0.99], cD)

            # 4) Subtract out the mean.

            # 5) Decimate for reconstruction later.
            cD = abs(cD[::(2**(levels - loop - 1))])
            cD = cD - numpy.mean(cD)

            # 6) Recombine the signal before ACF
            #    Essentially, each level the detail coefs (i.e. the HPF values) are concatenated to the beginning of the array
            cD_sum = cD[0:math.floor(cD_minlen)] + cD_sum

        if [b for b in cA if b != 0.0] == []:
            return self.no_audio_data()

        # Adding in the approximate data as well...
        cA = signal.lfilter([0.01], [1 - 0.99], cA)
        cA = abs(cA)
        cA = cA - numpy.mean(cA)
        cD_sum = cA[0:math.floor(cD_minlen)] + cD_sum

        # ACF
        correl = numpy.correlate(cD_sum, cD_sum, "full")

        midpoint = math.floor(len(correl) / 2)
        correl_midpoint_tmp = correl[midpoint:]
        peak_ndx = self.peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])
        if len(peak_ndx) > 1:
            return self.no_audio_data()

        peak_ndx_adjusted = peak_ndx[0] + min_ndx
        bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)
        print(bpm)
        return bpm, correl

    def detect(self, filename, window=3):
        samps, fs = self.read_wav(filename)
        data = []
        correl = []
        bpm = 0
        n = 0
        nsamps = len(samps)
        window_samps = int(window * fs)
        samps_ndx = 0
        max_window_ndx = math.floor(nsamps / window_samps)
        bpms = numpy.zeros(max_window_ndx)

        # Iterate through all windows
        for window_ndx in range(0, max_window_ndx):
            data = samps[samps_ndx:samps_ndx + window_samps]
            if not ((len(data) % window_samps) == 0):
                raise AssertionError(str(len(data)))

            bpm, correl_temp = self.process_data(data, fs)
            if bpm is None:
                continue
            bpms[window_ndx] = bpm[0]
            correl = correl_temp

            # Iterate at the end of the loop
            samps_ndx = samps_ndx + window_samps
            n = n + 1


#        n = range(0, len(correl))
#        plt.plot(n, abs(correl))
#        plt.show(block=True)

        return numpy.median(bpms)


@click.command()
@click.option("-w", "--window", default=3, is_flag=False)
@click.argument("filename")
def detect(window, filename):
    bpm = BPMDetector()
    result = bpm.detect(filename, window)
    print("estimated Beats Per Minute: %.1f" % result)


if __name__ == "__main__":
    detect()
