#!/usr/bin/env python3
import kaldi_native_fbank as knf
import torch
import numpy as np
from ctypes import *

c_float_p = POINTER(c_float)

knfc = simutil = CDLL("libknf.so")
knfc.knf_create.argtypes = [c_int, c_int]
knfc.knf_create.restype = c_void_p
knfc.knf_accept_waveform.argtypes = [c_void_p, c_int, c_float_p, c_int]
knfc.knf_num_frames_ready.argtypes = [c_void_p]
knfc.knf_get_frame.argtypes = [c_void_p, c_int]
knfc.knf_get_frame.restype = c_float_p

def main():
    sampling_rate = 16000
    samples = torch.randn(16000 * 10)

    k = knfc.knf_create(sampling_rate, 80)
    knfc.knf_accept_waveform(k, sampling_rate, samples.numpy().ctypes.data_as(c_float_p), len(samples))

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sampling_rate, samples.tolist())

    assert knfc.knf_num_frames_ready(k) == fbank.num_frames_ready
    for i in range(fbank.num_frames_ready):
        f1 = np.ctypeslib.as_array(knfc.knf_get_frame(k, i), shape=(80,))
        f2 = fbank.get_frame(i)
        assert np.allclose(f1, f2, atol=1e-3)

if __name__ == "__main__":
    torch.manual_seed(20220825)
    main()
    print("success")
