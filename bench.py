#!/usr/bin/env python3
import kaldifeat
import kaldi_native_fbank as knf
from ctypes import *
import time
import torch

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
    samples = torch.randn(16000 * 1000)
    samples.numpy()

    # kaldi feat

    opts = kaldifeat.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False

    online_fbank = kaldifeat.OnlineFbank(opts)
    start = time.time()
    online_fbank.accept_waveform(sampling_rate, samples)
    end = time.time()
    print("kalifeat", end - start)

    # knf

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False

    fbank = knf.OnlineFbank(opts)
    samples_list = samples.tolist()
    start = time.time()
    fbank.accept_waveform(sampling_rate, samples_list)
    end = time.time()
    print("kali-native-feat", end - start)

    # knfc

    k = knfc.knf_create(sampling_rate, 80)
    start = time.time()
    samples_ptr = samples.numpy().ctypes.data_as(c_float_p)
    knfc.knf_accept_waveform(k, sampling_rate, samples_ptr, len(samples))
    end = time.time()
    print("knfc", end - start)

if __name__ == "__main__":
    torch.manual_seed(20220825)
    main()

