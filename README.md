Plain C implementataion of the Kaldi online feature extractor using [FFTW](https://www.fftw.org/).

Produces identical output to [kaldifeat](https://github.com/csukuangfj/kaldifeat) and [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank).

## Benchmark

See `bench.py`. Wall-clock time to compute features for 1000 second waveform:

Implementation | Time (Seconds)
---|---
kaldifeat | 10.6698
kali-native-fbank | 0.7778
knf | 0.3258
