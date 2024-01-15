#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "knf.h"

int main()
{
    knf::FbankOptions opts;
    opts.frame_opts.dither = 0;
    opts.frame_opts.snip_edges = false;
    opts.frame_opts.samp_freq = 16000;
    opts.mel_opts.num_bins = 80;
    opts.mel_opts.high_freq = -400;
    knf::OnlineGenericBaseFeature<knf::FbankComputer> k1(opts);

    KNF * k2 = knf_create(16000, 80);

    srand48(1337);
#define N (16000*1000)
    float * samples = (float *)malloc(N * sizeof(float));
    assert(samples);
    struct timespec start, end;

    for (int i = 0; i < 10; i++) {
        for (int i = 0; i < N; i++)
            samples[i] = drand48();

        clock_gettime(CLOCK_MONOTONIC, &start);
        k1.AcceptWaveform(16000, samples, N);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double start1 = start.tv_sec + start.tv_nsec/1000000000.0;
        double end1 = end.tv_sec + end.tv_nsec/1000000000.0;
 
        clock_gettime(CLOCK_MONOTONIC, &start);
        knf_accept_waveform(k2, 16000, samples, N);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double start2 = start.tv_sec + start.tv_nsec/1000000000.0;
        double end2 = end.tv_sec + end.tv_nsec/1000000000.0;

        printf("%f, %f\n", end1 - start1, end2 - start2);

        k1.Pop(N);
        knf_pop(k2, N);
    }

    knf_destroy(k2);
}
