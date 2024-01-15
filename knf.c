#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include "knf.h"

static int calc_first_sample(int index, int window_shift, int window_size)
{
    int midpoint_of_frame = window_shift * index + window_shift / 2;
    int beginning_of_frame = midpoint_of_frame - window_size / 2;
    return beginning_of_frame;
}

static int calc_num_frames(int num_samples, int window_shift, int window_size, int flush)
{
    int num_frames = (num_samples + (window_shift / 2)) / window_shift;

    if (flush)
        return num_frames;

    int end_sample_of_last_frame = calc_first_sample(num_frames - 1, window_shift, window_size) + window_size;

    while (num_frames > 0 && end_sample_of_last_frame > num_samples) {
        num_frames--;
        end_sample_of_last_frame -= window_shift;
    }

    return num_frames;
}

/*
 *
 * Window
 *
 */

typedef struct {
    float * scale;
} Window;

static void window_init(Window * s, int window_size)
{
    assert(s->scale = malloc(sizeof(float) * window_size));
#define M_2PI 6.283185307179586476925286766559005
    double a = M_2PI / (window_size - 1);
    for (int i = 0; i < window_size; i++)
        s->scale[i] = pow(0.5 - 0.5 * cos(a * i), 0.85);
}

static void window_deinit(Window * s)
{
    free(s->scale);
}

static void remove_dc_offset(float * waveform, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += waveform[i];
    float mean = sum / n;
    for (int i = 0; i < n; i++)
        waveform[i] -= mean;
}

static void preemphasize(float * waveform, int n, float preemph_coeff)
{
    for (int i = n - 1; i > 0; i--)
        waveform[i] -= preemph_coeff * waveform[i - 1];
    waveform[0] -= preemph_coeff * waveform[0];
}

static void window_extract(Window * s, int sample_offset, const float * wave, int wave_size,
                           int index, int window_shift, int window_size, float * window)
{
    assert(sample_offset >= 0 && wave_size != 0);

    int start_sample = calc_first_sample(index, window_shift, window_size);
    assert(sample_offset == 0 || start_sample >= sample_offset);

    int wave_start = start_sample - sample_offset;
    int wave_end = wave_start + window_size;

    if (wave_start >= 0 && wave_end <= wave_size) {
        memcpy(window, wave + wave_start, window_size * sizeof(float));
    } else {
        for (int i = 0; i < window_size; i++) {
            int j = wave_start + i;
            while (j < 0 || j >= wave_size)
                j = (j < 0 ? 0 : 2 * wave_size) - j - 1;
            window[i] = wave[j];
        }
    }

    remove_dc_offset(window, window_size);

    preemphasize(window, window_size, 0.97f);

    for (int k = 0; k < window_size; k++)
        window[k] *= s->scale[k];
}

/*
 *
 * Fbank
 *
 */

#define NB_MEL_BINS 80

typedef struct {
    int size;
    float * in;
    fftwf_complex * out;
    fftwf_plan plan;
    float * power;
    struct {
        int offset;
        float * scale;
        int size;
    } bins[NB_MEL_BINS];
} Fbank;

static inline float MelScale(float freq)
{
    return 1127.0f * logf(1.0f + freq / 700.0f);
}

static void mel_init(Fbank * s, float sample_freq, int window_length_padded)
{
    for (int bin = 0; bin < NB_MEL_BINS; bin++) {
        s->bins[bin].scale = NULL;
        s->bins[bin].size = 0;
    }

    int num_fft_bins = window_length_padded / 2;
    float fft_bin_width = sample_freq / window_length_padded;
    float mel_low_freq = MelScale(20.0f);
    float mel_high_freq = MelScale(0.5f * sample_freq - 400.0f);
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (NB_MEL_BINS + 1);

    float * this_bin;
    assert(this_bin = malloc(num_fft_bins * sizeof(float)));

    for (int bin = 0; bin < NB_MEL_BINS; bin++) {
        float left_mel = mel_low_freq + bin * mel_freq_delta;
        float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
        float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

        int first_index = -1, last_index = -1;
        for (int i = 0; i < num_fft_bins; i++) {
            float freq = fft_bin_width * i;
            float mel = MelScale(freq);
            if (mel > left_mel && mel < right_mel) {
                if (mel <= center_mel)
                    this_bin[i] = (mel - left_mel) / (center_mel - left_mel);
                else
                    this_bin[i] = (right_mel - mel) / (right_mel - center_mel);
                if (first_index == -1)
                    first_index = i;
                last_index = i;
            }
        }

        assert(first_index != -1 && last_index >= first_index);

        s->bins[bin].offset = first_index;
        int size = last_index + 1 - first_index;

        assert(s->bins[bin].scale = realloc(s->bins[bin].scale, (s->bins[bin].size + size) * sizeof(float)));

        memcpy(s->bins[bin].scale + s->bins[bin].size, this_bin + first_index, size * sizeof(float));
        s->bins[bin].size += size;
    }

    free(this_bin);
}

static int fbank_init(Fbank * s, int sample_rate, int size)
{
    s->size = size;
    s->in = fftwf_malloc(sizeof(float) * size);
    s->out = fftwf_malloc(sizeof(fftwf_complex) * (size / 2 + 1));
    s->plan = fftwf_plan_dft_r2c_1d(size, s->in, s->out, FFTW_ESTIMATE);
    s->power = malloc(sizeof(float) * (size / 2 + 1));
    mel_init(s, sample_rate, size);
    return 0;
}

static void fbank_deinit(Fbank * s)
{
    fftwf_free(s->in);
    fftwf_free(s->out);
    fftwf_destroy_plan(s->plan);
    free(s->power);
    for (int i = 0; i < NB_MEL_BINS; i++)
        if (s->bins[i].scale)
            free(s->bins[i].scale);
}

static void fbank_compute(Fbank * s, const float * samples, float * feature)
{
    memcpy(s->in, samples, sizeof(float) * s->size);
    fftwf_execute(s->plan);

    s->power[0] = s->out[0][0] * s->out[0][0];
    int half_size = s->size / 2;
    for (int i = 1; i < half_size; i++) {
        float re = s->out[i][0];
        float im = s->out[i][1];
        s->power[i] = re * re + im * im;
    }
    s->power[half_size] = s->out[half_size][0] * s->out[half_size][0];

    for (int i = 0; i < NB_MEL_BINS; i++) {
        int offset = s->bins[i].offset;
        const float * scale = s->bins[i].scale;

        float energy = 0;
        for (int k = 0; k < s->bins[i].size; k++)
            energy += scale[k] * s->power[k + offset];

#define MAX(a, b) ((a)>(b)?(a):(b))
        feature[i] = log(MAX(energy, FLT_EPSILON));
    }
}

/*
 *
 * Frame
 *
 */

typedef struct Frame Frame;

struct Frame {
    float feature[NB_MEL_BINS];
    Frame * next;
};

typedef struct {
    Frame * root;
    int base_index;
    Frame * last;
} Frames;

static Frame * frame_alloc()
{
    Frame * frame = malloc(sizeof(Frame));
    assert(frame);
    return frame;
}

static void frames_init(Frames * s)
{
    s->root = NULL;
    s->base_index = 0;
    s->last = NULL;
}

static void frames_deinit(Frames * s)
{
    while (s->root) {
        Frame * frame = s->root;
        s->root = frame->next;
        free(frame);
    }
}

static int frames_size(const Frames * s)
{
    Frame * root = s->root;
    int size;
    for (size = s->base_index; root; root = root->next, size++) ;
    return size;
}

static void frames_push(Frames * s, Frame * frame)
{
    frame->next = NULL;
    if (!s->root)
        s->root = frame;
    else
        s->last->next = frame;
    s->last = frame;
}

static void frames_pop(Frames * s, int nframes)
{
    for (int i = 0; i < nframes && s->root; i++) {
        Frame * frame = s->root;
        s->root = frame->next;
        free(frame);
        s->base_index ++;
     }
}

static Frame * frames_get(const Frames * s, int index)
{
    Frame * root = s->root;
    for (int i = s->base_index; i < index && root; i++, root = root->next) ;
    assert(root);
    return root;
}

/*
 *
 * KNF
 *
 */

struct KNF {
    int sample_rate;
    int window_shift;
    int window_size;
    int window_size_padded;

    Frames frames;
    Window window;
    Fbank fbank;

    int input_finished;
    int waveform_offset;
    float * waveform_remainder;
    int waveform_remainder_size;

    float * window_buffer;
};

static int log2_int(unsigned int x)
{
    int y = 0;
    while (x >>= 1) y++;
    return y;
}

static inline int ceil_log2_int(unsigned int x)
{
    return log2_int((x - 1) << 1);
}

KNF * knf_create(int sample_rate, int feature_dim)
{
    KNF * s = malloc(sizeof(KNF));

    s->sample_rate = sample_rate;
    s->window_shift = sample_rate * 0.010f;
    s->window_size = sample_rate * 0.025f;
    s->window_size_padded = 1 << ceil_log2_int(s->window_size);
    assert(feature_dim == NB_MEL_BINS);

    frames_init(&s->frames);
    fbank_init(&s->fbank, s->sample_rate, s->window_size_padded);
    window_init(&s->window, s->window_size);

    s->input_finished = 0;
    s->waveform_offset = 0;
    s->waveform_remainder = NULL;
    s->waveform_remainder_size = 0;

    assert(s->window_buffer = malloc(s->window_size_padded * sizeof(float)));
    return s;
}

void knf_destroy(KNF * s)
{
    frames_deinit(&s->frames);
    window_deinit(&s->window);
    fbank_deinit(&s->fbank);
    if (s->waveform_remainder)
        free(s->waveform_remainder);
    free(s->window_buffer);
    free(s);
}

static void compute_features(KNF * s)
{
    int num_frames_old = frames_size(&s->frames);
    int num_samples_total = s->waveform_offset + s->waveform_remainder_size;
    int num_frames_new = calc_num_frames(num_samples_total, s->window_shift, s->window_size, s->input_finished);

    assert(num_frames_new >= num_frames_old);

    for (int index = num_frames_old; index < num_frames_new; index++) {
        window_extract(&s->window, s->waveform_offset, s->waveform_remainder, s->waveform_remainder_size,
                    index, s->window_shift, s->window_size, s->window_buffer);
        memset(s->window_buffer + s->window_size, 0, (s->window_size_padded - s->window_size) * sizeof(float));

        Frame * frame = frame_alloc();
        fbank_compute(&s->fbank, s->window_buffer, frame->feature);

        frames_push(&s->frames, frame);
    }

    int first_sample_of_next_frame = calc_first_sample(num_frames_new, s->window_shift, s->window_size);
    int samples_to_discard = first_sample_of_next_frame - s->waveform_offset;

    if (samples_to_discard > 0) {
        int new_num_samples = s->waveform_remainder_size - samples_to_discard;

        if (new_num_samples <= 0) {
            s->waveform_offset += s->waveform_remainder_size;

            free(s->waveform_remainder);
            s->waveform_remainder = NULL;
            s->waveform_remainder_size = 0;
        } else {
            float * new_remainder = malloc(new_num_samples * sizeof(float));
            assert(new_remainder);

            memcpy(new_remainder, s->waveform_remainder + samples_to_discard, new_num_samples * sizeof(float));
            s->waveform_offset += samples_to_discard;

            free(s->waveform_remainder);
            s->waveform_remainder = new_remainder;
            s->waveform_remainder_size = new_num_samples;
        }
    }
}

void knf_accept_waveform(KNF * s, int sample_rate, const float * samples, int n)
{
    assert(sample_rate == s->sample_rate);

    assert(s->waveform_remainder = realloc(s->waveform_remainder, (s->waveform_remainder_size + n) * sizeof(float)));
    memcpy(s->waveform_remainder + s->waveform_remainder_size, samples, n * sizeof(float));
    s->waveform_remainder_size += n;

    compute_features(s);
}

void knf_input_finished(KNF * s)
{
    s->input_finished = 1;
    compute_features(s);
}

int knf_num_frames_ready(KNF * s)
{
    return frames_size(&s->frames);
}

const float * knf_get_frame(KNF * s, int index)
{
    return frames_get(&s->frames, index)->feature;
}

int knf_is_input_finished(KNF * s)
{
    return s->input_finished;
}

void knf_pop(KNF * s, int n)
{
    frames_pop(&s->frames, n);
}
