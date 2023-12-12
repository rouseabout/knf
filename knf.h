#ifndef KNF_H
#define KNF_H

typedef struct KNF KNF;

#ifdef __cplusplus
extern "C" {
#endif

KNF * knf_create(int sample_rate, int feature_dim);
void knf_destroy(KNF *);
void knf_accept_waveform(KNF * s, int sample_rate, const float * samples, int count);
void knf_input_finished(KNF * s);
int knf_num_frames_ready(KNF * s);
const float * knf_get_frame(KNF * s, int index);
int knf_is_input_finished(KNF * s);
void knf_pop(KNF * s, int n);

#ifdef __cplusplus
}
#endif

#endif /* KNF_H */
