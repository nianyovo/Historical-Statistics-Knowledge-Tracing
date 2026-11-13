#ifndef HSF_C_API_H_
#define HSF_C_API_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(HSF_BUILD_DLL)
#define HSF_API __declspec(dllexport)
#elif defined(_WIN32)
#define HSF_API __declspec(dllimport)
#else
#define HSF_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HsfInteraction {
  int32_t user_id;
  int32_t item_id;
  int32_t concept_id;
  int32_t correct;
  double timestamp_sec;
  double resp_time_sec;
} HsfInteraction;

typedef struct HsfConfig {
  size_t window_size;
  double prior_alpha;
  double user_alpha;
  int32_t has_real_timestamp;
  int32_t has_real_resp_time;
} HsfConfig;

/* All functions return 0 on success, -1 invalid input, -2 allocation failure,
 * -3 other failure. Offsets have num_seqs + 1 entries, start at 0 and end
 * at num_rows. Output capacity is measured in floats, not bytes.
 * Handles must come from hsf_create_engine and must not be shared across
 * concurrent calls. Buffer ownership remains with the caller. */
HSF_API int hsf_create_engine(const HsfConfig* config, void** out_handle);
HSF_API int hsf_destroy_engine(void* handle);
HSF_API int hsf_fit_flat(void* handle, const HsfInteraction* rows,
                         size_t num_rows, const int64_t* offsets,
                         size_t num_seqs);
HSF_API int hsf_transform_flat(void* handle, const HsfInteraction* rows,
                               size_t num_rows, const int64_t* offsets,
                               size_t num_seqs, size_t num_threads,
                               float* output, size_t output_size);
HSF_API int hsf_fit_transform_oof_flat(
    void* handle, const HsfInteraction* rows, size_t num_rows,
    const int64_t* offsets, size_t num_seqs, const int32_t* fold_ids,
    size_t num_folds, size_t num_threads, float* output, size_t output_size);

#ifdef __cplusplus
}
#endif
#endif
