#ifndef _FAN_PROJ_KERNEL
#define _FAN_PROJ_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int ProjectionForwardLaucher(
    const float* bottom_data, const int batch_size, const int height,
    const int width, const int channels, const int output_height,
    const int output_width, float* top_data, cudaStream_t stream);


int ProjectionBackwardLaucher(const float* top_diff, const int batch_size,
    const int height, const int width, const int channels, const int output_height,
    const int output_width, float* bottom_diff, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif