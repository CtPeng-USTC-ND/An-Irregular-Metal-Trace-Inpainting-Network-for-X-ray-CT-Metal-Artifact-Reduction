#include <THC/THC.h>
#include <math.h>
#include "cuda/fan_projection_kernel.h"

extern THCState *state;

int fan_projection_forward_cuda(THCudaTensor *features, THCudaTensor *output)
{
    // Grab the input tensor
    float * data_flat = THCudaTensor_data(state, features);
    float * output_flat = THCudaTensor_data(state, output);
    
    int batch_size = THCudaTensor_size(state, features, 0);
    
    // data height
    int data_height = THCudaTensor_size(state, features, 2);
    // data width
    int data_width = THCudaTensor_size(state, features, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, features, 1);
    
    // output height
    int output_height = THCudaTensor_size(state, output, 2);
    // output width
    int output_width = THCudaTensor_size(state, output, 3);
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    
    ProjectionForwardLaucher(
        data_flat, batch_size, data_height,
        data_width, num_channels, output_height,
        output_width, output_flat, stream);

    return 1;
}


int fan_projection_backward_input_cuda(THCudaTensor *gradOutput,
    THCudaTensor *gradInput)
{
    // Grab the input tensor
    float * top_grad_flat = THCudaTensor_data(state, gradOutput);

    float * bottom_grad_flat = THCudaTensor_data(state, gradInput);

    // batch size
    int batch_size = THCudaTensor_size(state, gradInput, 0);
    
    // data height
    int data_height = THCudaTensor_size(state, gradInput, 2);
    // data width
    int data_width = THCudaTensor_size(state, gradInput, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, gradInput, 1);

    // output height
    int output_height = THCudaTensor_size(state, gradOutput, 2);
    // output width
    int output_width = THCudaTensor_size(state, gradOutput, 3);
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    ProjectionBackwardLaucher(
        top_grad_flat, batch_size, data_height,
        data_width, num_channels, output_height,
        output_width, bottom_grad_flat, stream);

    return 1;
}