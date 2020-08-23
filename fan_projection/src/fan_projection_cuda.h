int fan_projection_forward_cuda(THCudaTensor *input, THCudaTensor *output);

int fan_projection_backward_input_cuda(THCudaTensor *gradOutput,
    THCudaTensor *gradInput);