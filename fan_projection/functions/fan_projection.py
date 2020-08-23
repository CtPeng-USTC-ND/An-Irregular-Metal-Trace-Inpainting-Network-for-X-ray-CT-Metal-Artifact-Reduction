import torch
from torch.autograd import Function
from .._ext import fan_projection

class FanProjFunction(Function):
    def __init__(self, out_height, out_width):
        super(FanProjFunction, self).__init__()
        self.out_height = int(out_height)
        self.out_width = int(out_width)
        self.feature_size = None
        
    def forward(self, input):
        self.feature_size = input.shape

#         self.save_for_backward(input)

        output = input.new(*self._output_size(input))

        if not input.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(input, torch.autograd.Variable):
                if not isinstance(input.data, torch.cuda.FloatTensor):
                    raise NotImplementedError
            else:
                if not isinstance(input, torch.cuda.FloatTensor):
                    raise NotImplementedError
            # weight out_channels, in_channels, out_shape
            fan_projection.fan_projection_forward_cuda(
                input, output)
        return output

    def backward(self, grad_output):
#         input = self.saved_tensors[0]

        grad_input = None

#         if not grad_output.is_cuda:
#             raise NotImplementedError
#         else:
#             if isinstance(grad_output, torch.autograd.Variable):
#                 if not isinstance(grad_output.data, torch.cuda.FloatTensor):
#                     raise NotImplementedError
#             else:
#                 if not isinstance(grad_output, torch.cuda.FloatTensor):
#                     raise NotImplementedError
                    
#             if self.needs_input_grad[0]:
#                 batch_size, num_channels, data_height, data_width = self.feature_size
#                 grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()
#                 fan_projection.fan_projection_backward_input_cuda(grad_output, grad_input)

        return grad_input
    
    def _output_size(self, input):

        output_size = (input.size(0), 1, self.out_height, self.out_width)

        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size