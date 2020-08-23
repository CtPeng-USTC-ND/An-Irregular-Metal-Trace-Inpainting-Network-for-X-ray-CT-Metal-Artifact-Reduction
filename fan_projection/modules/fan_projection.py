from torch.nn.modules.module import Module
from ..functions.fan_projection import FanProjFunction

class FanProj(Module):
    def __init__(self, out_height, out_width):
        super(FanProj, self).__init__()
        self.out_height = int(out_height)
        self.out_width = int(out_width)
        self.f = FanProjFunction(self.out_height, self.out_width)
        
    def forward(self, input):
        
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(
                    input.dim()))
        
        return self.f(input)