
import tensorboardX 
import torch
from torchvision.models import resnet34
import torch.onnx
 
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) 
print(x.size(), y.size(), z.size())