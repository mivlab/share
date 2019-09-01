import torch
import cv2
from models import *
from torch.autograd import Variable
from torchvision import transforms, models

device = torch.device('cpu')
model = Darknet("config/yolov3-custom.cfg").to(device)
use_cuda = False
#if torch.cuda.is_available():
#    model.cuda()
model.load_state_dict(torch.load('checkpoints/yolov3_ckpt_6.pth', map_location=device))
model.eval()

# 用于导出C++版本的模型.
#traced_script_module = torch.jit.trace(model, torch.rand(1, 3, 28, 28))
#traced_script_module.save('params.pt')

#用于导出onnx的模型
dummy_input = torch.randn(1, 3, 416, 416, device='cpu')
torch.onnx.export(model, dummy_input, "yolov3.onnx",  export_params=True, verbose=True, training=False)
