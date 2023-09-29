import torch
import torchvision

import onnxruntime as ort

model = torchvision.models.resnet18(pretrained=True)
x = torch.randn(1, 3, 224, 224)
model.eval()

# export the model to an ONNX file
torch.onnx.export(model, x, "model.onnx", export_params=True,
                  opset_version=10, do_constant_folding=True,
                  input_names = ['input'], output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})

