import torch
import torchvision
import onnx
import torch.onnx
from ssdmobilenetv1master import *

# ## Load PyTorch Model
model = create_mobilenetv1_ssd(91)
print(model)
inputX = torch.randn(1, 3, 300, 300)
output = model(inputX)
print(output)
# ## Create ONNX
torch.onnx.export(model, inputX, "mobilenet_ssd.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=["input"], output_names=["output"],dynamic_axes={"input" : {0: "batch_size"},"output" : {0: "batch_size"}})

# ## Verifying ONNX
onnx_model = onnx.load("./mobilenet_ssd.onnx")
onnx.checker.check_model(onnx_model)


