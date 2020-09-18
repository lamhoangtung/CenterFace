import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('centerface.onnx')

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

# use model_simp as a standard ONNX model object
#with open('sim_onnx.onnx', 'wb') as f:
onnx.save_model(model_simp, 'sim_centerface.onnx')