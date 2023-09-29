import onnxruntime as ort
import numpy as np

model_path = "model.onnx"
sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

x = np.random.randn(1, 3, 224, 224).astype(np.float32)

res = sess.run([output_name], {input_name: x})
