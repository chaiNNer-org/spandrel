from libs.spandrel.spandrel import ModelLoader
import torch
import onnx
import onnxruntime as ort
import sys
import os
import cv2
# load a model from disk
try:
    model_str = sys.argv[1]
except:
    model_str = input("Paste model path/name here: ")
conversion_path = "converted_models/"
model = ModelLoader().load_from_file(model_str)
if not os.path.exists(conversion_path): os.mkdir(conversion_path)
model_name = os.path.basename(model_str)
scale = model.scale
model = model.model
state_dict = model.state_dict()
model.eval()
model.load_state_dict(state_dict, strict=True)
print("Exporting...")
with torch.inference_mode():
    mod = torch.jit.trace( model,
        torch.rand(1, 3, 32, 32))
    mod.save(os.path.join(f'{conversion_path}',f'{scale}x_{model_name}.pt'))
    torch.onnx.export(
        model,
        torch.rand(1, 3, 32, 32),
        os.path.join(f'{conversion_path}',f'{scale}x_{model_name}.onnx'),
        verbose=False,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
                "input": {0: "batch_size", 2: "width", 3: "height"},
                "output": {0: "batch_size", 2: "width", 3: "height"},
            }
    )
image = cv2.imread("test_images/image.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
with torch.no_grad():
    output = model(image)
output = output.squeeze(0).permute(1, 2, 0).numpy()
output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
cv2.imwrite(f"test_images/{model_name}_output.png", output)

onnx_model = onnx.load(os.path.join(conversion_path, f'{scale}x_{model_name}.onnx'))
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")
ort_session = ort.InferenceSession(os.path.join(conversion_path, f'{scale}x_{model_name}.onnx'))
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
ort_inputs = {input_name: image.numpy()}
ort_outs = ort_session.run([output_name], ort_inputs)
ort_output = ort_outs[0].squeeze(0).transpose(1, 2, 0)
ort_output = cv2.cvtColor(ort_output, cv2.COLOR_RGB2BGR)
cv2.imwrite(f"test_images/{model_name}_onnx_output.png", ort_output)
