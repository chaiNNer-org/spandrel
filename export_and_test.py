from libs.spandrel.spandrel import ModelLoader
import torch
import onnx
import onnxruntime as ort
import sys
import os
import cv2
import pnnx
import cv2
import numpy as np
import os
import ncnn
cwd = os.getcwd()

def fixNCNNParamInput(paramFile):
        """
        replaces in0 with data and out0 with output in a ncnn param file
        """
        with open(paramFile, "r") as f:
            lines = f.readlines()

        with open(paramFile, "w") as f:
            for line in lines:
                line = line.replace("in0", "data")
                line = line.replace("out0", "output")
                f.write(line)
class UpscaleNCNNImage:
    def __init__(
        self,
        modelPath: str = "models",
        modelName: str = "",
        vulkan: bool = True,
        tile_pad=10,
    ):
        self.modelPath = modelPath
        self.modelName = modelName
        self.vulkan = vulkan
        self.fullModelPath = os.path.join(self.modelPath, self.modelName)

    def renderImage(self, fullImagePath) -> np.array:
        net = ncnn.Net()

        # Use vulkan compute
        net.opt.use_vulkan_compute = self.vulkan

        # Load model param and bin
        net.load_param(self.fullModelPath + ".param")
        net.load_model(self.fullModelPath + ".bin")

        ex = net.create_extractor()

        # Load image using opencv
        img = cv2.imread(fullImagePath)

        # Convert image to ncnn Mat
        mat_in = ncnn.Mat.from_pixels(
            img, ncnn.Mat.PixelType.PIXEL_BGR, img.shape[1], img.shape[0]
        )

        # Normalize image (required)
        # Note that passing in a normalized numpy array will not work.
        mean_vals = []
        norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)

        # Try/except block to catch out-of-memory error
        try:
            # Make sure the input and output names match the param file
            ex.input("in0", mat_in)
            ret, mat_out = ex.extract("out0")
            out = np.array(mat_out)

            # Transpose the output from `c, h, w` to `h, w, c` and put it back in 0-255 range
            output = out.transpose(1, 2, 0) * 255

            return output
        except:
            ncnn.destroy_gpu_instance()

    def saveImage(self, imageNPArray, outputPath: str):
        cv2.imwrite(filename=outputPath, img=imageNPArray)

# load a model from disk
try:
    model_str = sys.argv[1]
except:
    model_str = input("Paste model path/name here: ")

if "https://" in model_str:
    from urllib.request import urlretrieve
    model_base_name = model_str.split("/")[-1]
    urlretrieve(model_str, model_base_name)
    model_str = model_base_name


conversion_path = "converted_models/"
s = torch.load(model_str, map_location="cpu")  # Ensure the model can be loaded
model = ModelLoader().load_from_file(model_str)
if not os.path.exists(conversion_path): os.mkdir(conversion_path)
model_name = os.path.basename(model_str)
scale = model.scale
model = model.model
state_dict = model.state_dict()
model.eval()
model.load_state_dict(state_dict, strict=True)
pt_path = os.path.join(f'{conversion_path}',f'{scale}x_{model_name}.pt')
print("Exporting...")
with torch.inference_mode():
    ex_input = torch.rand(1, 3, 32, 32)
    mod = torch.jit.trace( model,
        ex_input)
    mod.save(pt_path)
    torch.onnx.export(
        model,
        ex_input,
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
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.
with torch.no_grad():
    output = model(image)
output = output.squeeze(0).permute(1, 2, 0).numpy() * 255.
output = output.clip(0, 255).astype('uint8')
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
ort_output = ort_outs[0].squeeze(0).transpose(1, 2, 0) * 255.
ort_output = cv2.cvtColor(ort_output, cv2.COLOR_RGB2BGR)
cv2.imwrite(f"test_images/{model_name}_onnx_output.png", ort_output)
print("ONNX inference completed and output saved.")

print("Exporting ncnn model")
"""
Takes in a pytorch model, and uses JIT tracing with PNNX to convert it to ncnn.
This method removed unnecessary files, and fixes the param file to be compadible with most NCNN appliacitons.
"""


pnnxBinLocation = model_str + ".pnnx.bin"
pnnxParamLocation = model_str + ".pnnx.param"
pnnxPythonLocation = model_str + "_pnnx.py"
pnnxOnnxLocation = model_str + ".pnnx.onnx"
ncnnPythonLocation = model_str + "_ncnn.py"
ncnnParamLocation = model_str +  ".ncnn.param"
ncnnBinLocation = model_str +  ".ncnn.bin"

# pnnx gives out a lot of weird errors, so i will be try/excepting this.
# usually nothing goes wrong, but it cant take in the pnnxbin/pnnxparam location on windows.

model = pnnx.convert(
    ptpath=pt_path,
    inputs=ex_input,
    optlevel=2,
    fp16=False,
    pnnxbin=pnnxBinLocation,
    pnnxparam=pnnxParamLocation,
    pnnxpy=pnnxPythonLocation,
    pnnxonnx=pnnxOnnxLocation,
    ncnnpy=ncnnPythonLocation,
    ncnnbin=ncnnBinLocation,
    ncnnparam=ncnnParamLocation,
)


# remove stuff that we dont need
try:
    os.remove(pnnxBinLocation)
    os.remove(pnnxParamLocation)
    os.remove(pnnxPythonLocation)
    os.remove(pnnxOnnxLocation)
    os.remove(ncnnPythonLocation)
except Exception as e:
    pass
try:
    os.remove(os.path.join(cwd, "debug.bin"))
    os.remove(os.path.join(cwd, "debug.param"))
    os.remove(os.path.join(cwd, "debug2.bin"))
    os.remove(os.path.join(cwd, "debug2.param"))
except Exception as e:
    pass

image = cv2.imread("test_images/image.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.
ncnn_model = UpscaleNCNNImage(
    modelPath=cwd,
    modelName=f"{model_str}.ncnn",
)
output = ncnn_model.renderImage("test_images/image.png")
output_path = f"test_images/{model_name}_ncnn_output.png"
ncnn_model.saveImage(output, output_path)
