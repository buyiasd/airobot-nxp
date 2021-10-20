import torch
import torchvision
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--working_dir', default='./model')
parser.add_argument('--pth_file_name', default='best_model.pth')
args = parser.parse_args()

PTH_PATH = os.path.join(args.working_dir, args.pth_file_name)
ONNX_PATH = os.path.join(args.working_dir, 'airobot_alexnet_full.onnx')
TF_PATH = os.path.join(args.working_dir, 'airobot_alexnet_full')
TFLITE_QUANT_PATH = os.path.join(args.working_dir, '/airobot_alexnet_full_uint8.tflite')
TFLITE_PATH = os.path.join(args.working_dir, 'airobot_alexnet_full.tflite')

quant_learning_set = [np.random.rand(1, 3, 224, 224) for i in range(100)]

# Load pre-trained AlexNet model using CPU setting
model = torchvision.models.alexnet(pretrained=False)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Replace the final layer with a dummy layer for quantization
# If not needed, comment this line and uncomment the next line
# model.classifier[6] = Identity()
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)

updated_dict = torch.load(PTH_PATH, map_location='cpu')
model.load_state_dict(updated_dict)

print('PyTorch model loaded. \n')

dummy_input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))

torch.onnx.export(
    model=model,
    args=dummy_input,
    f=ONNX_PATH,
    verbose=False,
    export_params=True,
    do_constant_folding=False,
)

print('Conversion to ONNX done.\n')

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

tf_rep = prepare(onnx_model)
tf_rep.export_graph(TF_PATH)

print('Conversion to SavedModel done.\n')


def representative_dataset():
    for img in tf.data.Dataset.from_tensor_slices(quant_learning_set)\
            .batch(1).take(100):
        yield [tf.cast(img, tf.float32)]



## Saving original model final layer bias and weights
og_converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
print('SavedModel file loaded.')
tflite_model = og_converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f'Floating tflite model to {TFLITE_PATH}.\n')

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()
output_weight_index = tensor_details[9]['index']
output_bias_index = tensor_details[10]['index']

print(interpreter.tensor(output_weight_index)().shape)
print(interpreter.tensor(output_bias_index)().shape)
np.save(os.path.join(args.working_dir, '/npy/classifier_6_bias.npy'), interpreter.tensor(output_bias_index)())
np.save(os.path.join(args.working_dir, '/npy/classifier_6_weight.npy'), interpreter.tensor(output_weight_index)())
print(f"Weights and bias for the floating model saved to {os.path.join(args.working_dir, '/npy/')}.")
print('------------------------------------------------------------------')


