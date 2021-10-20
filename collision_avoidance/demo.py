import time
import numpy as np
import os
import tflite_runtime.interpreter as tflite
import cv2
import argparse
import serial
import time
MODEL_NAME = 'airobot_alexnet.tflite'
MODEL_PATH = '/collision_avoidance/models/'

# Normalizing parameters - same as the ones used in training
mean = np.expand_dims(np.expand_dims(
    255.0 * np.array([0.485, 0.456, 0.406]), 1), 1)
stdev = np.expand_dims(np.expand_dims(
    255.0 * np.array([0.229, 0.224, 0.225]), 1), 1)


def preprocess(img):
    global mean, stdev
    # x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
    x = img.transpose((2, 0, 1))
    # x = (x - mean) / stdev
    x = np.expand_dims(x, 0)
    return x


def softmax(arr):
    exp_arr = np.exp(arr)
    return exp_arr / exp_arr.sum()


def warmup(interpreter):
    # ignore the 1st invoke
    start_time = time.time()
    interpreter.invoke()
    delta = time.time() - start_time
    print(f'Warm-up time: {(delta * 1000):.3f}ms')
    print('============================================')


def get_output(interpreter, output_details):
    global weight, bias
    start_time = time.time()
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    full_model_output = output_data @ weight.T + bias
    result = np.squeeze(softmax(full_model_output))
    stop_time = time.time()
    print(full_model_output)
    print(f'Inference time: {((stop_time - start_time) * 1000):.3f}ms')
    print(f'Blocked probability = {result[0]:.5f}\n')
    return result[0], stop_time - start_time


parser = argparse.ArgumentParser()
parser.add_argument('--image', help='image to run detector on')
parser.add_argument('--webcam', action='store_true')
parser.add_argument('--model_name', default='airobot_alexnet_uint8.tflite')
parser.add_argument('--model_path', default='./models')
args = parser.parse_args()

weight = np.load(os.path.join(args.model_path, 'npy', 'classifier_6_weight.npy'))
bias = np.load(os.path.join(args.model_path, 'npy', 'classifier_6_bias.npy'))

if __name__ == '__main__':
    interpreter = tflite.Interpreter(model_path=os.path.join(args.model_path, args.model_name))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    assert input_details[0]['dtype'] == np.uint8

    if args.webcam:
        cap = cv2.VideoCapture(3)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        # Warmup
        _, frame = cap.read()
        input_data = preprocess(cv2.resize(frame, (224, 224)))
        interpreter.set_tensor(input_details[0]['index'], input_data)
        warmup(interpreter)

        start_time = time.time()
        interpreter.invoke()

        while True:
            _, frame = cap.read()
          
            input_data = preprocess(cv2.resize(frame, (224, 224)))
            interpreter.set_tensor(input_details[0]['index'], input_data)
            output, inference_time = get_output(interpreter, output_details)
            

            ser = serial.Serial('/dev/ttymxc0',115200)
            if ser.isOpen == False:
                ser.open()
            if output > 0.75 and output <= 0.8:
                serial_left = [126,35,0,70,0,127]
                for i in range(6):
                    ser.write(chr(serial_left[i]).encode("utf-8"))
                    time.sleep(0.01)
                print("retrun_left") 
            if output > 0.8 and output<=0.85:
                serial_leftt = [126,36,0,70,0,127]
                for i in range(6):
                    ser.write(chr(serial_leftt[i]).encode("utf-8"))
                    time.sleep(0.01)
               
            if output <= 0.75:
                serial_1 = [126,34,0,70,0,127]
                for i in range(6):
                    ser.write(chr(serial_1[i]).encode("utf-8"))
                    time.sleep(0.01)
              
            if output > 0.85 and output <= 0.9: 
                serial_ll = [126,37,0,70,0,127]
                for i in range(6):
                    ser.write(chr(serial_ll[i]).encode("utf-8"))
                    time.sleep(0.01)
            if output > 0.9: 
                serial_lll = [126,38,0,70,0,127]
                for i in range(6):
                    ser.write(chr(serial_lll[i]).encode("utf-8"))
                    time.sleep(0.01)


            cv2.putText(frame, f"Blocked Probability: {output:.3f}", (20, 25), cv2.FONT_HERSHEY_DUPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("preview", frame)
            cv2.waitKey(5)

    else:
        assert (args.image), "Specify --webcam or --image. "

        frame = cv2.imread(os.path.abspath(args.image))
        input_data = preprocess(cv2.resize(frame, (224, 224)))
        interpreter.set_tensor(input_details[0]['index'], input_data)
        warmup(interpreter)

        start_time = time.time()
        interpreter.invoke()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        output, inference_time = get_output(interpreter, output_details)
