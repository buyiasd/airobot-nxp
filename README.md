#AIRobot Collision Avoidance

In this release, we'll first collect an image classification dataset that will be used to help Smart-Car avoid collision. We'll teach Smart Car to detect two scenarios free and blocked. Then, we use the AlexNet model to generate the .npy model, and we convert the .npy model to a quantified .tflite model.
 We also provide a quantified .tflite model located in the path of  ./models/ airobot_alexnet_uint8.tflite. This model can run in the AI-Robot chip directly to generate obstacle probability of Smart-car. If the AI-Robot detect that there is no obstacle ahead, it will go straight. If the AI-Robot detect that there is obstacle ahead, it will turn left.



##Hardware and required drivers
In this release, the hardware device uses the AI-Robot chip, and the corresponding software version is Yocto 5.10.35. 

##Data collection 
1. Enter the project directory, and open the terminal to enter the ipython interactive interface 
```mermaid
$:~# cd PROJECT_PATH
$:~# python3
```
2. Use the class of DataCollection to collect data. 
```mermaid
>>>from data_collection import DataCollection
>>>dc = DataCollection()
>>>dc.savefree() & dc.saveblocked()
>>>dc.stop()
```
[Usage]:  We can run dc.savefree() to save free image and dc.saveblocked() to save blocked image.After completing the data collection, we can turn off the camera by running dc.stop()

##Training
In the training and model quantitation steps, the required environment version is onnx==1.10.1,onnx-tf==1.8.0,tensorflow=2.4.1,tensorflow-addons==0.13.0,python=3.9.6
 1. Run the following code in the terminal 
```mermaid
$：python3 training.py
```

##Model quantification 
1. Convert the trained model to .tflite and quantify the model
```mermaid
$:~# python3 model_conversion_full.py
$:~# python3 model_conversion.py
```
[Usage]:  [--help] [--webcam use the video to run the model] [--image use an image to run the model] [--model_path the path of the model ] [--model_name the name of use model]

##Inference(Collision Avoidance)
1. Run the quantified .tflite model in the AI-Robot chip to control the smart car to avoid obstacles.
```mermaid
$:~#python3 demo-no-final-layer.py --webcam --model_path ./models/
```
[Usage]:  [--help] [--webcam use the video to run the model] [--image use an image to run the model] [--model_path the path of the model ] [--model_name the name of use model]

##References

The project is inspired by https://github.com/NVIDIA-AI-IOT/jetbot. Codes are changed based on https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/collision_avoidance

