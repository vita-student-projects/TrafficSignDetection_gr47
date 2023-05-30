# YODSO : You Only Detect Signs Once

### Introduction:

This project is part of the Deep Learning for Autonomous Vehicles class. Each group had to solve a specific task related to autonomous driving. Our group chose the task "traffic sign detection and classification", where the goal is to find the bounding boxes and class labels of signs, given an image. <br>
We are proud to introduce to the community YODSO (You Only Detect Signs Once). It is an implementation and finetuning of the model Yolov8. Yolov8 is the state-of-the-art deep learning model for object detection, developed by [Ultralytics](https://docs.ultralytics.com/) and released in 2023. In this Readme, we explain the steps to obtain our inferences and comment onf our results.

### Installation:

All the scripts are designed to be run on the SCITAS clusters. One can connect and request resources with the following commands:
```
ssh -X <gaspar_username>@izar.epfl.ch
Sinteract  -g gpu:1 -p gpu -c 8 -m 30G -t 01:29:00 -a civil-459-2023
```

Once on a node in the SCITAS clusters, one must set up the virtual environment.
```
virtualenv --system-site-packages <venvs_name>
source <venvs_name>/bin/activate
pip install ultralytics --user
pip install thop --user
```

Then a module is required to install.
```
module load gcc/8.4.0-cuda python/3.7.7		
```

Once all of that is done, the scripts are ready to be run.
Please note that the scripts expect the available resource to be 8 cores when running. If there is not enough resource available, a warning will be displayed, but it should work nonetheless.


### Data:

Our code is designed to work on the SCITAS clusters. It fetches all the image data from the folder /work/vita/nmuenger_trinca/annotations/. In this folder, there are .txt files that indicate the relative path to all the images from a set (the training set for example). As the original yolov8 model expected the data to be located in another folder, the path "/work/vita/nmuenger_trinca/annotations/" is hardcoded in the ultralytics\yolo\data\dataloaders\stream_loaders.py file, line 179. This path is also specified in the .yaml file (named *tsr_dataset_yaml*), where we specify how the dataset should be read. <br>
We use for training the [BelgiumTS dataset](https://btsd.ethz.ch/shareddata/). It contains 210 classes of traffic signs. The defined traffic signs are given in the image at the end of the readme.<br>
For an in-depth explanation of the data format YOLO expects, please refer to this [description](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data). 

### Models:

There are two kinds of models. The [models](https://github.com/ultralytics/assets/releases) pretrained by the Ultralytics team on the COCO dataset (Pretrained-Ultralytics) and our models which we finetuned on the BelgiumTS dataset (42-epochs). Both kinds come in 3 different sizes: Nano, Medium and XLarge, with performances increasing with the model's size.
Before anything, one must first download all the models from google drive (https://drive.google.com/drive/folders/1r6YhmoF5XZMKCcJ15O63RUL7nu7NAXp6?usp=sharing) and put them in the correct files. <br>

The pretrained models from Ultralytics must be placed in the *trained_models* folder while our finetuned models are in the *retrained_models_TS* folder.

The pretrained models must first be trained with our training script in order to predict traffic signs. Our finetuned models can directly be used for inference.

### Training:

The script to retrain the network is called retrain_TS.py. One can run it with this command, specifying the model to use and the number of epochs. The model to retrain must be in the *trained_models* folder.
```
python retrain_TS.py --model yolov8m.pt --epochs 3
```
The newly trained model and some metrics are then stored in the *run/detect* folder.

### Inference:

To infer the bounding boxes and classes of an image, we use the *inference_TS.py* script. One can specify the model to use, the path to the image (which must be in the same folder as the path specified in the .yaml file) and the file to output the predictions. An example command is provided below.

```
python inference_TS.py --model_path yolov8m_tsd_30epochs.pt --data_path images/01/image.006950.jp2 --output_file predictions_TS/prediction.txt
```

This will create the file *prediction.txt* file in the folder *predictions_TS* and store all the bounding boxes files in a csv format. The inference time on the image is also displayed.

We also made a script to infer from frames of a video and write in a json file. This script is named *video_inference_TS.py* and requires the frames to be placed in the *work/vita/nmuenger_trinca/annotations/video_frames/* folder in .png format. 

### Testing:

We also provide a test script to compare the performances of the models' different sizes on the whole test set. This script outputs all the results in the terminal and creates a plot comparing the performances. It is called *metrics_test_TS.py*.


### Disclaimer:

We discovered that not all the signs of the BelgiumTS dataset were labeled (see example below). During training, if the network correctly identifies a sign that was not labeled, it is told that it is wrong. We think that this is the reason we didn't get better results, as the yolov8 network is state-of-the-art and finetuning it should keep the performances high. 

The image below displays a visualisation of this problem. This picture was automatically generated by the YOLO training script and displays the labelisation of some of the images in the validation set. Most of the traffic signs are well annotated but we can see that some of the 'one-way street' signs (class C1.png) are not labeled at all. 

<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/image_read_me/Labels.jpeg" width="800">
</p>

### Results :

Our project goal was to retrain yolov8 and finetune it on the BelgiumTS dataset. We obtain our best results with the following method. For 14 epochs, we trained only the last part of the detection head (modules 16, 18, 19 & 21). Then we unfreezed the rest of the head (modules 12 & 15) and continued the training for 14 epochs. We found out in a previous training that unfreezing the backbone led to a drop in performance. Thus, we decided to keep it frozen and not retrain it, but we did finetune the whole head for another 14 epochs. The plot below displays our training curves.

<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/image_read_me/train_loss.png" width="400">
</p>

The architecture of yolov8 and the modules' numbers are shown in the image below. 

<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/image_read_me/yolo_archi.png" width="400">
</p>

The performance of the model on the whole test set is shown in the graph below. The three different sizes are compared, in terms of performance (map50 and map50-95) and inference time. In the second plot, we display the model size in Ko.

<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/image_read_me/map_time_with_test.png" width="400">
</p>

<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/image_read_me/size_vs_model.jpeg" width="400">
</p>

As an example, we selected an image with multiple signs to detect and infer the results with the 3 different models. We also compare the time it took to infer the prediction on this image (in milliseconds)

Ground truth 
<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/image_read_me/labels_GT.png" width="800">
</p>

Inference with the Nano model 
<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/image_read_me/Pred_n.png" width="800">
</p>

Inference with the Medium model
<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/image_read_me/Pred_m.png" width="800">
</p>

Inference with the Extra Large model
<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/image_read_me/Pred_x.png" width="800">
</p>

Comparison of inference time on one image.
<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/image_read_me/time_vs_model.jpeg" width="400">
</p>

### Table with the class names 

<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/image_read_me/defined_sign.png" width="400">
</p>







