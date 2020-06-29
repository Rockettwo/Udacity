This is the project repo for the final Capstone Project of the Udacity CarND.

The goal for this project is to navigate the car in the simulation considering the traffic lights and their state. The project was split up into several steps according to the online walkthroughs.

1. The basic waypoint_updater was implemented in order to follow the lane and be ready to implement the controller and the detection and classification task.
2. The dbw_node was implemented using the given PID and yaw controller.
3. The classification task took the most time. I decided to use a _ssdlite_mobilenet_v2_coco_ because it is fast and accurate. The steps of fine-tuning the model in Windows 10 environment are explained below. 
    This model was then implemented in the tl_detector node. At first I had some timing problems but I could solve them by installing the complete setup in a dualboot Ubuntu 16.04.
4. As the last big step, the waypoint_updater was implemented to use the output of the classifier, published to '/traffic_waypoint', which is the stop point if the light is red and -1 else.
5. Finetuning some parameters.

The currently selected model is trained mostly on the simulation dataset in order to get the optimal result for the task. Nevertheless another model is provided (model3.pb in ros/src/tl_detector) which was also trained on the Carla-data and also performs well on the testtrack.

The visualized structure:
![](imgs/final-project-ros-graph-v2.png | width=100 )


### Native Installation
* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Usage

1. Clone the project repository

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Install CUDA and cDNN and other resources

CUDA Version should be 8.x to be compatible with tf 1.3.0. 
I also had to install the following resources:
```bash
pip install python-rospkg
pip install pyyaml
```

4. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
5. Run the simulator

### Training the SSD on Windows with Powershell
I had large problems with training my Neuronal Network, so I want to provide some points I had to consider when training on Windows.

Check to have the a working setup. I used (NVIDIA GeForce GTX 1050 Ti) CUDA 10.0 with cDNN 7.5 in combination with tensorflow v1.15.3 and python v3.7.4.
I tried to apply to the instructions in https://github.com/marcomarasca/SDCND-Traffic-Light-Detection but this setup didn't work for me exactly. My modifications are explained in the following.

I used the [dataset of AlexLechner](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0) to train my networks as it comes already in the tf record format. The used model is going to be the _ssdlite_mobilenet_v2_coco_ from the TensorFlow ObjectDetection API. This model is pretty fast and was shown to achieve a high accuracy on the training data. It was chosen primarly because of its speed as I already encountered latency issues in my setup before. The configuration file and model is provided in the repo. The models can be downloaded from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). These models are pretrained on the COCO-dataset which already includes traffic light data. The configuration file has to be adjusted in order to fit your current setup.

1. Download model you want to train from mentioned sources.
2. Download the TensorFlow Object Detection API and perform the required installation steps:
    -   Checkout your root path for the model trainig
    -   Clone the tensorflow object models repo:    
        ```bash
        git clone https://github.com/tensorflow/models.git temp
        ```
    -   Copy temp\research\object_detection and temp\research\slim 
        ```bash
        cp -r temp\research\object_detection object_detection
        cp -r temp\research\slim slim
        ```
    -   Install dependencies (tf if not already installed - select version to fit to python version and cuda version)
        ```bash
        pip install tensorflow-gpu==1.15.3
        pip install Cython contextlib2 pillow lxml matplotlib pycocotools-windows absl-py
        ``` 
    -   Download protoc and unzip
        ```bash
        wget https://github.com/protocolbuffers/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip -OutFile protoc.zip
        Expand-Archive .\protoc.zip -DestinationPath .\protoc
        cp .\protoc\bin\protoc.exe .\
        ```
    -   Run protoc on files
        ```bash
        .\protoc.exe object_detection\protos\*.proto --python_out=.
        ```
    -   Add to PYTHONPATH
        ```bash
        $PYTHONPATH+=";" + $pwd.Path + ";" + $pwd.Path + "\slim"
        ```
    -   Run tests
        ```bash
        python object_detection\builders\model_builder_test.py
        ```
3. Create folders and populate with data
    -   data folder
        ```bash
        mkdir data
        cd data
        wget https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0 -OutFile dataset.zip
        Expand-Archive .\dataset.zip -DestinationPath ..\data
        rm -rf .\dataset.zip
        cd ..
        ```
    -   model folder
        ```bash
        mkdir model
        ```
        Copy the model to this path e.g.
        ```bash
        wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz -OutFile .\ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
        tar -xvzf .\ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
        rm -rf .\ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
        ```
4. Configure the pipeline in `model\ssdlite_mobilenet_v2_coco_2018_05_09\pipline.config`.
5. Run the training
    ```bash
   python object_detection\model_main.py --pipeline_config_path=model\ssdlite_mobilenet_v2_coco_2018_05_09\pipline.config \
                                         --model_dir=model\ssdlite_mobilenet_v2_coco_2018_05_09 
   ```
6. Run tensorboard
    ```bash
   tensorboard --logdir=model\ssdlite_mobilenet_v2_coco_2018_05_09
   ```
    Open your browser and follow the training process.
    
7. After training is finished your model has to be exported. This should be done in tf 1.4.0 to achieve compability with tf 1.3.0 on Carla.
   Best practice is to create a new environment with python 3.6. (see for example [this](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/) post.) 
   Then do the following steps (again most copied from [here](https://github.com/marcomarasca/SDCND-Traffic-Light-Detection)).
    -   **!!!!! This assumes to be executed in the same directory as above !!!!**
   
    -   Install dependencies and tf
        ```bash
        pip install tensorflow==1.4.0
        pip install pillow lxml matplotlib
        ```
    -   Set the working commit
        ```bash          
        cd temp
        git checkout d135ed9c04bc9c60ea58f493559e60bc7673beb7
        cd ..
        ```
    -   Extract the interesting part
        ```bash
        mkdir exporter
        cp -r temp\research\object_detection exporter\object_detection
        cp -r temp\research\slim exporter\slim
        ```
    -   Run protoc on files
        ```bash
        .\protoc.exe exporter\object_detection\protos\*.proto --python_out=.
        ```
    -   Add to PYTHONPATH
        ```bash
        cd exporter
        $PYTHONPATH+=";" + $pwd.Path + ";" + $pwd.Path + "\slim"
        cd ..
        ```
    -   Run tests
        ```bash
        python exporter\object_detection\builders\model_builder_test.py
        ```
8. Export the model
    ```bash
    python exporter\object_detection\export_inference_graph.py --input_type=image_tensor \
                       --pipeline_config_path=model\ssdlite_mobilenet_v2_coco_2018_05_09\pipline.config \
                       --trained_checkpoint_prefix=model\ssdlite_mobilenet_v2_coco_2018_05_09\model.ckpt-15000 \
                       --output_directory=.\converted
   ```
9. Use the exported model and include it in your process
