# PRNU_image_source_identification
PRNU based Image source identification with CNN

## Docker
To run the tests for [cnn fast source device identification](https://github.com/polimi-ispl/cnn-fast-sdi) we setup a docker image that uses `cuda9.0` required by `tensorflow 1.11`.

Before building the image you can download the data used for our tests from [this folder](https://drive.google.com/drive/folders/1acw1PHVOqNpinJwpvI-08Xs-r1DE_AkD?usp=sharing)

Build the image with `sudo docker build -t cnn_fast_sdi_runtime .`

Run it with `sudo docker run -it --gpus all cnn_fast_sdi_runtime /bin/bash` or if you need to move the data or the models inside the container run it with the `-v` parameter: `sudo docker run -it --gpus all -v path/in/host/machine:/~/shared cnn_fast_sdi_runtime /bin/bash`

To run the tests refer to the [official github repository](https://github.com/polimi-ispl/cnn-fast-sdi). If you want to test using our data you will have to do 2 things:

1. Add the `data/X/Noises_lists` and `data/X/Vision` folders inside the `cnn-fast-sdi` folder of the docker container. You can do this by running the container with the `-v` parameter or by modifying the `Dockerfile`

2. Modify the list of devices in the `cnn-fast-sdi/utility_dataset.py` file. 
* If you use `data/Vision_Frontal_Merged` you need to use this devices list `vision_dev_list = 'D01;D02;D03;D04;D05;D06;D07;D08;D09;D10;D11;D12;D13;D14;D15;D16;D17;D18;D19;D20;D21;D22;D23;D24;D25;D26;D27;D28;D29;D30;D31;D32;D33;D34;D35;D36;D37;D38;D39;D40;D41;D42;D43;D44;D45;D46'`. 
* If you use `data/Vision_Frontal_Separated` you need to use this devices list `vision_dev_list = 'D01;D02;D03;D04;D05;D06;D07;D08;D09;D10;D11;D12;D13;D14;D15;D16;D17;D18;D19;D20;D21;D22;D23;D24;D25;D26;D27;D28;D29;D30;D31;D32;D33;D34;D35;F01;F02;F03;F04;F05;F06;F07;F08;F09;F10;F11'` where `DXY` is a device from the [VISION](https://lesc.dinfo.unifi.it/VISION/) dataset, `FXY` is a device for which the samples have been acquired with the frontal camera.
