# PRNU_image_source_identification
This is the repository for the project of the course "Trends and Applications of Computer Vision" at the University of Trento A.Y. 2022/2023.

The goal of the project is to test the [PCE](https://github.com/polimi-ispl/prnu-python) method and a [CNN](https://arxiv.org/pdf/2001.11847.pdf) method for the task of image source identification.

## Data
For rear camera devices we used the 35 devices in the [VISION](https://lesc.dinfo.unifi.it/VISION/) dataset while for the front camera devices we used images taken with the front camera of 11 devices. For all the devices we used 50 flat images to extract the PRNU and 40 natural images to test the methods.

The data used for our experiments is available [here](https://drive.google.com/file/d/1stkF2aT1JhA9NjRRQ8jnSUKhi2wGlosm/view?usp=share_link).

The `data` folder is divided in 2 folders `Vision_Frontal_Merged` and `Vision_Frontal_Separated`. These folders contains the `Vision` folder and the `Noises_lists`, these two folders are required by the `cnn-fast-sdi` scripts and contain respectively the PRNUs of the devices and the noise residuals of the natural images.
* `Vision_Frontal_Merged`: The devices inside here are ordered alphabetically regardless if the device is frontal or rear camera. This is useful if you want to test frontal and rear cameras together. The name of the devices are `DXY` from `D01` to `D46`.
* `Vision_Frontal_Separated`: The devices inside here are ordered from `D01` to `D35` for the [VISION](https://lesc.dinfo.unifi.it/VISION/) devices and from `F01` to `F11` for the frontal camera devices.

You can infer the real names of the devices from the files in the [`device_labels`](./devices_labels/) folder using the order `D01 - D46` and `F01 - F11`.

## Results
You can explore our results in the [results](./results/) folder that is divided in `PCE` (PCE method), `Pcn` and `EffB2` (CNN networks). Inside these folders you can find the folders for different crop sizes that contain the results for the frontal and Vision tests.

## Demo
To run the demo you need to run the docker container by exposing the port `8888` using the parameter `-p 8888:8888`. Then, inside the container, you need to install `jupyter_http_over_ws` with this command:
```shell
pip install jupyter_http_over_ws
```
and enable it with this command:
```shell
jupyter serverextension enable --py jupyter_http_over_ws
```
Finally you can run the notebook with this command:
```shell
jupyter notebook Demo.ipynb --port=8888 --ip 0.0.0.0 --no-browser --allow-root
```
and connect to it through the link that will appear on the terminal.

By default the demo will use a `Pcn 512` network, it will load the prnus of the frontal camera devices and it will run the model on all the images contained in `./shared/Telegram/`. The reported accuracy refers to the `Apple_iPhone13` that is the device used during the live demo, but you can change it by modifying a string in the final loop.

## Docker
To run the tests for [cnn fast source device identification](https://github.com/polimi-ispl/cnn-fast-sdi) we setup a docker image that uses `cuda9.0` required by `tensorflow 1.11`.

Before building the image you can download the data used from [here](https://drive.google.com/file/d/1stkF2aT1JhA9NjRRQ8jnSUKhi2wGlosm/view?usp=share_link)

Build the image with 
```shell
sudo docker build -t cnn_fast_sdi_runtime .
```

Run it with 
```shell
sudo docker run -it --gpus all cnn_fast_sdi_runtime /bin/bash
```
or if you need to move the data or the models inside the container run it with the `-v` parameter: 
```shell
sudo docker run -it --gpus all -v path/in/host/machine:/~/shared cnn_fast_sdi_runtime /bin/bash
```

To run the tests refer to the [official github repository](https://github.com/polimi-ispl/cnn-fast-sdi). If you want to test using our data you will have to do 2 things:

1. Add the `data/X/Noises_lists` and `data/X/Vision` folders inside the `cnn-fast-sdi` folder in the docker container. You can do this by running the container with the `-v` parameter or by modifying the `Dockerfile`.

2. Modify the list of devices in the `cnn-fast-sdi/utility_dataset.py` file:
* If you use `data/Vision_Frontal_Merged` you need to use this devices list 
```python
vision_dev_list = 'D01;D02;D03;D04;D05;D06;D07;D08;D09;D10;D11;D12;D13;D14;D15;D16;D17;D18;D19;D20;D21;D22;D23;D24;D25;D26;D27;D28;D29;D30;D31;D32;D33;D34;D35;D36;D37;D38;D39;D40;D41;D42;D43;D44;D45;D46'
``` 
* If you use `data/Vision_Frontal_Separated` you need to use this devices list 
```python
vision_dev_list = 'D01;D02;D03;D04;D05;D06;D07;D08;D09;D10;D11;D12;D13;D14;D15;D16;D17;D18;D19;D20;D21;D22;D23;D24;D25;D26;D27;D28;D29;D30;D31;D32;D33;D34;D35;F01;F02;F03;F04;F05;F06;F07;F08;F09;F10;F11'
``` 
where `DXY` is a device from the [VISION](https://lesc.dinfo.unifi.it/VISION/) dataset, `FXY` is a frontal camera device.

### Run a test
In general to run a test you can use this command 
```shell
python3 test_cnn.py --model_dir ./downloaded_pretrained_models/{model_name} --output_file {out_file}.npz --crop_size {crop_size} --base_network {architecture} --list_dev_test '{devices_list}
``` 
e.g.: 
```shell
python3 test_cnn.py --model_dir ./downloaded_pretrained_models/cropr224_Pcn --output_file ../shared/results/cropr224_Pcn_test.npz --crop_size 224 --base_network Pcn --list_dev_test 'F01;F02;F03;F04;F05;F06;F07;F08;F09;F10;F11'
```
to test a `Pcn` network with crop size `224` on frontal camera devices.

Before running the command make sure to have the `Vision` and `Noises_lists` folders in the local directory and the model inside the `downloaded_pretrained_models` folder.