# Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

----
## Overview

In this project, a convolutional neural network (CNN) was build to classify traffic signs. The [data](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip) used in training and testing this model, which are provided by Udacity, come from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). It is a pickled data set in which all the images have been resized to 32x32.

____
## Final result

The CNN structure is: Input -> conv -> relu -> conv -> relu -> pooling -> conv -> relu -> conv -> relu -> pooling -> conv -> relu -> conv -> relu -> pooling -> dropout -> fully-connected -> relu -> dropout -> fully-connected - relu -> output. The accuracies on the training, validaton and testing data set are **0.9938**, **0.9960** and **0.9758** respectively.

The projected was written in Jupyter notebook, which can be found at [GermanTrafficSign_Tensorflow.ipynb](./GermanTrafficSign_Tensorflow.ipynb). The corresponding html file can be found at [GermanTrafficSign_Tensorflow.html](./GermanTrafficSign_Tensorflow.html).

----
## Working on AWS GPU

1. You will need an AWS account with permission to use g2-2xlarge.

2. Select "udacity-carnd" in the community AMI (not necessary, but there is a setup environment).

3. Select the instance type: "g2-2xlarge".
 
4. Adjust the storage size (if necessary, at least 16 GB).

5. Configure the security group. For Jupyter notebook, you will need add a TCP port 8888.

6. Launch the instance.

7. Copy your files to the instance.

8. Log in
   `ssh carnd@ec2-52-58-44-132.eu-central-1.compute.amazonaws.com(your Public DNS (IPv4))`

9. Upgrade or install packages
    ```
    pip install scikit-learn --upgrade
    pip install tensorflow-gpu
    pip install tensorflow-gpu --upgrade
    ```
10. Solve the CUDA problem :(
    ```
    sudo apt-get remove nvidia-*
    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run
    sudo bash ./NVIDIA-Linux-x86_64-367.57.run  --dkms
    ```

11. Run Jupyter notebook

12. Access the Jupyter notebook locally from your web browser by visiting: [IPv4 Public IP of the EC2 instance]:8888

13. Enjoy
