# Lab 2: Infrastructure

Estimated Time: 10 minutes

## Introduction

If you have completed [the first workshop](../../workshops/mask_detection_labeling/index.html) and you have your own vision model, there's no need for you to complete this section. 
However, if you haven't, and don't have your own custom final dataset ready, you can use my dataset to get started. Additionally, we will need to quickly prepare the infrastructure that we need to manually train our Computer Vision model in this workshop.

From this infrastructure, we will learn how to do the things that we've done automatically with RoboFlow during *the first* workshop (training, data augmentation), but using 100% open-source code.

The OCI service that we'll use for this is **OCI Compute**. 

### Prerequisites

* It's highly recommended to have completed [the first workshop](../../workshops/mask_detection_labeling/index.html) before starting to do this one, as we'll use some files and datasets that come from our work in the first workshop.

* An [Oracle Free Tier, Paid or LiveLabs Cloud Account](https://signup.cloud.oracle.com/?language=en&sourceType=:ow:de:ce::::RC_WWMK220210P00063:LoL_handsonLab_introduction&intcmp=:ow:de:ce::::RC_WWMK220210P00063:LoL_handsonLab_introduction)
* Active Oracle Cloud Account with available credits to use for Data Science service.


### Objectives

In this lab, you will complete the following steps:

&check; Downloading a Computer Vision dataset (if you don't come with your own)

&check; Creating the necessary infrastructure in OCI to train models

&check; Installing Python dependencies effectively


## Task 1: Download Dataset

If you haven't completed the [first workshop](../../workshops/mask_detection_labeling/index.html), then you can use my dataset to get started. For this, go into [the project's RoboFlow Universe URL](https://universe.roboflow.com/jasperan/public-mask-placement/dataset/4):

![access dataset](./images/access_dataset.png)

Once we're on this website, we choose to download the dataset **in YOLOv5 format**:

![click the download button](./images/click_download_button.png)

We unzip it, and make sure that our _`data.yaml`_ file looks like this:

![yaml format](./images/yaml_modified.png)

This file holds all links between our YOLOv5 dataset, so once we have our paths ready and verified that the class names and the number of classes are correct, we can proceed to augment and train this dataset.

## Task 2: Create OCI Instance

First, we'll go into our [OCI Compute panel](https://cloud.oracle.com/compute/instances) and create a new instance. The name of the instance can be anything you want, and you should choose the Availability Domain (AD) depending on your region and the AD's allocated resources (varies over time). I usually go with AD 1, and if I run into any issues, I iterate.

![create instance - step 1](./images/create_instance_1.png)

We're going to use a **platform image** from Oracle called **OCI DSVM**. This image contains several tools for data exploration, analysis, modeling and development. It also includes a Jupyter Notebook, a conda environment ready to use and several more things (like Christmas for a Data practicioner).

![create instance - step 3](./images/create_instance_3.png)

We can find the platform image by selecting the *Marketplace* button:

![create instance - step 2](./images/create_instance_2.png)

Network settings for the Virtual Machine are very standard. Just make sure to create a new VCN and a new subnet, so that there's no possible way we get any networking issues from other OCI projects you may have.
![create instance - step 4](./images/create_instance_4.png)

Finally, we'll let OCI generate a SSH keypair, which we'll use to connect to the machine.

![create instance - step 5](./images/create_instance_5.png)

## Task 3: Access OCI Instance

To access our instance, let's copy the primary IP address that was allocated to our VM once the provisioning of this machine is done.

![access instance - step 1](./images/access_instance_1.png)

Once we have the IP address, and having previously saved our public-private keypair (which is what we will use to authenticate ourselves to the machine), let's connect through SSH. 


### Linux & macOS Users

Open a new terminal and run:

```
<copy>
ssh -i <private_key_absolute_path> opc@IP_ADDRESS
</copy>
```

For example, I have my SSH keypair saved in my downloads folder. Imagine my VM's IP address is 192.168.0.1. I will execute:

```
<copy>
ssh -i /home/user/Downloads/my_key.key opc@192.168.0.1
</copy>
```
> **Note**: `opc` is the default user for Oracle Linux Operating Systems, like the one running in our VM.

### Windows Users

For Windows users, I recommend using [MobaXterm](https://mobaxterm.mobatek.net/). 

First, set up a new session, making sure to specify the remote host (IP address), username (opc), and the private key's location:

![access instance - mobaxterm](./images/access_instance_mobaxterm.png)

Now, just click on "Quick Connect" and connect:

![access instance - mobaxterm 2](./images/access_instance_mobaxterm_2.png)


> **Note**: we will connect to our VM  and start training / augmenting our data with open-source repositories. 

## Task 4: Clone Open-Source Repositories 

Once we have connected to our instance, let's download two repositories: YOLOv5 and YOLOv8. You're free to choose either one of them to train and augment our computer vision models, but this guide will show you how to proceed with YOLOv5.

Run the following commands:

```
<copy>
# download yolov5
git clone https://github.com/ultralytics/yolov5.git 
# download yolov8*
git clone https://github.com/ultralytics/ultralytics.git
</copy>
```
> **Note**: `git` is another tool that's already installed in the custom image we used to spin up our instance. *YOLOv8 can also be installed directly from pip. More information [in this link.](https://github.com/ultralytics/ultralytics#documentation)

## Task 5: Transfer Dataset

Now that we're connected to the machine, let's move the files from our computer to our OCI Compute Instance.

### For Linux & macOS Users

We can use the _`scp`_ tool to help us transfer files through SSH:

```
<copy>
scp -i ~/.ssh/id_rsa.pub FILENAME USER@SERVER:/home/USER/FILENAME
</copy>
```
> **Note**: the _`-r`_ option stands for recursive (we must give it a directory instead of a file name).
So, in our case, it will be:

```
<copy>
scp -i ~/.ssh/id_rsa.pub -r /home/$USER/Downloads/dataset_directory opc@192.168.0.1:/home/$USER/final_directory
</copy>
```
> **Note**: in this case, my OCI Compute Instance IP is 192.168.0.1. `opc` is the username for Oracle Linux distributions, like the one we are using for this case. And the private key shall be the one we used to connect through SSH in the previous task.


### For Windows Users 

Use the integrated MobaXterm FTP explorer to transfer files, dropping files from our computer to MobaXterm's explorer, like here but the opposite:

![transfer to moba ftp](./images/transfer_moba.gif)

## Task 6: Install Python Dependencies

Once we have the repositories ready, we need to install dependencies that will allow us to run YOLO code:

```
<copy>
cd /home/$USER/yolov5
pip install -r /home/$USER/yolov5/requirements.txt
</copy>
```

Now that we have cloned our repositories, uploaded our dataset, and have our machine and conda environment ready, we're virtually ready to start training. You may now [proceed to the next lab](#next).

## Acknowledgements

* **Author** - Nacho Martinez, Data Science Advocate @ Oracle DevRel
* **Last Updated By/Date** - May 17th, 2023