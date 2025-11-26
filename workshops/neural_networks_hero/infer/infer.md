# Lab 5: Inference (Real-Time Predictions)

Estimated Time: 10 minutes

## Introduction

You may be asking yourself: how can I perform inference / how can I use my model?

Note that, since training & augmentation is done, we can spin down our OCI Instance, unless we want to perform real time inference on it.

In my opinion, it's better to use inference on your **local computer**, as this will be the only way to achieve true real time inference.

And now, we have arrived at the last lab of this workshop. This lab will teach you how to use the model in real time.

There are two notable ways to use the model:

- Using the _integrated_ YOLOv5 predictor and processor (beginner level)
- Using your own custom Python code (intermediate-advanced level)

We'll quickly go over both of these methods so you can use which case is better for you.

### Prerequisites

- It's highly recommended to have completed [the first workshop](../../workshops/mask_detection_labeling/index.html) before starting to do this one, as we'll use some files and datasets that come from our work in the first workshop.
- An [Oracle Free Tier, Paid or LiveLabs Cloud Account](https://signup.cloud.oracle.com/?language=en&sourceType=:ow:de:ce::::RC_WWMK220210P00063:LoL_handsonLab_introduction&intcmp=:ow:de:ce::::RC_WWMK220210P00063:LoL_handsonLab_introduction)
- Active Oracle Cloud Account with available credits to use for Data Science service.

### Objectives

In this lab, you will complete the following steps:

&check; Perform the easiest form of inference with YOLOv5

&check; Perform a more advanced form of inference, with custom Python code

## Task 1: Inference with Integrated YOLOv5 (Beginner)

This inference method is the easiest one, as it's already implemented by YOLO, and we just have to invoke it. I highly recommend running inference on your own local computer.

First, we go to our YOLOv5 directory:

```bash
<copy>
cd /home/$USER/yolov5/
</copy>
```

And invoke their predictor, called _`detect.py`_. Let's invoke it:

```bash
<copy>
~/anaconda3/bin/python detect.py --weights=<path_to_weights_file.pt> --img <model image dimensions> --conf <conf_threshold (0-1)> --source=<SOURCE>
</copy>
```

Each parameter represents the following:
- _`--img`_: dimensions of the file(s) we're going to pass the model. If the model was trained with X image size, it usually makes sense to specify a similar image dimension here.
- _`--weights`_: path to the final `best.pt` file (returned after model training).
- _`--source`_: this option is great because it allows us to specify any type of source. We can give it things like: 
    * YouTube video URL
    * Directory (it will perform inference on every file inside the directory)
    * Individual Video, in which case it will perform inference frame-by-frame and merge the result into a final video file.
    * Individual Image

For example, let us execute:

```bash
<copy>
~/anaconda3/bin/python detect.py --weights="./models/mask_model/weights/best.pt" --img 640 --conf 0.4 --source="./videos/my_video.mp4"
</copy>
```

## Task 2: Custom Inference with Python (Advanced)

For this method, we're going to use **PyTorch** as the supporting framework. We need PyTorch to load the model, obtain results that make sense, and return these results.

To follow this short tutorial, you will need two files:

- [requirements.txt](./files/requirements.txt) file, containing all dependencies you need to install before running the main Python file
- [The code](./files/pytorch_inference.py)

If you run the following Python code, you will be able to run your own custom model. PyTorch Model Hub's library allows us to load our PyTorch-compatible, YOLO-trained model. Then, we will make predictions, draw bounding boxes and print results in real-time.

You can always modify this code to your convenience, to implement things like:

- Saving to a file
- Streaming results over RTMP or HTTP

Or even expand the functionality, with things like counting objects, combining several Computer Vision models to achieve something more complex, integrating with databases... you name it. If you're interested in expanding the original functionality, refer to the following article, which illustrates how to do some intermediate-level things:

- [YOLOv5 and OCI: Implementing Custom PyTorch Code From Scratch](https://medium.com/oracledevs/yolov5-and-oci-implementing-custom-pytorch-code-from-scratch-7c6b82b0b6b1)

## Task 3: Conclusions

We have arrived at the end of this workshop.

In my case, I processed this example video against our newly-trained model, and it produced the following results:

[Watch the video](youtube:LPRrbPiZ2X8)

By this point, you should already be able to:

&check; Use OCI to help you train your own Computer Vision models.

&check; Learn about **Automatic Data Augmentation** to improve our datasets (with YOLO), and how to perform it.

&check; Learned how to train the model on custom data, and how to use this model in real time (inference).

I hope this helped you bootstrap your Computer Vision, PyTorch, and YOLO skills to the next level.

If you’re curious about the goings-on of Oracle Developers in their natural habitat like me, come join us [on our public Slack channel!](https://bit.ly/odevrel_slack) We don’t mind being your fish bowl 🐠.

Stay tuned...

## Acknowledgements

- **Author** - Nacho Martinez, Data Science Advocate @ Oracle DevRel
- **Last Updated By/Date** - June 1st, 2023
