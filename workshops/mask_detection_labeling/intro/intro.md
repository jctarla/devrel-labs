# Introduction

Estimated Time: 5 minutes

I've always been curious about using Vision ML in some projects of mine. I  dreamt of knowing how a Tesla autopilot worked on the inside, and whether I could make my own AI system at some point in my life. I was tired of dreaming, so I decided to learn by example (hands-on).

I set my focus on re-learning everything I knew about Vision ML (which is what I call image/video processing concerning Machine Learning in one way or another).

There are some topics, like Computer Vision, that are considered "too hard". Through this workshop, I aim to show you that it's not that hard to do this yourself (and not expensive either!).

Computer Vision has been a growing industry since its conception, and vision ML is one of the many components of Computer Vision. If you're interested in content like this, make sure to follow me, and stay tuned for part 2 (more info at the end).

Today, we're going to create a *computer vision model*, which will detect different mask-wearing states. So, it will be able to tell you if you, or anyone else that you want, is properly wearing a COVID-19 mask or not. The three different states we will consider are:
- A person with a mask, which we will label as `mask`.
- A person with a mask, but *incorrectly* worn (see examples below), which we will label as `incorrect`.
- A person with no mask at all, which we will label as `no mask`.

![validation batch - girl](./images/val_batch0_labels.jpg)

> **Note**: as you can see, the little girl in the second row and third column is wearing the mask with their nose showing, which is *incorrect*. We want our custom model to detect cases like these, which are also the hardest to represent, as there are a lot of pictures of people with and without masks, but there aren't as many pictures of people wearing masks incorrectly on the Internet; which causes our dataset to be imbalanced.

## Task 1: Final Result

This is a YouTube video illustrating what you will be able to achieve after completing this workshop.

[Watch the video](youtube:LPRrbPiZ2X8)

## Task 2: Objectives

In this lab, you will complete the following steps:

&check; Using RoboFlow to find good data

&check; Manipulating data and creating an optimal dataset

&check; Preparing OCI compute instances for training Computer Vision Models

## Task 3: OCI Elements

This solution is designed to work mainly with OCI Compute. We will use an OCI Compute Instance to save costs (compared to other Cloud providers) and train our Computer Vision model.

You can read more about the services used in the lab here:
- [OCI Compute](https://www.oracle.com/cloud/compute/)

## Task 4: Useful Resources

Here are three articles to get you from beginner to Computer Vision *hero*. This workshop is partly based on the content in these Medium articles.

- [Creating a CMask Detection Model on OCI with YOLOv5: Data Labeling with RoboFlow](https://medium.com/oracledevs/creating-a-cmask-detection-model-on-oci-with-yolov5-data-labeling-with-roboflow-5cff89cf9b0b)

- [Creating a Mask Model on OCI with YOLOv5: Training and Real-Time Inference](https://medium.com/oracledevs/creating-a-mask-model-on-oci-with-yolov5-training-and-real-time-inference-3534c7f9eb21)

- [YOLOv5 and OCI: Implementing Custom PyTorch Code From Scratch](https://medium.com/oracledevs/yolov5-and-oci-implementing-custom-pytorch-code-from-scratch-7c6b82b0b6b1)


You may now [proceed to the next lab](#next).

## Acknowledgements

* **Author** - Nacho Martinez, Data Science Advocate @ Oracle DevRel
* **Last Updated By/Date** - March 10th, 2023