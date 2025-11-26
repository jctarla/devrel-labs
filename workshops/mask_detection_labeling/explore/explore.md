# Lab 1: Understand YOLOv5

Estimated Time: 10 minutes

## Overview - What is YOLOv5?

In this lab, I want you to grasp how [YOLOv5](https://github.com/ultralytics/yolov5), a specific Transfer Learning Model, and a Computer Vision framework, can help us solve the problem of trying to detect correct / incorrect mask placement in images.
 

YOLO (You Only Look Once) is a popular real-time object detection system developed by Joseph Redmon and Ali Farhadi. It's one of the latest versions of the YOLO system and was released in 2021.

Like other versions of YOLO, [YOLOv5](https://github.com/ultralytics/yolov5) is designed for fast, accurate, **real-time** object detection. It uses a _single_ convolutional neural network (CNN) to predict bounding boxes and class probabilities for objects in an image or video. The model is trained to predict the locations of objects in an input image and assign them to predefined categories, such as "car," "person," or "building."

In my personal experience -- and even though YOLOv5 isn't the newest detection system -- it's the one with the *lowest open issue to closed issue ratio*, which means that, for each open issue, more than 25 issues have already been closed.

![YOLOv5 example](./images/yolov5_example.jpg)

YOLOv5 improves upon previous versions by using more efficient network architectures and optimization techniques, resulting in faster and more accurate object detection. It also includes the ability to run on lower-power devices.

![YOLOv5 performance](./images/yolov5_performance.png)
> **Note**: this is a figure detailing the performance and accuracy of YOLOv5 compared to EfficientDet, and the [different variations of YOLOv5](https://github.com/ultralytics/yolov5#why-yolov5) (these are different checkpoints).

YOLOv5 has been widely adopted and is used in a variety of applications, including self-driving cars, robotics, and security systems, which is why I decided to start with this detection system rather than any others.

And you may ask yourself: "why YOLOv5 and not others?" Well, I compared YOLOv5 to YOLOv7, which was developed during this year (2022) and is more recent than YOLOv5. However, it currently has an open/closed issue ratio of [3.59](https://github.com/WongKinYiu/yolov7/issues), *87 times higher than YOLOv5*!. Therefore, I recommend YOLOv5 for getting started, as it's complete and the open-source community is more on top of this project. 
> **Note**: YOLOv8 was released very recently and I'm personally planning on making the switch between YOLOv5 and YOLOv8 in the upcoming months. Code is very similar, at least from the programmer's perspective, so making the change is something you can even do in this project if you're interested.

## Task 1: Why Custom Models?

Custom-detection machine learning (ML) models can provide numerous benefits in various applications.

- One major benefit is **increased accuracy and performance** when we compare these models to **general** models. Custom detection models are tailored specifically to the task and data at hand, allowing them to learn and adapt to the specific characteristics and patterns in the data. This can lead to higher accuracy and better performance than the previously-mentioned general models, which are not as tailored to the task.
- Also, a custom detection model will require a smaller number of resources to **train** it, as well as when making predictions with the model once it's been trained.
- Finally, there isn't a general-purpose model that's previously been trained to detect mask placement, so we have no other choice than to go with the custom model.

Here's an example of the final result of this workshop being run against my web camera feed. Since there are no pre-trained models that predict correct mask placement, achieving this isn't possible if we don't create a **custom model**:

![A GIF of myself.](./images/myself.gif)
So, now that we understand the **need** to create a custom YOLOv5 model, let's get straight into it.

You may now [proceed to the next lab](#next).


## Acknowledgements

* **Author** - Nacho Martinez, Data Science Advocate @ Oracle DevRel
* **Last Updated By/Date** - March 10th, 2023