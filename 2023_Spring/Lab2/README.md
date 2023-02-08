# Lab 2 - Tiling-based Convolution for ResNet-50 Layer with HD Input

## Motivation

Suppose that you are given the following image and asked to draw tight-fitting boxes around various objects in the image. These could be cars, buses, pedestrians, etc. Furthermore, you have to classify each object with a different color. Seems quite trivial, right?

![input](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/89ede284-b058-42b6-acea-45604ef7ce3f/Input.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230208%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230208T212106Z&X-Amz-Expires=86400&X-Amz-Signature=877e13b976556f350d2b8ecc9bec234ab6cea15f748cdae96e9e1b096cfba8a1&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Input.png%22&x-id=GetObject)

For humans, it may be straightforward to accomplish this task. But for a computer, it requires massive amount of computation power and a well-designed, well-trained deep learning architecture to achieve the same. Not just for training the model, but also for inference.

Here's the final output obtained from a multi-object tracking model called [QDTrack](https://github.com/SysCV/qdtrack), which is the current state-of-the-art in computer vision research for this task.

![output](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/594aa4f0-29a2-40c3-aed8-f9e254d5e014/Final_Output.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230208%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230208T213443Z&X-Amz-Expires=86400&X-Amz-Signature=e0b2abe2fb1e2575cb8dfff0c9158c1d2017faceef201a6a52d83ac758016e53&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Final_Output.png%22&x-id=GetObject)

As may be evident, this "multiple object detection" is useful in many real-world applications, most predominantly in autonomous driving. However, models like QDTrack are huuuuuggge! Not just the model, but the size of inputs (HD and Full-HD) also add immensely to the computation complexity. And with scale comes challenges. In this lab, we will learn to tackle these challenges as we implement a convolution layer of the QDTrack model using HLS. 

Are you ready?

## Convolution for Object Detection

At the core of almost every object detection model is a convolution neural network (CNN) such as VGG-16, ResNet50, Xception, YOLO, MobileNet, etc. These are essentailly feature extraction networks. They "look" at images and extract salient features such as edges, shapes, and so on. 

Introduced in 2015, ResNet-50 is (still) one of the most popular CNN architectures and forms the backbone of many multi-object detection models, including QDTrack. Let's consider the first convolution layer of ResNet-50 which involves a 7 x 7 convolution. After passing the above input image through this layer, we get 64 two-dimensional "feature maps" as shown below. As you can observe, each layer extracts different features from the input image.

![layer](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/dd914915-8292-4b19-8ae7-92fc9008ccc0/Conv_Output.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230208%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230208T230507Z&X-Amz-Expires=86400&X-Amz-Signature=9c4b07e8b863d794887cf77e15b4207754400b14d4376d140753f229c5f711fb&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Conv_Output.png%22&x-id=GetObject)

We are interested in implementing this convolution layer of ResNet-50 with the above HD input image. Suppose that the input image (feature map) is described by a 3D tensor ```X(ID, IH, IW)```. We use a filter ```W``` with 64 kernels, each of dimensions ```(ID, KH, KW)``` where ```KH``` and ```KW``` are kernel window heights and widths. The resulting output feature map can be described by another 3D tensor ```Y(OD, OH, OW)```. The output feature map dimesions are a function of the stride ```S``` and padding size ```P``` chosen. For the first layer of ResNet-50, the values of these parameters are described in the table below.

| Layer Parameters |  Layer Values |
| ---------------- | ----------------- |
| Kernel Size (KH x KW)  | 7 x 7 |
| Filter Size	(OD) | 64 |
| Input Feature Map (ID, IH, IW) | (3, 736, 1280) |
| Stride (S) | 2 |
| Padding (P) | 3	|	
| Output Feature Map (OD, OH, OW) | (64, 368, 640) |

If these values do not make any sense to you in the first glance, don't worry. You can learn these concepts easily through this well-written [CNN cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks). 

This lab will be split into following parts:
- **Part A**: C Model creation for functional correctness (30 points)
- **Part B**: Unoptimized but synthesizable tiled-convolution in HLS (30 points)
- **Part C**: Optimized (and synthesizable) tiled-convolution in HLS (40 points)

More of Part B and Part C later. To start with, you have to simply implement a 7 x 7 convolution with all your C/C++ knowledge and ensure functional correctness.

## Part A: Implementation of C Model for Convolution (30 points)

Before we jump into the HLS implementation and start to play around with those pragmas that we love, we need to have a functionally correct implementation prepared to assist us with debugging. Without a reference C model, it is incredibly painful to debug the tiling-based convolution that you will implement after this.

In this part of the lab, you have to write a C/C++ program that performs 7 x 7 convolution. You are provided with a testbench which reads the input image, layer parameters, as well the reference output feature map and performs a point-to-comparison with the obtained feature map. Mean Squared Error (MSE) is used as the metric for evaluation.

- ```bin``` contains the reference binary files
- ```sim.cpp``` is the testbench that checks for functional correctness
- ```model_conv.cpp``` is the source code file you need to update to implement 7 x 7 convolution

There are a few caveats in handling border pixels that you would need to keep in mind while writing your convolution code. This [article](https://sharc-knowledgebase.netlify.app/articles/cnn/tiling-based_convolution_for_hls/) from last year's assignment explaining 3 x 3 convolution may be useful. Most of the concepts are extensible to 7 x 7 convolution. 

**Note**: This part of the lab is **simulation-only**. You do not have to run Vitis synthesis!

### How to Run the Code
Just like in Lab 1, simply use the provided ```Makefile```. Run ```make``` and then execute the output (if successfully generated) with ```csim.out```.

### What to Submit 
1. ```model_conv.cpp``` that passes the functional test.
2. A brief report (preferably in PDF format) describing:
   - What is the MSE obtained?
   - How did you handle the non-unit stride and border pixels?

### Submission Deadline
Submission: on Canvas for course students, via email (to TA) for Special Problems students

Due date: **Feb. 18 (Sat), 11:59 PM, no extension**.

## Part B: Tiling-based Unoptimized Convolution (30 points)
To be released soon.

## Part C: Tiling-based Optimized Convolution (40 points)
To be released soon.

## Grading Rubric
### Part A.1
> simulationTestPass &rarr; +25 points
### Part A.2
> Missing or incomplete information &rarr; -2 points for each question

## Academic Integrity and Honor Code
You are required to follow Georgia Tech's [Honor Code](https://policylibrary.gatech.edu/student-life/academic-honor-code) and submit your original work in this lab. You can discuss this assignment with other classmates but you should code your assignment individually. You are **NOT** allowed to see the code of (or show your code to) other students.

We will be using the Stanford MOSS tool to detect plagiarism. When there is reasonably clear evidence of a violation, a referral to the Office of the Dean of Students will occur, and all hearings and other resulting procedures will be followed to completion.
