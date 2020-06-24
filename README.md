# Face-recognition-using-deeplearning-hog-model
This repo consists of :

1.Face-recognition with image

2.Face-recognition with vedio

3.Realtime Face-recognition

Using Hog model with libraries and dataset

# Libraries used:

1.Numpy

2.Dlib

3.Opencv

# Introduction to the HOG Feature Descriptor

HOG, or Histogram of Oriented Gradients, is a feature descriptor that is often used to extract features from image data. It is widely used in computer vision tasks for object detection.

Let’s look at some important aspects of HOG that makes it different from other feature descriptors:

1.The HOG descriptor focuses on the structure or the shape of an object. Now you might ask, how is this different from the edge features we extract for images? In the case of edge features, we only identify if the pixel is an edge or not. HOG is able to provide the edge direction as well. This is done by extracting the gradient and orientation (or you can say magnitude and direction) of the edges

2.Additionally, these orientations are calculated in ‘localized’ portions. This means that the complete image is broken down into smaller regions and for each region, the gradients and orientation are calculated. We will discuss this in much more detail in the upcoming sections

3.Finally the HOG would generate a Histogram for each of these regions separately. The histograms are created using the gradients and orientations of the pixel values, hence the name ‘Histogram of Oriented Gradients’

Defination of HOG:
The HOG feature descriptor counts the occurrences of gradient orientation in localized portions of an image.
