# CVPR Report on Project 2 - Bag-of-Words classifier

## Disclamer
The code used in this project was largely generated with the assistance of a large language model, due to my limited proficiency in programming languages. However, I am able to read, understand, and critically evaluate the implemented code. Thus, all design choices, results, and any potential errors or inaccuracies present in this project are solely my responsibility.

## Motivation
I selected this project because it aligns closely with my academic interests and has strong potential for reuse in a future Electronic System Design course. In that course, I plan to develop a system that acquires image data from a camera through an FPGA platform (DE10-SoC) and performs image processing tasks, with the goal of recognizing vehicle license plate numbers. This project provides a solid foundation for that objective, as it allows me to explore relevant concepts and techniques that can be extended and adapted for more advanced image acquisition and processing applications.

## Introduction
In this project, a complete Bag-of-Words image classification pipeline is implemented and evaluated on a 15-class scene recognition dataset. The pipeline includes visual vocabulary construction using k-means clustering, histogram-based image representation, and classification using both nearest-neighbor and multiclass linear Support Vector Machine classifiers. The performance of the proposed approach is assessed using confusion matrices and overall classification accuracy.

## Tools
- Google Colab
- ChatGPT/Gemini

## Setting up the enviroment
The chosen development environment for this project is Google Colab Notebook, as it offers a straightforward setup, minimal configuration requirements, and provides all the necessary computational tools. To organize the project, a dedicated folder was created in Google Drive, containing the provided training and test datasets obtained from Moodle. A Google Colab notebook was then used to access this directory and process the image data stored within it.

## 1. Building a Visual Vocabulary
To complete this task it was required to sample a certain number of sift descriptors, cluster them and save the centroids for future use. This was done similarly to the approach presented in class in the LabLecture2. The requirement stated that we would need a variable number of samples and cluster numbers, and this was implemented seemplesly in the code. SInce we are dealing with very different scenes in the dataset a MAX_PER_IMAGE limit on sift descriptors was used to try and avoid overrepresentation since some images might contain a very large number of descriptors like an image of a forest which might yield thousands of SIFT descriptors, while a simple image of a clear sky or a plain wall might only yield a few dozen.
Collecting all descriptors for each image would let more "complex" images dominate the pool and bias KMeans toward their features. Limiting descriptors per image ensures each image contributes more equally, producing a more balanced and robust visual vocabulary.
Proceeding with the task k-means clustering was implemented leaving the number of clusters as a user definable value to be able to experiment and fine tune the pipeline. Note that the n_init parameter in the KMeans fucntion is set to 10 essentialy runing the algorithm 10 times with different initializations and than choosing the best solution.

## 2. Representing train set images as histograms
After constructing the visual vocabulary, we now need to represent each  image of the training set as a normalized histogram having k bins, each corresponding to a visual word.
For a given image, local features are first extracted using the SIFT descriptor. Unlike the vocabulary construction stage, no explicit limit is imposed on the number of descriptors per image at this step, since the goal is to capture the full distribution of visual patterns present in each image.

Each extracted descriptor is then assigned to its closest visual word by finding the nearest cluster centroid in the previously learned k-means model. This assignment is performed using the k-means prediction function ```words = kmeans_model.predict(descriptors)```, which returns, for each descriptor, the index of the corresponding visual word.
We then build a histogram using the ```np.histogram``` function which counts how many times each visual word (each cluster index) appears in the "words" array, defined earlier.

To make the representation invariant to the number of detected keypoints and comparable across images, the histogram is normalized using L1 normalization, meaning that the sum of all bin values equals one. This step is essential to prevent images with many detected features from dominating distance-based classification methods, like the ones we will use in a later stage.

This procedure is then repeated for all images in the training set, producing a matrix of size ```N × k```, where N is the number of training images and k is the vocabulary size. Each row of this matrix represents the BoW histogram of one image and serves as input to the subsequent classification stages.

## 3. Nearest Neighbor classifier
