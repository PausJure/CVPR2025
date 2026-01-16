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
To complete this task it was required to sample a certain number of sift descriptors, cluster them and save the centroids for future use. This was done similarly to the approach presented in class in the LabLecture2. The requirement stated that we would need a variable number of samples and cluster numbers, and this was implemented seemplesly in the code. Since there was an ambiguity in the task that stated sample 10k-100k descriptors from images meaning total or per image. Nonthe less Experiments were done at a later stage and the optimal value seems to be around....

