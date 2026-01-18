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
The first classifier we are required to implement is the nearest neighbor classifier. The goal is to assess how well the histogram-based representation discriminates between the 15 scene categories using a simple classification approach.

Firstly, each image in the test set is represented using the same Bag-of-Words pipeline adopted for the training images, in step 1 and 2. Then A Nearest Neighbor classifier is implemented, with ```n_neighbors=1```, to assign a label to each test image. Note that modifying the ```n_neighbors=K``` K parameter value to a higher number could provide better results since the classifier looks at the ```K``` closest training samples and assigns the test sample the class that is most common among those ```K``` neighbors (a majority vote). 

For simlicity the distance employed at this stage is the Euclidean distance which may not be the best choice but is a reasonable starting point, since we will later use more sophisticated classifiers and also explore the Chi squared distance. Note that other distance options for finding the "closest" neighbors, like cosine distance or Manhattan distance could be used.

Unlike parametric models, here the Nearest Neighbor classifier does not require an explicit training phase; it simply stores the feature vectors and corresponding labels of the training images.

After implementing setting up and implementing the k-NN classfier it is evaluated by comparing the predicted label of each test image with the ground truth (the actual label of the image). this is done in the following line of code: ```accuracy = accuracy_score(y_test, y_pred_nn)```, where ```y_test``` represents the true labels of the test dataset (like "Coast", "Forest" ...)
and ```y_pred_nn``` represents the predicted labels generated by the NN classifier for the same test dataset.
The final accuracy value is printed on stdout in percentage by simply multiplying ```accuracy``` by ```100```.

As for the results, this classifier performs decently well with an accuracy of roughly 30% with euclidean distance and ```n_neihbors=1```. Changing to manhattan distance increases the accuracy to 35%; however changing ```n_neihbors``` value to 15 seems to be the sweetspot pushing the accuracy to rougly 37%. 

In addition to accuracy, a confusion matrix is computed to provide a detailed analysis of performance. The confusion matrix reports, for each true category, the number of test images assigned to each predicted category. This allows easy identification of frequently confused classes, like visually similar indoor or outdoor environments. In this implementation, the confusion matrix is visualized using a heatmap representation, making inspection of the classifier’s strengths and weaknesses across the 15 categories easy to spot.
The most "confused" classes seem to be, Forest and Open County, Which makes sense since some open country images often include forests.


The Nearest Neighbor classifier provided a simple and effective baseline for evaluating the Bag-of-Words representation. While it did not model class boundaries explicitly, its performance reflected the quality of the visual vocabulary and the histogram representation. 
The results obtained with this method will serve as a reference point for comparison with more advanced classifiers, such as the linear Support Vector Machine presented in the following section.

# 4. Training a multiclass linear SVM
Since standard SVMs are inherently binary classifiers, we will train a multiclass linear Support Vector Machine (SVM) classifier using the one-vs-rest strategy, with the goal of learning decision boundaries for each of the 15 scene categories. 

Before training the SVM classifiers, the BoW histograms are normalized to ensure that all features are on a comparable scale. A ```StandardScaler``` is fitted on the training data and then applied to both the training and test histograms. This results in features with zero mean and unit variance, which helps the linear SVM treat all features equally during learning.

The scaler is fitted only on the training data to avoid using any information from the test set, while the same transformation is applied to the test data to ensure consistency between training and evaluation.

To handle the 15-category classification problem, a one-vs-rest strategy is used. This means that a separate binary classifier is trained for each of the 15 categories: samples belonging to the target class are labeled as positive (1), while all other samples are labeled as negative (0).

A total of 15 linear SVM classifiers are trained, one for each scene category. Each classifier learns a linear decision boundary that separates its class from the rest. The LinearSVC implementation from scikit-learn is used with a regularization parameter ```𝐶=1.0```, which balances the trade-off between maximizing the margin and minimizing classification errors.
Note that a lower C creates a wider margin, allowing more misclassifications but potentially leading to a more generalized model while a higher valued C creates a narrower margin, penalizing misclassifications more heavily. This results in a model that fits the training data more closely but might be prone to overfitting.

The trained classifiers will be later combined during evaluation to produce multiclass predictions.
Note that:
- The SVM training process is made deterministic using a fixed random seed
- A sufficiently large maximum number of iterations ensures that the optimization algorithm has time to converge
- The dual formulation is disabled to improve efficiency, as the number of training samples exceeds the feature dimensionality.


Compared to the Nearest Neighbor approach, the linear SVM classifier provides a more powerful alternative by explicitly learning class-specific boundaries in the feature space. While still relatively simple, this model should outperform the Nearest Neighbor baseline since it better captures the discriminative patterns across scene categories. The effectiveness of this approach is evaluated in the following section.



## 5. Evaluating the multiclass SVM










