# Machine Learning Projects and Challenges

| Contribution   |
| :---   |
| Leonardo Villamil Dominguez  |
| Litian Chen                  |

This repository provides a collection of Machine Learning, Deep Learning and Computer Vision projects that the authors developed while completing an Advance Machine Learning. The problem definition for each project is listed below.

---
## Regression Models Based on Socio-economic Factors
The selected data contains socio-economic data from the 1990 US census for various US communities, and the number of violent crimes per capita. The purpose of this implementation is to explore the link between the various socio-economic factors and crime.

Given the provided data, This example included:
* The process of Spliting the data into training, validation and testing sets.
* The training of a linear regression model to predict the number of violent crimes per captia from the socio-economic data.
* The training of a LASSO regression model to predict the number of violent crimes per captia from the socio-economic data.
* The training of a Ridge regression model to predict the number of violent crimes per captia from the socio-economic data.

---
## Classification of Forest Type from Aerial Sensors Data
Land use classification is an important task to understand our changing environment. One approach to this involves the use of data from aerial sensors that captures different spectral reflectance properties of the ground below. From this data, the land type can be classified.

This project uses a training and testing data that include 27 spectral properties and an overall classification of land type, which can be one of:
* s: ‘Sugi’ forest
* h: ‘Hinoki’ forest
* d: ‘Mixed deciduous’ forest
* o: ‘Other’ non-forest land

Using this data we have train three multi-class classifiers to classify land type from the spectral data.
* A K-Nearest Neighbours Classifier
* An ensemble of Support Vector Machines (SVM)
* A Random Forest.

---
## Training and Adapting Deep Networks with Limited Data
When training deep neural networks, the availability of data is a frequent challenge. Acquisition of additional data is often difficult, due to logistical and/or financial reasons. As such, methods including fine tuning and data augmentation are common practices to address the challenge of limited data.

On this project we have been provided with two portions of data from the Street View House Numbers (SVHN) dataset. A training set containing only 1,000 samples total distributed across the 10 classes and A testing set of 10,000 samples total distributed across the 10 classes. These sets do no overlap, and have been extracted randomly from the original SVHN testing dataset. Note that the training set being significantly smaller than the test set is by design.

This implementantion included:
* The training of a deep learning model using no data augmentation, on the limited SVHN training set.
* The training of a model making use of data augmentation on the provided abridged SVHN training set.
* A fine tune approach, using an existing deep learnig model, trained on another dataset (such as MNIST, KMINST or CIFAR).

---
## Person Re-identification
Person re-identification is the task of matching a detected person to a gallery of previously seen people, and determining their identity. In formulation, the problem is very similar to a typical biometrics task (where dimension reduction techniques such as PCA and/or LDA, or deep network methods using Siamese networks can be applied), however large changes in subject pose and their position relative to the camera, lighting, and occlusions make this a challenging task.

Person re-identification (and performance for other retrieval tasks) is typically evaluated using Top-N accuracy and Cumulative Match Characteristic (CMC) curves. Top-N accuracy refers to the percentage of queries where the correct match is within the top N results.

This project makes use of the Market-1501 dataset. The training set contains the first 300 identities. Each identity has several images. In total, there are 5, 933 colour images, each of size 128x64. The testing consist of a randomly selected pair of images from the final 301 identities.All images are colour, and of size 128x64.

In using these datasets, te proposed solution has explored several approaches, including:
* The Development and evaluation of non-deep learning methods including PLC and LDA for person re-identification. 
* The Development and evaluation of deep learning based method, including the implementation of Siamese Neural Networks for One-shot Image Recognition and person re-identification.

---
## Clustering and Recommendations for Movies
Recommendation engines are typically built around clustering, i.e. finding a group of people similar to a person of interest and making recommendations for the target person based on the response of other subjects within the identified cluster.

This projects uses the MovieLens small dataset, which contains movie review data for 600 subjects. The data contains a tags.csv and a links.csv files. The first contains the movie ratings, and consists of a user ID, a movie ID, a rating (out of 5), and a timestamp. The second is a list of tags applied to movies by users.

Using this data, this implementation develop a method to cluster users based on their movie viewing preferences and provide recommendations for new content for the users.

---
## Semantic Person Search - Classification
Semantic person search is the task of matching a person to a semantic query. For example, given the query ‘1.8m tall man wearing jeans a red shirt’, a semantic person search method should return images that feature people matching that description.

Te selected dataset contains the following semantic annotations: Gender, Pose, Torso Clothing Type, Torso Clothing Colour, Torso Clothing Texture, Leg Clothing Type, Leg Clothing Texture and Luggage. n addition, the dataset contains semantic segmentation for each image in the training data

On this project we developed the initial state of a sematic person search system by using the selected data to implement multiple classifiers that, given an input image, classify the traits accordingly. This development also managed the impact of missing data and unknown classes by implementing Deep Neuronal Networks with multiple-outputs in combination with Auto-encoders and Fine tunning of Segmentation Networks.


