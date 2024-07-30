
# Lung Cancer Detection Using Machine Learning

This project aims to detect lung cancer in patients using various machine learning algorithms. The goal is to classify whether a patient has lung cancer (binary classification).




## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Performance](#models-and-performance)
- [Results](#results)

## Introduction
Lung cancer is one of the most common types of cancer, with high mortality rates. Early detection is crucial for effective treatment. This project uses machine learning models to classify lung cancer presence based on various attributes.


## Dataset
The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/jillanisofttech/lung-cancer-detection). It includes multiple features like age, gender, smoking history, anxiety, peer pressure, and others. The target variable indicates whether the patient has lung cancer.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/abdultalha0862/Lung-Cancer-Detection-Using-Machine-Learning.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Lung-Cancer-Detection-Using-Machine-Learning
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
##   Models

We used various machine learning algorithms to identify lung cancer, including:
 - Random Forest Algorithm
 - Decision Tree Algorithm
 - Support Vector Machine
 - Logistic Regression 
 - K-Nearest Neighbors
 - Naive Bayes
 - XGBoost Classifier
 - Gradient Boosting Classifier
 


## Results and comparison
The results section provides a detailed comparison of the performance of each model. The accuracy are provided for each model, allowing for an easy comparison of their effectiveness in detecting lung cancer.

Below is a summary of the accuracy results for each model:

| Algorithm                       | Accuracy (%) |
|---------------------------------|--------------|
| Decision Tree Classifier        | 91.07        |
| Logistic Regression             | 91.07        |
| Naive Bayes                     | 91.07        |
| XGBoost Classifier              | 87.50        |
| Gradient Boosting Classifier    | 87.50        |
| Random Forest Classifier        | 85.71        |
| K-Nearest Neighbors (KNN)       | 85.71        |
| Support Vector Machine (SVM)    | 83.93        |

## Acknowledgements
We express our gratitude to the contributors and the Kaggle community for generously providing the dataset. This project would not have been feasible without their valuable support and resources.