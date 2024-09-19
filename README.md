# Data Mining Project

## Overview

This project implements a knowledge extraction model to analyze vaccine data across different companies. We utilized two classification algorithms, three clustering algorithms, and one association rule mining algorithm to derive insights from the dataset. The project also involved extensive data preprocessing, and steps were taken to prevent overfitting and ensure robust evaluation.

## Project Goals

1. **Classification**: Predict which vaccine company each age group is associated with.
2. **Clustering**: Group the data characteristics based on the vaccines received by various age groups.
3. **Association Rules**: Extract associations between age groups and vaccine types.

## Algorithms Used

### Classification
- **Random Forest Classifier**: Selected for its ability to handle multiple categories with high accuracy.
- **K-Nearest Neighbors (KNN)**: Defined outcomes by creating regions that lead to consistent predictions.

### Clustering
- **K-Means**: Grouped the data into five clusters, corresponding to five vaccine types.
- **DBSCAN**: Created clusters based on data density.
- **Agglomerative Clustering**: Formed clusters using Euclidean distance, set to five groups (one for each vaccine type).

### Association Rule Mining
- **Apriori Algorithm**: Extracted frequent itemsets and rules between vaccines and age groups, focusing on age groups that had received at least one dose.

## Data Preprocessing

Data preprocessing was performed using Python libraries, and the following steps were taken:

- Removed irrelevant columns: `FirstDoseRefused`, `ReportingCountry`, `UnknownDose`, `YearWeekISO`.
- Converted categorical features into numerical ones using techniques like `get_dummies` and `map`.
- Applied **Principal Component Analysis (PCA)** to reduce the dataset to 9 principal components.
- Used normalization to simplify the dataset's feature values.
- Split the dataset into training and test sets using `train_test_split`.

## Methodology

The project used **Scikit-Learn** for implementing classification and clustering algorithms. The `train_test_split` method was employed to create training and testing datasets. Below are the key steps involved:

- **Random Forest Classifier**: Implemented using the `fit_transform` function on training data.
- **K-Nearest Neighbors**: Used a loop to select the best number of neighbors, based on accuracy.
- **Clustering Algorithms**: K-Means and Agglomerative required only the declaration of the desired cluster number. DBSCAN used a graph to determine the optimal `eps` parameter.
- **Apriori Algorithm**: Preprocessed data to extract association rules using minimum support and confidence thresholds.

## Experimental Evaluation

### Classification Performance
- **Random Forest Classifier**: Achieved an accuracy of 0.54.
- **K-Nearest Neighbors**: Struggled with time complexity but provided satisfactory accuracy.

### Clustering Performance
- **K-Means**: Effectively clustered the data into five categories based on vaccine type.
- **DBSCAN & Agglomerative**: Created clusters based on data density and Euclidean distance, respectively.

### Association Rule Mining
Generated association rules between vaccines and age groups, with the following diagrams produced:
- Support vs. Confidence
- Support vs. Lift
- Lift vs. Confidence

### Dimensionality Reduction
Principal Component Analysis (PCA) reduced the dataset to nine principal components, balancing dimensionality and feature representation.

## General Observations

The classifiers struggled with achieving high accuracy due to the complexity of categorizing features into five distinct categories. By reducing the vaccine companies to two categories, we achieved an accuracy of 0.9. However, clustering techniques proved more effective than classification, given the distribution of the dataset.

The **K-Folds method** was applied to check for overfitting and underfitting during the classification process. No significant overfitting or underfitting was detected.

## References

- [Scikit-Learn](https://scikit-learn.org/stable/)
- [DataCamp](https://www.datacamp.com/)
- [Analytics India Magazine](https://analyticsindiamag.com/)
- [Real Python](https://realpython.com/)