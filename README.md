# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Data Preparation: Collect and preprocess customer data, including features like age, income, and spending score, scaling if necessary.
Model Training: Apply the K-Means algorithm to the data, specifying the desired number of clusters (k).
Model Evaluation: Evaluate clustering performance using metrics like inertia or silhouette score to validate the cluster quality.
Segmentation: Assign each customer to a cluster and analyze the group characteristics for insights.

## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: Priyadharshan S
RegisterNumber:  212223240127
*/


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
%matplotlib inline

# Download the dataset and read it into a Pandas dataframe
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/CustomerData.csv', index_col=0)

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


km = KMeans(n_clusters=5, random_state=42)
km.fit(X)

img = plt.imread('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/images/cameraman.png')

k = 2

X = img.reshape(-1, 1)
km = KMeans(n_clusters=k, random_state=42)

km.fit(X)

seg = np.zeros(X.shape)
for i in range(k):
    seg[km.labels_ == i] = km.cluster_centers_[i]
seg = seg.reshape(img.shape)
plt.imshow(seg)
```

## Output:
![image](https://github.com/user-attachments/assets/8ae00b4e-3e4b-4e0a-a942-694d0e7df75d)
![image](https://github.com/user-attachments/assets/93ec60d3-05a9-463c-9361-d76cf03e374c)
![image](https://github.com/user-attachments/assets/ee6e5914-127f-42fd-b85c-1f05e674841e)

## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
