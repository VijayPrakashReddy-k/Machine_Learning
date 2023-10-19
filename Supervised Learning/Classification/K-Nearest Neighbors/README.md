## K-Nearest Neighbors Models in Supervised Learning

K-Nearest Neighbors (KNN) is a simple, yet powerful supervised machine learning algorithm used for classification and regression tasks. The basic idea behind KNN is to predict the target variable for a new observation based on the outcomes of its ‘K’ most similar instances (neighbors) from the training dataset. Unlike other learning algorithms, K-Nearest Neighbors does not produce a model per se. Instead, it memorizes the training dataset, and predictions are made on-the-fly by finding the ‘K’ training observations closest to the new observation.

### KNN Algorithm

`The KNN algorithm can be described in the following steps:`

  **1. Choose the number of neighbors, K.**
  
  **2. For a new observation, calculate the distance between the observation and every other instance in the training dataset.**
  
  **3. Sort these distances and select the top ‘K’ instances from the training dataset.**
  
  **4. For classification: Return the mode of the target variable of the ‘K’ nearest neighbors.**
  
  **5. For regression: Return the mean of the target variable of the ‘K’ nearest neighbors.**

The effectiveness of K-Nearest Neighbors is determined by the choice of

        The distance metric and
        
        The value of ‘K’.

Commonly used distance metrics include the ***Euclidean distance, Manhattan distance, and Minkowski distance.***

### Mathematical Representation

Given a new observation x
, the distance d
 between x
 and a training observation xi
 is often computed using the Euclidean distance:

### Euclidean Distance Formula

The formula to calculate the Euclidean distance between two vectors \( x \) and \( x_i \) in \( p \)-dimensional space is:
```math
d(x, x_i) = \sqrt{\sum_{j=1}^p (x_j - x_{ij})^2} 
```
Where:
- \( d(x, x_i) \) represents the distance between two vectors, \( x \) and \( x_i \).
- \( p \) is the number of dimensions.
- \( x_j \) and \( x_{ij} \) are the elements of the vectors \( x \) and \( x_i \) respectively.

### Choosing the Right Value of K

Choosing the appropriate value of ‘K’ is crucial. *A small ‘K’ captures noise in the data, while a large ‘K’ may smoothen the decision boundaries excessively.*

One approach is to use **Cross-Validation.** For each potential value of ‘K’, perform cross-validation and choose the ‘K’ that gives the best cross-validated performance.


### Advantages and Disadvantages of KNN
**Advantages:** 

- Simple to understand and easy to implement.

- No assumptions about the distribution of the data.

- Naturally handles multi-class classification.

**Disadvantages:**

- Computationally expensive, especially with a large dataset.

- Sensitive to irrelevant features.

- Sensitive to the scale of the data. It’s recommended to normalize the data.

### Data Preparation for K-Nearest Neighbors Models

Data preparation plays a significant role in the success of K-Nearest Neighbors (KNN) models. Due to the nature of the algorithm, which relies on distance computations between observations, even slight irregularities or issues in the data can affect the model’s performance drastically.

#### Scaling and Normalization:

KNN calculates the distance between pairs of instances. If one feature has a larger range of values than another, it will dominate the distance computations. This can skew the results and may lead to inaccurate predictions.

Normalize or scale the features so they all have similar scales. 

The two common methods are:

**1. Min-Max Scaling:**

This scales the data to have values between 0 and 1.
```math
x′=x−min(x)/max(x)−min(x)
```

**2. Z-Score Normalization:**

This scales the data based on the mean and standard deviation.
```math
x′=x−x¯/s
```

### Variable Importance in K-Nearest Neighbors Models:

The K-Nearest Neighbors (KNN) algorithm doesn’t provide a direct measure of feature importance.

### Methods to Assess Variable Importance in KNN

**1. Permutation Importance:**

This method involves perturbing each feature and observing the effect on the model’s performance. The rationale is that if a feature is important, randomizing its values will degrade the model’s performance.

**2. Forward Selection and Backward Elimination:**

Both are wrapper methods where the idea is to either incrementally add features (forward selection) or remove features (backward elimination) and observe the effect on performance.

**3. Limitations and Considerations:**

`Computational Expense:` Methods like permutation importance can be computationally expensive, especially with high-dimensional data.

`Interactions:` These methods mostly consider each feature’s individual importance. However, interactions between features can also be influential.

`Correlated Features:` If two features are highly correlated, one might appear less important if the other is already in the model. Consider multicollinearity when interpreting results.

### Impact of Correlated Predictors in K-Nearest Neighbors Models

Correlated variables that are in a k-nearest neighbors model will create redundant terms in the distance calculations. If your predictor variables are highly correlated, consider replacing your predictors with a reduced dimension principal component vectors.


### Distance Metrics in K-Nearest Neighbors Models
K-Nearest Neighbors (KNN) is inherently a distance-based algorithm. At its core, KNN operates by computing distances between instances to identify the ‘nearest’ neighbors for prediction. Given the centrality of distance computation in KNN, the choice of distance metric can profoundly influence the model’s behavior and performance.

#### Commonly Used Distance Metrics in KNN

**1. Euclidean Distance:**
The most commonly used distance metric, Euclidean distance, calculates the “straight line” distance between two points in Euclidean space.

The formula to calculate the Euclidean distance between two vectors \( p \) and \( q \) in \( n \)-dimensional space is:

```math
d(p, q) = \sqrt{\sum_{i=1}^n (p_i - q_{i})^2} 
```
Where:
- \( d(p, q) \) represents the distance between vectors \( p \) and \( q \).
- \( n \) is the number of dimensions.
- \ ( p_i \) and \( q_i \) are the elements of the vectors \( p \) and \( q \) respectively.

It’s intuitive and works well for many datasets, especially when features are continuous and are on similar scales.

**2. Manhattan (L1) Distance:**

Also known as the “taxicab” or “city block” distance, it calculates the distance traveled along axes at right angles.
```math
d(p,q)=\sum_{i=1}^n|p_i−q_i|
```
Manhattan distance can be more robust to outliers in some cases as compared to Euclidean distance.
