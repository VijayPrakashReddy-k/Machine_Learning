# Machine_Learning
"Data is the new oil" is a saying which you must have heard by now along with the huge interest building up around Big Data and Machine Learning in the recent past along with Artificial Intelligence and Deep Learning. Besides this, data scientists have been termed as having "The sexiest job in the 21st Century" which makes it all the more worthwhile to build up some valuable expertise in these areas. Getting started with machine learning in the real world can be overwhelming with the vast amount of resources out there on the web. It's About ML Algorithms and Datasets.
![u4](https://user-images.githubusercontent.com/42317258/52902951-e81b2e00-323d-11e9-887f-fe1cddbbee6d.PNG)

Machine learning algorithms have 3 broad categories -
1.Supervised learning — the input features and the output labels are defined.
2.Unsupervised learning — the dataset is unlabeled and the goal is to discover hidden relationships.
3.Reinforcement learning — some form of feedback loop is present and there is a need to optimize some parameter.

![u3](https://user-images.githubusercontent.com/42317258/52902946-e18cb680-323d-11e9-8658-eed2c58228d7.PNG)
 ![ml1](https://user-images.githubusercontent.com/42317258/50925123-61717500-1477-11e9-9b73-6fe5eb5c57d9.PNG)
# 1.Supervised learning
## a) Regression (When response is a Continuous value)
![ml2](https://user-images.githubusercontent.com/42317258/50925694-1ce6d900-1479-11e9-815b-f92e41814e1a.PNG)

With linear regression, the objective is to fit a line through the distribution which is nearest to most of the points in the training set.
In simple linear regression, the regression line minimizes the sum of distances from the individual points, that is, the sum of the “Square of Residuals”. Hence, this method is also called the “Ordinary Least Square”.
Linear regression can also be achieved in case of multidimensional data i.e. data-sets that have multiple features. In this case, the ‘line’ is just a higher dimensional plane with dimensions ‘N-1’, N being the dimension of the dataset.
## b) Classification (When response is a categorical value)
#### Logistic Regression
![ml3](https://user-images.githubusercontent.com/42317258/50926175-62f06c80-147a-11e9-8c1f-4fd239cfa32f.PNG)

It’s a classification algorithm, that is used where the response variable is categorical. The idea of Logistic Regression is to find a relationship between features and probability of particular outcome.
Contrary to linear regression, logistic regression does not assume a linear relationship between the dependent and independent variables. Although a linear dependence on the logit of the independent variables is assumed.
In other words, the decision surface is linear.
![ml4](https://user-images.githubusercontent.com/42317258/50926686-9f709800-147b-11e9-87e5-845cc36665ef.PNG)

Support vector machine (SVM) is a supervised machine learning algorithm that can be used for both classification and regression challenges.
In SVM, we plot the data points in an N-dimensional space where N is the number of features and find a hyper-plane to differentiate the datapoints.
This is a good algorithm when the number of dimensions is high with respect to the number of data points.
Due to dealing with high dimensional spaces, this algorithm is computationally expensive.
![ml6](https://user-images.githubusercontent.com/42317258/50927384-a4364b80-147d-11e9-9eeb-793e82ff6ccd.PNG)

Decision tree is a classifier in the form of a tree structure.
Decision trees classify instances or examples by starting at the root of the tree and moving through it until a leaf node which is the target value.
Generating decision trees are useful as they mimic human understanding and thus, the models are easy to understand.
Small trees are better as the larger the trees, the less the accuracy.

### Naive Bayes algorithm :
It is a classification technique based on **Bayes’ Theorem** with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.
Historically, this technique became popular with applications in **Email filtering, spam detection, and document categorization.**
Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.

Bayes Theorem works on conditional probability. **Conditional probability is the probability that something will happen, given that something else has already occurred**.  Using the conditional probability, we can calculate the probability of an event using its prior knowledge.

![n1](https://user-images.githubusercontent.com/42317258/52218103-ebb6c880-28bf-11e9-98ab-034e51055630.PNG)

->P(c|x) is the posterior probability of class (c, target) given predictor (x, attributes).

->P(c) is the prior probability of class.

->P(x|c) is the likelihood which is the probability of predictor given class.

->P(x) is the prior probability of predictor.

![n2](https://user-images.githubusercontent.com/42317258/52219056-ea869b00-28c1-11e9-9eb0-ffea8371dda2.PNG)

### Types of Naive Bayes Classier:
**1.Multinomial Naive Bayes:** This is mostly used for **document classication problem**, i.e whether a document belongs to the category of **sports, politics, technology etc**. The features/predictors used by the classier are the frequency of the words present in the document.

![n3](https://user-images.githubusercontent.com/42317258/52220266-7e596680-28c4-11e9-8ee7-70e055e4e444.PNG)

**2.Bernoulli Naive Bayes:** This is **similar to the multinomial naive bayes but the predictors are boolean variables**. The parameters that we use to predict the class variable take up only values **yes or no**, for example if a word occurs in the text or not.

![n4](https://user-images.githubusercontent.com/42317258/52220277-831e1a80-28c4-11e9-948b-35bc47d7e229.PNG)


**3.Gaussian Naive Bayes:** When the predictors take up a continuous value and are not discrete, we assume that these values are sampled from a gaussian distribution.
![n11](https://user-images.githubusercontent.com/42317258/52268706-36891c80-2962-11e9-8a18-98cbc59ca3e0.PNG)

# 2.Unsupervised Learning :
No labeled responses, the goal is to capture interesting structure or information.
**Applications include:**<br>
- Visualise structure of a complex dataset
- Density estimations to predict probabilities of events
- Compress and summarise the data
- Extract features for supervised learning
- Discover important clusters or outliers
## I.Clustering
## II.Association Rule Mining
## II.Dimensionality Reduction / Transformation
![u1](https://user-images.githubusercontent.com/42317258/52902748-151a1180-323b-11e9-9553-960d49dc815f.PNG)
![u2](https://user-images.githubusercontent.com/42317258/52902750-1c411f80-323b-11e9-9439-143506b8d8fb.PNG)

## I.Clustering :
- Cluster analysis or clustering is the most commonly used technique of unsupervised learning. It is used to find data clusters such that each cluster has the most closely matched data.<br>

![a1](https://user-images.githubusercontent.com/42317258/52966741-09ab1f80-33ce-11e9-89b3-8e8577c432cb.PNG)

**Clustering Algorithms :** <br>
The types of Clustering Algorithms are: <br>
- 1.Prototype-Based Clustering <br>
- 2.Graph-Based Clustering (Contiguity-Based Clustering) <br>
- 3.Density-Based Clustering <br>
- 4.Well Separated Clustering <br>

![imgonline-com-ua-twotoone-r8gkhrwveobfuwqw](https://user-images.githubusercontent.com/42317258/52966715-fa2bd680-33cd-11e9-94e6-ed6070663018.png)

### 1.Prototype-based Clustering :
· If the data is *Numerical,* the prototype of the cluster is often **a Centroid i.e., the average of all the points in the cluster.**
· If the data has *Categorical attributes,* the prototype of the cluster is often **a medoid i.e., the most representative point of the cluster.**
· Objects in the cluster are closer to the prototype of the cluster than to the prototype of any other cluster.
· Prototype based clusters can also be referred to as **“Center-Based” Clusters.**
· These clusters tend to be **globular.(Circular)**
· **K-Means and K-Medoids** are the examples of Prototype Based Clustering algorithms
Prototype-based clustering assumes that most data is located near prototypes; example: **Centroids (average) or medoid (most frequently occurring point)** K-means, a Prototype-based method, is the most popular method for clustering that involves:
- Training data that gets assigned to matching cluster based on similarity <br>
- The iterative process to get data points in the best clusters possible <br>

#### i.K-means Clustering :
You have a set of data that you want to group into and you want to put them into *clusters,* which means objects that are similar in nature and similar in characteristics need to be put together. This is what k-means clustering is all about. 

![u8](https://user-images.githubusercontent.com/42317258/52905218-2d9b2380-325d-11e9-882a-e6bd64159a6c.PNG)

The term K is basically is a number and you need to tell the system how many clusters you need to perform. If K is equal to 2, there will be 2 clusters if K is equal to 3, 3 clusters and so on and so forth. That's what the K stands for and of course, there is a way of finding out what is the best or optimum value of K.(Elbow Method)

![u6](https://user-images.githubusercontent.com/42317258/52905209-178d6300-325d-11e9-9f91-7de79312b4f8.PNG)

Attempts to split data into K groups that are closest to K centroids.
This can be thought of as creating stereotypes among groups of people.
The algorithm to implement K means clustering is quite simple.

![u9](https://user-images.githubusercontent.com/42317258/52905222-368bf500-325d-11e9-8c7b-fbc1d504f59a.PNG)

-> 1.You randomly pick K centroids <br>
-> 2.Assign each datapoint to the centroid closest to it.<br>
-> 3.Recompute the centroids based on the average position of each centroid’s points <br>
-> 4.Iterate till points stop changing assignments to centroids. <br>
To predict you just find the centroid they are closest to. <br>

**Algorithm :**
The algorithms starts with initial estimates for the **Κ centroids,** which can either be randomly generated or randomly selected from the data set. The algorithm then iterates between two steps:

**1. Data assigment step :**
- Each Centroid defines one of the clusters. In this step, each data point is assigned to its nearest centroid, based on the squared Euclidean distance. More formally, if **ci is the collection of centroids in set C, then each data point x is assigned to a cluster based on**

              argmin   dist(Ci,x)^2
      Ci belongs to C
      
where dist( · ) is the standard (L2) Euclidean distance. Let the set of data point assignments for each ith cluster centroid be Si.

**2. Centroid update step :**
- In this step, the **centroids are recomputed.** This is done by taking the mean of all data points assigned to that centroid's cluster.

            Ci = 1/|Si| summation of (Xi)
                        Xi belongs to Si
                        
The algorithm iterates between steps one and two until a stopping criteria is met (i.e., no data points change clusters, the sum of the distances is minimized, or some maximum number of iterations is reached).

- **A key challenge in Clustering is that you have to pre-set the number of clusters.** This influences the Quality of clustering.
- Unlike Supervised Learning, here one does not have ground truth labels. Hence, to check the quality of clustering, one has to use *intrinsic methods,* such as the **within-cluster SSE,** also called **"Distortion".**

![u7](https://user-images.githubusercontent.com/42317258/52905214-1fe59e00-325d-11e9-9ecc-517c65071162.PNG)

- In the scikit-learn ML library, this value is available via the **inertia_ attribute** after fitting a K-means model.
- One could plot the Distortion against the number of clusters k. Intuitively, if k increases, distortion should decrease. This is because the samples will be close to their assigned centroids.
- This plot is called the **Elbow method.** It indicates the optimum number of clusters at the position of the elbow, the point where distortion begins to increase most rapidly.
- The adjoining Elbow method suggests that k = 3 is the most optimum number of clusters.

### 2.Graph-Based Clustering (Contiguity-Based Clustering) :

![u5](https://user-images.githubusercontent.com/42317258/52905207-11978200-325d-11e9-9481-458d3a896e2e.PNG)
· Two objects are connected only if they are within a specified distance of each other.
· Each point in a cluster is closer to at least one point in the same cluster than to any point in a different cluster.
· Useful when clusters are **irregular and intertwined(twist or twine together).**
· This does **not work efficiently when there is noise in the data,** as shown in the below picture, a small bridge of points can merge two distinct clusters into one.

· Clique is another type of Graph Based Cluster
· Agglomerative hierarchical clustering has close relation with Graph based clustering technique.


