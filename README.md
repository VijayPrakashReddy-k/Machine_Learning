# Machine_Learning
"Data is the new oil" is a saying which you must have heard by now along with the huge interest building up around Big Data and Machine Learning in the recent past along with Artificial Intelligence and Deep Learning. Besides this, data scientists have been termed as having "The sexiest job in the 21st Century" which makes it all the more worthwhile to build up some valuable expertise in these areas. Getting started with machine learning in the real world can be overwhelming with the vast amount of resources out there on the web. It's About ML Algorithms and Datasets
        ![ml1](https://user-images.githubusercontent.com/42317258/50925123-61717500-1477-11e9-9b73-6fe5eb5c57d9.PNG)

Machine learning algorithms have 3 broad categories -
1.Supervised learning — the input features and the output labels are defined.
2.Unsupervised learning — the dataset is unlabeled and the goal is to discover hidden relationships.
3.Reinforcement learning — some form of feedback loop is present and there is a need to optimize some parameter.
## 1.Supervised learning
### a) Regression
![ml2](https://user-images.githubusercontent.com/42317258/50925694-1ce6d900-1479-11e9-815b-f92e41814e1a.PNG)

With linear regression, the objective is to fit a line through the distribution which is nearest to most of the points in the training set.
In simple linear regression, the regression line minimizes the sum of distances from the individual points, that is, the sum of the “Square of Residuals”. Hence, this method is also called the “Ordinary Least Square”.
Linear regression can also be achieved in case of multidimensional data i.e. data-sets that have multiple features. In this case, the ‘line’ is just a higher dimensional plane with dimensions ‘N-1’, N being the dimension of the dataset.
### b) Classification
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

![ml5](https://user-images.githubusercontent.com/42317258/50927489-f7a89980-147d-11e9-987d-0d6fad8a0b0c.PNG)

Attempts to split data into K groups that are closest to K centroids.
This can be thought of as creating stereotypes among groups of people.
The algorithm to implement K means clustering is quite simple.

You randomly pick K centroids
Assign each datapoint to the centroid closest to it.
Recompute the centroids based on the average position of each centroid’s points
Iterate till points stop changing assignments to centroids.
To predict you just find the centroid they are closest to.

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
**1.Multinomial Naive Bayes:** This is mostly used for document classication problem, i.e whether a document belongs to the category of sports, politics, technology etc. The features/predictors used by the classier are the frequency of the words present in the document.

**2.Bernoulli Naive Bayes:** This is similar to the multinomial naive bayes but the predictors are boolean variables. The parameters that we use to predict the class variable take up only values **yes or no**, for example if a word occurs in the text or not.

**3.Gaussian Naive Bayes:** When the predictors take up a continuous value and are not discrete, we assume that these values are sampled from a gaussian distribution.
