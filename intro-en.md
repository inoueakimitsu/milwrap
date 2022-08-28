# Use scikit-learn models in multiple instance learning based on the count-based assumption 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inoueakimitsu/milwrap/blob/master/milwrap-experiment.ipynb.ipynb)

- [Use scikit-learn models in multiple instance learning based on the count-based assumption](#use-scikit-learn-models-in-multiple-instance-learning-based-on-the-count-based-assumption)
  - [Overview](#overview)
  - [Multiple instance learning problem setting and previous research](#multiple-instance-learning-problem-setting-and-previous-research)
    - [Overview of Multiple Instance Learning](#overview-of-multiple-instance-learning)
      - [Differences between ordinary supervised learning and MIL label formats](#differences-between-ordinary-supervised-learning-and-mil-label-formats)
      - [Field of application](#field-of-application)
    - [Two Perspectives on Multiple Instance Learning](#two-perspectives-on-multiple-instance-learning)
      - [The granularity of the correct label](#the-granularity-of-the-correct-label)
        - [Presence Assumption](#presence-assumption)
        - [Threshold Assumption](#threshold-assumption)
        - [Count-based assumption](#count-based-assumption)
        - [Compatibility between different granularity levels](#compatibility-between-different-granularity-levels)
      - [Viewpoint based on the relationship between the nature of the instance and the nature of the correct label](#viewpoint-based-on-the-relationship-between-the-nature-of-the-instance-and-the-nature-of-the-correct-label)
  - [Introduction to milwrap](#introduction-to-milwrap)
    - [Overview of the MIL methods](#overview-of-the-mil-methods)
      - [MIL methods](#mil-methods)
      - [SIL (Single Instance Learning) approach](#sil-single-instance-learning-approach)
      - [MIL-specific approach](#mil-specific-approach)
    - [mi-SVM](#mi-svm)
      - [Algorithm](#algorithm)
      - [Characteristics of mi-SVM](#characteristics-of-mi-svm)
    - [milwrap](#milwrap)
      - [Use of learning devices other than SVM](#use-of-learning-devices-other-than-svm)
      - [Support for multi-class classification](#support-for-multi-class-classification)
      - [Support of additional Multiple Instance Learning assumptions](#support-of-additional-multiple-instance-learning-assumptions)
  - [Accuracy evaluation experiment](#accuracy-evaluation-experiment)
    - [Method](#method)
      - [Existing method](#existing-method)
      - [Proposed method](#proposed-method)
      - [Oracle](#oracle)
      - [Hyperparameter](#hyperparameter)
    - [Dataset](#dataset)
    - [Results](#results)
    - [Discussion](#discussion)
  - [How to use milwrap](#how-to-use-milwrap)
    - [Install](#install)
    - [API](#api)
  - [References](#references)

## Overview

This article provides an overview of Multiple Instance Learning (MIL) and introduces the Python library milwrap for MIL.

There are three advantages of using milwrap.

- Multi-class classification
- Support of Count-based assumption
- Adapt any supervised learning model with scikit-learn-style APIs to the MIL framework.

In the experiments in this report, we have shown the following:

- When milwrap is used, it is possible to obtain the same level of accuracy with Multiple Instance Learning as with Single Instance Learning, under some conditions.
- Using information on data following Count-based Assumption without reducing the granularity of the data as much as possible will result in higher accuracy.

This article is organized as follows :

- Multiple instance learning problem setting and previous research
- milwrap algorithm
- Accuracy evaluation experiment
- How to use milwrap library

## Multiple instance learning problem setting and previous research

### Overview of Multiple Instance Learning

#### Differences between ordinary supervised learning and MIL label formats

Multiple instance learning is a weakly supervised learning task.
In short, MIL is a task that does not have a correct label per sample but has a correct one per sample group.
For example, you can learn in a situation where you have only coarse-grained information: Samples 1, 2, and 3 contain at least one positive example, but samples 4, 5, 6, and 7 contain no positive examples.


#### Field of application

According to Section 2.4 of Herrera, Francisco, et al. (2016), we used MIL in the following situations:

- Alternative representations
  - Different views, appearances, and descriptions of the same object are available
  - For example, it is used in the drug activity prediction
- Compound objects
  - For multi-part compound objects.
  - In the Image Classification Task example, the picture titled "British Breakfast" contains various objects in addition to the elements of the British meal.
- Evolving objects
  - For objects that are sampled multiple times over time.
  - For example, it is used in the prediction of bankruptcy.

In Section 2.4 of Herrera, Francisco et al (2016), the following application areas are listed :

- Bioinformatics
  - drug activity prediction
    - mutagenicity prediction of compound molecules
    - activity prediction of molecules as anticancer agents
  - protein identification
    - recognition of Thioredoxin-fold proteins
    - Binding proteins of the Calmodulin protein are identified in a multi-instance classification process
    - annotation of gene expression patterns
- Image Classification and Retrieval
  - This task deals with images that contain subimage.
  - image classification
  - facial recognition
  - Image search
- Web Mining and Text Classification
  - web mining (bag = index page, instance = other websites to which the page links)
  - document classification (bag = document, instance = passage)
- Object Detection and Tracking
  - horse detection
  - pedestrian detection
  - detection of landmines based on radar images
- Medical Diagnosis and Imaging
  - detection of tumors
  - detection of myocardial infarction based on electrocardiography (ECG) recordings
  - detection frailty and dementia in senior citizens using sensor data
  - detection of colonic polyps, abnormal growths in the colon using video classification
- Other classification Applications
  - prediction of student performance (bag = student, instance = work that the student has done)
  - automatic subgoal discovery in reinforcement learning (bag = trajectory of an agent, instance = observations made along this trajectory)
    - The label indicates whether the trajectory succeeded or failed.
  - stock selection problem

As you can see, there are many applications of MIL.

### Two Perspectives on Multiple Instance Learning

Let's take a closer look at the MIL.
The MIL can be broken down in two ways:

- The granularity of the correct label
- Relationship between instance and label

#### The granularity of the correct label

This is the framework presented by Weidmann et al. (2003).
MIL can be divided into Presence Assumption, Threshold Assumption, and Count-based Assumption according to the granularity of the correct answer label in order of coarse granularity.


##### Presence Assumption

This is the case where each bag is labeled whether it contains one or more instances of positive examples. Many MIL literature addresses tasks of this granularity.

##### Threshold Assumption

A case where each bag is labeled whether it contains more than N instances of positive examples.

##### Count-based assumption

The case where each bag is labeled with a lower bound L and upper bound U contains instances of positive examples. The milwrap described in this article can be applied to this case.

##### Compatibility between different granularity levels

A coarse-grained task can be rewritten as a fine-grained task. Thus, a technique that can handle fine-grained tasks can handle tasks with coarse-grained data. For example, a task with Presence Assumption data can be covered by a technique that can handle count-based Assumption tasks. The reverse is not true.

#### Viewpoint based on the relationship between the nature of the instance and the nature of the correct label

Although not discussed in depth in this article, there is a perspective that considers a stochastic process by which the labels of the bag are generated given unobservable labels on an instance-by-instance basis.

For example, there are models in which the greater the number of instances of positive examples in a bag, the greater the probability that the bag's label will be positive.

This perspective is called Collective Assumption.

milwrap does not support collective assumption yet.

## Introduction to milwrap

Now you have an overview of the MIL and a subclassification of tasks.

This section describes the milwrap library that can be used conveniently to solve MIL tasks.

First, we give a general overview of the MIL method.
Next, we will explain the mi-SVM which was the base of milwrap.
Then, we will explain the extension of milwrap to mi-SVM.

### Overview of the MIL methods

#### MIL methods

There are two approaches for MIL: one approach allows for instance-by-instance classification at inference time, and the other approach allows for bag-by-bag classification only.

This perspective is important, and for predicting pharmacological activity, it is important to predict on a per-instance basis.
On the other hand, there are cases such as image classification where there is no problem only by the prediction of the bag unit.

Note that this distinction is not a clear-cut one. For example, one of the deep learning-based methods can only make per-bag decisions in the big picture but can obtain per-instance results through the Attention mechanism, which can be viewed as an intermediate approach between the two approaches.
Be sure to consider whether per-instance predictions are really necessary. If per-instance forecasts are only needed to increase the interpretability of the per-bag forecasts, then the Collective Assumption method described above, which has more room for modeling, is a more flexible way to model.

#### SIL (Single Instance Learning) approach

One algorithm for MIL is Single Instance Learning (SIL), which allocates a per bag label as the label for all instances belonging to that bag. Since bags often contain many negative examples, it is assumed that the label is wrong, but surprisingly, there are many cases in which this is acceptable.

By all means, please consider an approach that accepts errors by regarding SIL as a baseline.


#### MIL-specific approach

Many techniques have been proposed to solve MIL tasks.
In recent years, many new methods have been proposed.

For example :

- Iterative Discrimination
- Diverse Density

See Reference 2 for more information.

### mi-SVM

mi-SVM is a MIL-specific solution that applies SVM to the Presence Assumption binary class-classification task.

#### Algorithm

The pseudocode for mi-SVM is quoted below.

```
initialize y_i = Y_I for i \in I
REPEAT
  compute SVM solution w, b for data set with  imputed labels
  compute outputs f_i ~ <w, x_i> + b for all x_i in positive bags
  set y_i = sgn(f_i) for every i _in I, Y_I = 1
  FOR (every positive bg B_I)
    IF (\sum_{i \in I}(1 + y_i)/2 == 0)
      compute i* = argmax_{i \in I} f_i
      set y_{i*} = 1
    END
  END
WHILE (imputed labels have changed)
OUTPUT (w, b)
```

mi-SVM is an approach that generates a pseudo-label, learns from the pseudo-label, and updates the pseudo-label using the estimated output score of SVM.


#### Characteristics of mi-SVM

The advantages of mi-SVM from the user's perspective are as follows:

- It is easy to implement because it does not differ from the SVM of normal SIL at the time of inference.
- Accuracy with the Presence assumption is good.

The challenges are :

- As it is based on SVM, it is unsuitable for large sample data.
- With the SIL task, subsampling is sufficient, but with the MIL setting, the subsampling method is not trivial (it is necessary to decide whether subsampling is performed per bag or instance).
- Does not support Threshold Assumption or Count-based Assumption.


### milwrap

#### Use of learning devices other than SVM

In milwrap, the part where SVM is used in the above mi-SVM algorithm is replaced with a method other than SVM which can calculate a predictive score.

For example, decision trees, logistic regression, Lasso regression, and deep-learning models are available.

In SVM, when a large amount of data is used, the gram matrix becomes large.
The motivation for milwrap development was to use the SGDClassifier and kernel approximation of scikit-learn.

Milwrap Version 0.1.3 can use a learner with the scikit-learn supervised learner APIs.

So if you use a skorch that wraps PyTorch like scikit-learn, you can MIL a PyTorch model.

For the estimation, milwrap is not required and it is a pure scikit-learn model (or skorch).

#### Support for multi-class classification

Since mi-SVM was only for binary classification, we developed the algorithm to allow multi-classification.

When you modify a label, it is an important aspect to decide which class to assign the label that needs to be modified to. In fact, this should be considered task-specific.


#### Support of additional Multiple Instance Learning assumptions

Milwrap supports count-based assumption, which is the most granular assumption.
When learning with milwrap, each bag is given a range setting for the minimum and maximum number for each class.
Learn to minimize the number of instances outside these ranges.

## Accuracy evaluation experiment

In this experiment, the data for a multi-class, count-based assumption task was used.
We compare the performances of the existing method and the proposed method and check whether the extension of the milwrap contributes to the improvement of accuracy.


### Method

#### Existing method

The existing method is mi-SVM.
The mi-SVM supports only single-class and Presence assumptions.
For this reason, we let one vs rest learn a binary class classifier for each class.
To support the Presence assumption, we reduce the granularity and use only information that is more than 1 or not.

#### Proposed method

The proposed method uses milwrap and SVM as a learner.
Two variations of the proposed method are used to see if multi-classing was effective.
A multiclass algorithm is called `milwrap-multiclass` and a milwrap OVR strategy is called `milwrap-ovr`.

#### Oracle

For comparison, we use single instance SVM to learn the case where the label per instance is obtained. We call it an `oracle`.


#### Hyperparameter

The model's hyperparameters are selected by cross-validation.
Hyperparameter candidates are as follows:

- C: 0.01, 0.1, 1.0, 10.0
- gamma: 0.01, 0.1, 1.0, 10.0

### Dataset

Data is artificially generated in the following ways:

1. Generate an average vector of each feature of each class from a multivariate uniform distribution. The number of classes is 5, and the number of dimensions is 7. The minimum value is 0, and the maximum value is 1.
2. Generate a discrete uniform distribution of the total number of instances for each bag. The number of bags is 100 for training and 100 for testing. Each bag must have a minimum of one instance and a maximum of 100 instances.
3. Generate a ratio for each class of each bag with a zero-inflated Dirichlet distribution. For each of the `prob_of_non_zero` cases of 0.2, 0.5, 0.7, and 1.0, assume that each class has a composition ratio of 0 with a probability of 1 - `prob_of_non_zero`. For classes whose composition ratio is not 0, determine the composition ratio with a Dirichlet distribution with alpha = 1.0.
4. Generate a class for each instance in the multinomial distribution. Note that even if `prob_of_non_zero` is 1.0, the number of instances allocated in the multinomial distribution can be 0.
5. The features of each instance are generated with an isotropic multivariate normal distribution. The standard deviation of each feature is 0.5.
6. Generate reference data according to count-based assumption with pre-determined adaptive width intervals, where intervals is `[0, 5, 10, 20, 50, 100]`.

### Results

The table below shows the RMSE of each method (`milwrap-multiclass`, `milwrap-ovr`, `misvm`, `oracle`) of the best hyperparameter setting.

| prob_of_non_zero | milwrap-multiclass | milwrap-ovr | misvm-ovr | oracle |
| :--------------- | :----------------- | :---------- | :-------- | :----- |
| 20%              | 8.33               | 8.90        | 8.72      | 8.50   |
| 50%              | 7.88               | 8.30        | 8.11      | 8.07   |
| 70%              | 5.63               | 8.65        | 22.48     | 5.47   |
| 100%             | 4.17               | 5.92        | 20.77     | 4.08   |

Overall, `milwrap-multiclass` is better than `milwrap-ovr` and `misvm-ovr`.
In particular, the higher the `prob_of_non_zero`, the greater the relative advantage of `milwrap-multiclass` over `misvm-ovr`.

The reason for this is that when prob_of_non_zero is high, the amount of information in the Presence assumption is reduced, which increases the advantage of a milwrap, which supports the Count-based assumption.

`milwrap-multiclass` is more accurate than `milwrap-ovr`.

`milwrap-multiclass` is quite close to `oracle`. Up to 50% of `prob_of_non_misvm`, `misvm-ovr` is almost as accurate as an oracle. This fact clearly shows the effectiveness of misvm's algorithm.

In situations where `prob_of_non_zero` is more than 50%, `misvm-ovr` is much more difficult, and in these situations, you should use `milwrap-multiclass`.

### Discussion

In this report, we compared the performances of the existing method and the proposed method for multiclass, count-based assumption tasks, and clarified that the extension of the mi-SVM to milwrap contributes to the improvement of estimation accuracy.

In the experiments in this report, we have shown the following:

- When milwrap is used, it is possible to obtain the same level of accuracy with Multiple Instance Learning as with Single Instance Learning, under some conditions.
- Using information on data following Count-based Assumption without reducing the granularity of the data as much as possible will result in higher accuracy.

The weak point is that learning milwrap is computationally intensive and takes more time compared to normal learning. In order to solve this problem, from the viewpoint of improving learning efficiency, it is possible to use a high-speed but low-accurarcy learner in the initial stage, and switch to a low-speed but high-precision learner in the latter stage. These directions are future issues.


## How to use milwrap

### Install

Run the following command:

```shell
pip install milwrap
```

### API

Prepare a single-instance supervised learning algorithm.
You can use classes with the `predict_proba()` method.
For example, if you are using logistic regression:

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
```

Wrap the learner with `MilCountBasedMultiClassLearner`.

```python
from milwrap import MilCountBasedMultiClassLearner 
mil_learner = MilCountBasedMultiClassLearner(clf)
```

Prepare data in the following format:

```python
# Prepare following dataset
#
# - bags ... list of np.ndarray
#            (num_instance_in_the_bag * num_features)
# - lower_threshold ... np.ndarray (num_bags * num_classes)
# - upper_threshold ... np.ndarray (num_bags * num_classes)
#
# bags[i_bag] contains not less than lower_threshold[i_bag, i_class]
# i_class instances.
```

Prepare data in the following format :

```python
# run multiple instance learning
clf_mil, y_mil = learner.fit(
    bags,
    lower_threshold,
    upper_threshold,
    n_classes,
    max_iter=10)
```

After learning is complete, it can be used as an ordinary single-instance learned model.

The `milwrap` library is unnecessary in inference.


```python
# after multiple instance learning, you can predict instance labels
clf_mil.predict([instance_feature])
```

## References

1. Andrews, Stuart, Ioannis Tsochantaridis, and Thomas Hofmann. "Support vector machines for multiple-instance learning." Advances in neural information processing systems 15 (2002). https://proceedings.neurips.cc/paper/2002/file/3e6260b81898beacda3d16db379ed329-Paper.pdf
2. Herrera, Francisco et al. “Multiple Instance Learning - Foundations and Algorithms.” (2016). http://www.amazon.co.jp/dp/3319477587
3. Weidmann, Nils, Eibe Frank, and Bernhard Pfahringer. "A two-level learning method for generalized multi-instance problems." European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2003. https://link.springer.com/content/pdf/10.1007/978-3-540-39857-8_42.pdf
