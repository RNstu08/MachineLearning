# Topic 12: Naive Bayes Classifiers

## Overview

Naive Bayes classifiers are a family of simple, yet often powerful, **probabilistic classifiers** based on **Bayes' Theorem**. They are termed "naive" because they make a strong (and often unrealistic) assumption that all input features are **conditionally independent** of each other, given the class label.

Despite this simplification, Naive Bayes classifiers are widely used due to their efficiency, ease of implementation, and good performance in many real-world scenarios, particularly in text classification (e.g., spam filtering).

## Core Concepts

### 1. Bayes' Theorem

The foundation of Naive Bayes is Bayes' Theorem, which describes the probability of an event based on prior knowledge of conditions that might be related to the event. For classification, given a set of features `X = (x₁, x₂, ..., xₚ)` and a class `Cₖ`, it's stated as:

`P(Cₖ | X) = [ P(X | Cₖ) * P(Cₖ) ] / P(X)`

Where:
* `P(Cₖ | X)`: **Posterior probability** - Probability of instance `X` belonging to class `Cₖ`.
* `P(X | Cₖ)`: **Likelihood** - Probability of observing features `X` given class `Cₖ`.
* `P(Cₖ)`: **Prior probability** - Prior probability of class `Cₖ`.
* `P(X)`: **Evidence** - Probability of observing features `X`. (Acts as a normalizing constant and is often ignored for classification as it's the same for all classes).

The classification rule is to choose the class `Cₖ` that maximizes the posterior probability, or simply `P(X | Cₖ) * P(Cₖ)`.

### 2. The "Naive" Assumption of Feature Independence

Calculating `P(X | Cₖ)` directly is difficult. The naive assumption simplifies this by considering each feature to be conditionally independent of every other feature, given the class `Cₖ`:

`P(X | Cₖ) = P(x₁ | Cₖ) * P(x₂ | Cₖ) * ... * P(xₚ | Cₖ) = Π P(xᵢ | Cₖ)`

This allows the classifier to estimate `P(xᵢ | Cₖ)` for each feature `xᵢ` independently, which is much more feasible.

## Types of Naive Bayes Classifiers

The type of Naive Bayes classifier depends on the assumed distribution of `P(xᵢ | Cₖ)` for the features:

### a. Gaussian Naive Bayes

* **Assumption:** Features are continuous and follow a Gaussian (normal) distribution within each class.
* **Estimating `P(xᵢ | Cₖ)`:** Uses the mean and standard deviation of feature `xᵢ` for class `Cₖ`, calculated from the training data, and the Gaussian probability density function.
* **Use Case:** Continuous numerical features (e.g., sensor readings, measurements).
* **Scikit-learn:** `sklearn.naive_bayes.GaussianNB`.

### b. Multinomial Naive Bayes

* **Assumption:** Features represent counts or frequencies (typically non-negative integers).
* **Estimating `P(xᵢ | Cₖ)`:** Based on the frequency of feature `xᵢ` in samples belonging to class `Cₖ`.
* **Laplace/Additive Smoothing (parameter `alpha`):** Crucial for handling features not seen in the training set for a particular class (prevents zero probabilities).
* **Use Case:** Text classification (e.g., word counts using `CountVectorizer` or TF-IDF scores).
* **Scikit-learn:** `sklearn.naive_bayes.MultinomialNB`.

### c. Bernoulli Naive Bayes

* **Assumption:** Features are binary (0 or 1), indicating the presence or absence of an attribute.
* **Estimating `P(xᵢ | Cₖ)`:** Based on the frequency of feature `xᵢ` being present (or absent) in samples of class `Cₖ`.
* **Smoothing:** Also typically uses Laplace smoothing.
* **Use Case:** Text classification with binary term occurrence (word present/absent), or any problem with binary features.
* **Scikit-learn:** `sklearn.naive_bayes.BernoulliNB`.

## Advantages

* **Simple and Easy to Implement:** Understandable and straightforward.
* **Computationally Efficient:** Very fast to train and predict, as it involves simple calculations (no complex iterative optimization).
* **Requires Less Training Data:** Can perform reasonably well even with small datasets compared to more complex models.
* **Good Performance in Practice:** Especially effective in text classification (e.g., spam filtering, sentiment analysis).
* **Handles High-Dimensional Data Well:** The independence assumption helps manage the "curse of dimensionality."
* **Less Prone to Overfitting:** Due to its simplicity and often the use of smoothing techniques.

## Disadvantages

* **"Naive" Independence Assumption:** The core assumption of feature independence is often violated in real-world data, which can limit accuracy if dependencies are strong.
* **Zero-Frequency Problem:** If a feature value in the test set was not seen for a class during training, it can result in zero posterior probability for that class. Additive/Laplace smoothing (parameter `alpha`) mitigates this.
* **Continuous Feature Assumption (Gaussian NB):** Gaussian NB assumes features are normally distributed within each class, which might not always hold true.

## Common Applications

* **Text Classification:**
    * Spam Filtering
    * Sentiment Analysis
    * Document Categorization / Topic Classification
* **Medical Diagnosis (with caution):** As a preliminary diagnostic tool.
* **Recommendation Systems:** Classifying users or items.

## Implementation with Scikit-learn

**General Workflow:**
1.  Load and preprocess data.
    * For text, this involves **vectorization** (e.g., using `CountVectorizer` or `TfidfVectorizer` to convert text into numerical features).
    * For `GaussianNB`, feature scaling (`StandardScaler`) can be good practice, though it's less critical than for distance-based algorithms.
2.  Split data into training and testing sets.
3.  Instantiate the appropriate Naive Bayes model (`GaussianNB`, `MultinomialNB`, `BernoulliNB`).
    * Key parameter: `alpha` for smoothing in `MultinomialNB` and `BernoulliNB`; `var_smoothing` for `GaussianNB`.
4.  Train the model: `model.fit(X_train_features, y_train)`.
5.  Make predictions: `model.predict(X_test_features)`.
6.  Evaluate performance using appropriate classification metrics.

---
This README provides a summary of Naive Bayes Classifiers.