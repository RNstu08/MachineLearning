# Topic 10: K-Nearest Neighbors (KNN)

## Overview

K-Nearest Neighbors (KNN) is a simple, versatile, and intuitive **non-parametric**, **instance-based (or memory-based) learning** algorithm. It can be used for both **classification** and **regression** tasks.

* **Non-parametric:** It makes no strong assumptions about the underlying data distribution.
* **Instance-based (Lazy Learner):** It doesn't learn an explicit model or function during a distinct training phase. Instead, it memorizes the entire training dataset and performs computations only when a prediction is requested for a new instance.

The core idea is based on the principle that instances similar to each other (i.e., "near" in the feature space) likely belong to the same class or have similar target values.

## How KNN Works

### 1. For Classification:

1.  **Choose K:** Select a positive integer `K` (the number of nearest neighbors to consider).
2.  **Calculate Distances:** For a new, unseen data point, calculate its distance to *every* data point in the training set using a chosen distance metric.
3.  **Find K Nearest Neighbors:** Identify the `K` training instances with the smallest distances to the new point.
4.  **Majority Vote:** Assign the class label to the new data point that is most frequent among its `K` nearest neighbors. (An odd `K` is often chosen for binary classification to avoid ties).

### 2. For Regression:

1.  **Choose K:** Same as above.
2.  **Calculate Distances:** Same as above.
3.  **Find K Nearest Neighbors:** Same as above.
4.  **Average/Median:** The predicted value for the new data point is typically the **average** (or sometimes the median) of the target values of its `K` nearest neighbors.

## Key Components

### 1. Distance Metrics

The choice of distance metric is crucial. Common metrics include:

* **Euclidean Distance (L2 norm):** The straight-line distance between two points.
    `d(p, q) = sqrt(Σ(pᵢ - qᵢ)²)`
* **Manhattan Distance (L1 norm):** The sum of the absolute differences of their coordinates.
    `d(p, q) = Σ|pᵢ - qᵢ|`
* **Minkowski Distance:** A generalization: `(Σ|pᵢ - qᵢ|^m)^(1/m)`
    * `m=1`: Manhattan
    * `m=2`: Euclidean

**Crucial Note on Feature Scaling:** KNN is highly sensitive to the scale of features because distance calculations can be dominated by features with larger numerical ranges. **It is almost always essential to scale features (e.g., using Standardization or Min-Max scaling) before applying KNN.**

### 2. Choosing the Value of 'K' (Hyperparameter)

`K` is a critical hyperparameter that needs careful tuning:

* **Small `K` (e.g., K=1):**
    * Model is sensitive to noise and outliers (high variance).
    * Decision boundaries can be very complex and irregular.
* **Large `K`:**
    * Model becomes smoother, less sensitive to noise (lower variance).
    * Can lead to underfitting if `K` is too large (high bias), as it might oversmooth local patterns.
* **Finding Optimal `K`:** Typically done using **cross-validation** on the training set, evaluating model performance for different `K` values.

## Pros and Cons

**Pros:**
* **Simple to understand and implement.**
* **No explicit training phase** (or very fast, just stores data).
* **Naturally handles multi-class classification.**
* **Flexible:** Can learn complex, non-linear decision boundaries (for classification) or fit non-linear data (for regression).
* **Non-parametric:** Makes no assumptions about the data distribution.

**Cons:**
* **Computationally expensive at prediction time:** Requires calculating distances to all training points for each new prediction. This can be slow for large datasets (though optimizations like KD-trees or Ball trees exist).
* **Requires a lot of memory:** Needs to store the entire training dataset.
* **Sensitive to feature scaling.**
* **Sensitive to irrelevant or redundant features:** These can distort distance calculations.
* **"Curse of Dimensionality":** Performance tends to degrade in high-dimensional spaces as the concept of "nearness" becomes less meaningful.
* **Performance depends heavily on the choice of `K` and the distance metric.**

## Implementation with Scikit-learn

Scikit-learn provides easy-to-use implementations:

* **For Classification:** `sklearn.neighbors.KNeighborsClassifier`
* **For Regression:** `sklearn.neighbors.KNeighborsRegressor`

**Key Parameters:**
* `n_neighbors`: The value of `K`.
* `weights`:
    * `'uniform'` (default): All K neighbors have an equal vote/contribution.
    * `'distance'`: Closer neighbors have more influence than farther ones.
* `metric`: The distance metric to use (e.g., `'euclidean'`, `'manhattan'`, `'minkowski'`).
* `p`: Parameter for the Minkowski metric (e.g., `p=1` for Manhattan, `p=2` for Euclidean).

**Workflow Example:**
```python
from sklearn.neighbors import KNeighborsClassifier # or KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# 1. Load and split your data (X_train_raw, X_test_raw, y_train, y_test)
# ...

# 2. Scale features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train_raw)
# X_test = scaler.transform(X_test_raw)

# 3. Tune K using GridSearchCV (example for classifier)
# param_grid = {'n_neighbors': np.arange(1, 21)}
# knn_gscv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
# knn_gscv.fit(X_train, y_train)
# best_k = knn_gscv.best_params_['n_neighbors']
# best_model = knn_gscv.best_estimator_

# 4. Or, use a specific K
# best_k = 5 # example
# best_model = KNeighborsClassifier(n_neighbors=best_k)
# best_model.fit(X_train, y_train) # For KNN, fit() primarily stores X_train and y_train

# 5. Make predictions
# predictions = best_model.predict(X_test)

# 6. Evaluate
# For classification: accuracy_score, confusion_matrix, classification_report
# For regression: mean_squared_error, r2_score