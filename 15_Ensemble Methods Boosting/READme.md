# Topic 15: Ensemble Methods - Boosting

## 1. Overview of Boosting

Boosting is a powerful ensemble learning technique that builds a strong predictive model by sequentially combining multiple "weak learners" (models that perform slightly better than random guessing, often simple decision trees). Unlike Bagging, which builds models in parallel to reduce variance, Boosting builds models iteratively, with each new model attempting to correct the errors made by its predecessors. This process primarily aims to **reduce bias** and can also reduce variance, leading to highly accurate models.

**Core Idea:**
* Models are trained sequentially.
* Each subsequent model focuses more on the training instances that were difficult for the previous models (e.g., by re-weighting misclassified samples or fitting to residual errors).
* The final prediction is a weighted combination of the predictions from all weak learners.

## 2. AdaBoost (Adaptive Boosting)

AdaBoost is one of the earliest and most fundamental boosting algorithms.

**How it Works (Conceptual):**
1.  **Initialize Sample Weights:** All training samples start with equal weights.
2.  **Iteratively Train Weak Learners:**
    a.  Train a weak learner (e.g., a decision stump) on the weighted training data.
    b.  **Calculate Weighted Error:** Determine the error rate of this learner on the weighted data.
    c.  **Calculate Learner Weight:** Assign a weight to the learner based on its accuracy (more accurate learners get higher weights).
    d.  **Update Sample Weights:** Increase the weights of misclassified samples and decrease the weights of correctly classified samples. This makes the next learner focus more on the "hard" examples.
3.  **Final Prediction:** A weighted majority vote (for classification) or weighted sum (for regression) of the predictions from all weak learners, using their calculated learner weights.

**Key Focus:** Adapts by giving more weight to misclassified instances.

## 3. Gradient Boosting Machines (GBM)

Gradient Boosting is a more generalized boosting framework.

**How it Works (Conceptual):**
1.  **Initialize Model:** Start with a simple initial prediction (e.g., the mean of the target for regression).
2.  **Iteratively Train Weak Learners:**
    a.  **Compute Pseudo-Residuals:** Calculate the difference between the actual values and the current ensemble's predictions. For general loss functions, these are the negative gradients of the loss function with respect to the current predictions.
    b.  **Train a Weak Learner:** Fit a new weak learner (typically a regression tree) to these pseudo-residuals. The goal of this tree is to predict the errors of the current ensemble.
    c.  **Update Ensemble Model:** Add the predictions of this new tree (scaled by a **learning rate/shrinkage factor `η`**) to the current ensemble's predictions.
3.  **Final Prediction:** The sum of the initial prediction and the scaled predictions of all sequentially added trees.

**Key Features:**
* **Loss Functions:** Can be used with various differentiable loss functions (e.g., squared error for regression, deviance/log-loss for classification).
* **Weak Learners:** Typically shallow decision trees (regression trees).
* **Shrinkage (Learning Rate `η`):** A crucial hyperparameter that scales the contribution of each tree. Smaller values (e.g., 0.01-0.1) require more trees (`n_estimators`) but often lead to better generalization.

## 4. Advanced, Highly Efficient GBM Implementations

Standard GBMs are powerful, but specialized libraries have optimized them for speed, performance, and features.

### a. XGBoost (Extreme Gradient Boosting)

* **Key Features:**
    * **Regularization:** Built-in L1 (Lasso) and L2 (Ridge) regularization on leaf weights and tree complexity.
    * **Sparsity Awareness:** Efficiently handles missing values and sparse data.
    * **Parallel Processing:** Parallelizes tree construction (e.g., finding splits).
    * **Cache Awareness & Hardware Optimization.**
    * **Advanced Tree Pruning:** Uses `gamma` (min_split_loss) for pruning.
    * **Built-in Cross-Validation & Early Stopping.**

### b. LightGBM (Light Gradient Boosting Machine)

* **Key Features (Focus on Speed & Efficiency):**
    * **Gradient-based One-Side Sampling (GOSS):** Focuses training on instances with larger gradients (more "wrong").
    * **Exclusive Feature Bundling (EFB):** Bundles mutually exclusive sparse features to reduce feature dimensionality.
    * **Leaf-wise Tree Growth:** Grows trees by splitting the leaf that yields the largest loss reduction (can be faster and more accurate but needs care with `max_depth` or `num_leaves` to avoid overfitting on small datasets).
    * **Histogram-based Algorithms:** For fast and memory-efficient split finding.
    * **Direct Categorical Feature Support.**

### c. CatBoost (Categorical Boosting)

* **Key Features (Focus on Categorical Data & Robustness):**
    * **Superior Categorical Feature Handling:** Uses advanced techniques like "ordered target statistics" and "ordered boosting" to convert categorical features to numerical values effectively, preventing target leakage.
    * **Ordered Boosting:** A permutation-based approach to reduce prediction shift and improve generalization.
    * **Symmetric (Oblivious) Trees:** Often builds trees where the same splitting criterion is used for all nodes at the same level, leading to balanced, less complex trees and faster predictions.
    * **Robustness to Overfitting & Good Defaults:** Often performs well with default parameters.

## 5. Key Hyperparameters (Common & Advanced)

* **`n_estimators` (or `iterations`, `num_boost_round`):** Number of trees/boosting rounds. More is often better up to a point, then diminishing returns or overfitting.
* **`learning_rate` (or `eta`):** Shrinkage parameter. Scales the contribution of each tree. Smaller values require more trees but often generalize better. Typical: 0.01-0.3.
* **Tree-Specific Parameters (for weak learners):**
    * `max_depth`: Maximum depth of individual trees. Boosting often uses shallow trees (e.g., 3-8).
    * `min_samples_leaf` (or `min_child_samples`, `min_data_in_leaf`): Minimum samples in a leaf node.
    * `min_child_weight` (XGBoost): Minimum sum of instance hessian needed in a child.
    * `num_leaves` (LightGBM): Max number of leaves in a tree (key for leaf-wise growth).
* **Sampling Parameters:**
    * `subsample` (or `bagging_fraction`): Fraction of training instances to sample for each tree.
    * `colsample_bytree` (XGBoost) / `feature_fraction` (LightGBM): Fraction of features to sample for each tree.
* **Regularization Parameters:**
    * `reg_alpha` (L1): L1 regularization term.
    * `reg_lambda` (L2): L2 regularization term.
    * `gamma` (XGBoost, `min_split_loss`): Minimum loss reduction to make a split.
* **Objective/Loss Function:** Specifies the task (e.g., `binary:logistic`, `reg:squarederror`, `multi:softmax`).
* **Categorical Feature Handling (LightGBM, CatBoost):** Parameters to specify categorical features for specialized internal handling.

**Tuning Strategy:** Often involves setting a small `learning_rate`, then finding optimal `n_estimators` using early stopping with a validation set, followed by tuning tree structure and regularization parameters using `GridSearchCV`, `RandomizedSearchCV`, or Bayesian optimization.

## 6. Advantages of Boosting

* **High Predictive Accuracy:** Often achieve state-of-the-art performance on many tasks.
* **Handles Different Types of Data:** Can work with numerical and categorical features (especially CatBoost and LightGBM).
* **Good at Handling Complex Relationships:** The additive nature allows capturing non-linearities and feature interactions.
* **Feature Importance:** Provides estimates of feature importance.
* **Robustness (Advanced Implementations):** Libraries like XGBoost, LightGBM, and CatBoost include many features to prevent overfitting and improve stability.

## 7. Disadvantages of Boosting

* **Sensitive to Noisy Data and Outliers:** Since boosting focuses on "hard" examples, noisy data or outliers can disproportionately influence the model if not handled carefully.
* **Computationally Intensive Training:** Training can be slow, especially with many trees, deep trees, or large datasets, although advanced libraries have significantly optimized this.
* **Prone to Overfitting (if not tuned carefully):** While designed to reduce bias, boosting models can still overfit if the number of trees is too high, the learning rate is too large, or individual trees are too complex without proper regularization.
* **Less Interpretable than Single Trees:** The final model is an ensemble of many trees, making it harder to interpret directly compared to a single decision tree.
* **More Hyperparameters to Tune:** Compared to simpler models, boosting algorithms (especially the advanced ones) have many hyperparameters that may require careful tuning.

## 8. Libraries

* **Scikit-learn:** `AdaBoostClassifier`, `AdaBoostRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`.
* **Specialized Libraries:** `xgboost`, `lightgbm`, `catboost` (these are usually preferred for top performance and features).