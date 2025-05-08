# Topic 14: Ensemble Methods - Bagging & Random Forests

## 1. Overview of Ensemble Learning

Ensemble learning is a machine learning technique where multiple individual models (often called "base learners" or "weak learners") are combined to produce a single, more robust, and often more accurate predictive model. The core idea is that a "wisdom of the crowd" approach can outperform any single constituent model by leveraging their diverse strengths and mitigating individual weaknesses.

**Key Goals of Ensemble Learning:**
* **Reduce Variance:** Smooth out predictions and make the model less sensitive to the specifics of the training data (e.g., Bagging, Random Forests).
* **Reduce Bias:** Improve the model's ability to capture the underlying true signal (e.g., Boosting).
* **Improve Overall Predictive Performance:** Achieve higher accuracy and better generalization.

For ensembles to be effective, base learners should ideally be **accurate** (better than random guessing) and **diverse** (make different types of errors).

## 2. Bagging (Bootstrap Aggregating)

Bagging is an ensemble technique designed primarily to **reduce the variance** of models that are prone to overfitting, such as decision trees.

**How Bagging Works:**
1.  **Bootstrap Sampling:** Create $B$ different training datasets (bootstrap samples) by sampling *with replacement* from the original training dataset. Each bootstrap sample typically has the same size as the original dataset.
2.  **Train Base Learners:** Train an independent base learner (e.g., a decision tree) on each of the $B$ bootstrap samples. This results in $B$ different models.
3.  **Aggregate Predictions:**
    * **For Classification:** The final prediction is determined by a **majority vote** among the predictions of the $B$ models.
    * **For Regression:** The final prediction is typically the **average** of the predictions from the $B$ models.

**Key Benefits of Bagging:**
* **Variance Reduction:** Averaging the predictions of multiple models trained on different data subsets helps to smooth out noise and reduce the overall variance of the ensemble.
* **Out-of-Bag (OOB) Evaluation:** Since each bootstrap sample omits some original training instances (on average ~36.8%), these "out-of-bag" samples can be used to get an unbiased estimate of the ensemble's performance without needing a separate validation set. Scikit-learn's `BaggingClassifier` and `RandomForestClassifier` support this via `oob_score=True`.

**Scikit-learn:** `sklearn.ensemble.BaggingClassifier`, `sklearn.ensemble.BaggingRegressor`.

## 3. Random Forests

Random Forests are a specific and highly effective type of ensemble method that uses **Decision Trees** as base learners and extends the principles of Bagging.

**How Random Forests Work:**
1.  **Bootstrap Sampling (Bagging):** Like Bagging, each tree in the forest is trained on a bootstrap sample of the original training data.
2.  **Random Feature Subspace (Feature Randomness at Each Split):** This is the key addition that distinguishes Random Forests. When building each individual decision tree, at each node, instead of considering all available features to find the best split, only a **random subset of features** is considered.
    * A new random subset of features is chosen for every split point in every tree.
    * The number of features to consider (`max_features`) is a tunable hyperparameter (e.g., $\sqrt{p}$ for classification, $p/3$ for regression, where $p$ is the total number of features).
3.  **Aggregate Predictions:**
    * **For Classification:** Majority vote from all trees.
    * **For Regression:** Average of predictions from all trees.

**Why Random Feature Subspace is Beneficial:**
* **Decorrelates Trees:** Prevents a few strong features from dominating all trees, leading to more diverse individual trees.
* **Increases Diversity:** More diverse trees generally lead to a better ensemble.
* **Further Reduces Variance:** The increased diversity often results in greater variance reduction compared to just Bagging decision trees.

**Key Hyperparameters for Random Forests:**
* `n_estimators`: The number of trees in the forest. More trees are generally better up to a point, after which performance plateaus.
* `max_features`: The number (or fraction) of features to consider at each split. Crucial for controlling tree diversity.
* Decision Tree specific parameters (e.g., `max_depth`, `min_samples_split`, `min_samples_leaf`): These control the complexity of individual trees. In Random Forests, trees are often grown deep, as the ensemble averaging mitigates individual tree overfitting.
* `bootstrap`: Whether to use bootstrap sampling (default is `True`).
* `oob_score`: Whether to use out-of-bag samples for performance estimation.

**Reduced Overfitting:**
Random Forests are much less prone to overfitting than single decision trees because:
* Averaging/voting over many diverse trees cancels out the noise learned by individual trees.
* Bootstrap sampling and random feature selection ensure tree diversity.

**Feature Importance:**
Random Forests can provide an estimate of the importance of each feature, typically calculated as the **Mean Decrease in Impurity (MDI)**. This measures how much each feature contributes to reducing impurity (e.g., Gini impurity or MSE) across all splits in all trees in the forest.
* Accessible via the `feature_importances_` attribute in Scikit-learn models.
* Useful for understanding data and for feature selection.

**Scikit-learn:** `sklearn.ensemble.RandomForestClassifier`, `sklearn.ensemble.RandomForestRegressor`.

## Summary: Bagging vs. Random Forests

* **Bagging** is a general technique that can be used with various base estimators.
* **Random Forests** are a specific implementation of Bagging that uses decision trees as base estimators and adds the crucial step of random feature selection at each split to further enhance diversity and performance. Random Forests are generally preferred over simple Bagging of decision trees.
