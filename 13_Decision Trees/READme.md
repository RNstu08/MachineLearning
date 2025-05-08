# Topic 13: Decision Trees

## Overview

Decision Trees are versatile supervised learning algorithms used for both **classification** and **regression** tasks. They predict the value of a target variable by learning simple decision rules inferred from the data features, represented as a tree-like structure.

Imagine a flowchart where each internal node represents a "test" on a feature, each branch represents an outcome of the test, and each leaf node represents a class label (for classification) or a continuous value (for regression). One of the key strengths of decision trees is their **interpretability** (they are often called "white box" models).

## 1. Tree Structure

A decision tree has the following components:

* **Root Node:** The topmost node representing the entire dataset, where the first decision split occurs.
* **Internal Nodes (Decision Nodes):** Nodes that represent a test on a specific feature (e.g., "Is Petal Width < 0.8 cm?"). Each internal node has outgoing branches.
* **Branches (Edges):** Represent the outcome of a test (e.g., "Yes" or "No," or a value range). They connect nodes.
* **Leaf Nodes (Terminal Nodes):** Nodes that do not split further. They provide the final output:
    * For classification: A class label (e.g., "Iris-setosa").
    * For regression: A continuous value (typically the average of target values in that leaf).
* **Depth:** The length of the longest path from the root to a leaf.

**Example (Conceptual Text Diagram):**

Is Outlook == Sunny?
|--- Yes: Is Humidity <= 70?
|      |--- Yes: Leaf (Play Tennis: Yes)
|      |--- No:  Leaf (Play Tennis: No)
|--- No: (Outlook == Overcast or Rainy) ...


## 2. How Decision Trees Work (Building the Tree)

Decision trees are typically built using a **greedy, top-down, recursive partitioning** algorithm:
1.  Start with all training samples at the root node.
2.  Find the "best" feature and the best split point (threshold for numerical, category for categorical) that divides the data into subsets that are as "pure" as possible regarding the target variable.
3.  Create child nodes for these subsets.
4.  Repeat the process recursively for each child node.
5.  Stop when a stopping criterion is met (e.g., node is pure, max depth reached, min samples in node).

This is a **greedy** approach because it makes the locally optimal decision at each step without guaranteeing a globally optimal tree.

## 3. Splitting Criteria (Measuring Purity/Impurity)

To find the "best" split, the algorithm uses a criterion to measure the homogeneity (purity) or heterogeneity (impurity) of the target variable within the resulting subsets.

### For Classification Trees:

* **Gini Impurity:** Measures the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the class distribution in the node.
    `Gini(t) = 1 - Σ [p(Cₖ|t)]²`
    (0 for pure node, 0.5 for 2 classes with 50/50 split). The goal is to minimize Gini impurity.
* **Entropy / Information Gain:**
    * Entropy measures the disorder or uncertainty in a node.
        `Entropy(t) = - Σ p(Cₖ|t) * log₂(p(Cₖ|t))`
        (0 for pure node, 1 for 2 classes with 50/50 split using log base 2).
    * Information Gain is the reduction in entropy achieved by a split. The algorithm maximizes Information Gain.

### For Regression Trees:

* **Mean Squared Error (MSE):** The algorithm tries to find splits that minimize the variance (or MSE) within the resulting child nodes. The prediction at a leaf is the average of target values in that leaf.
    `MSE(t) = (1/Nₜ) * Σ (yᵢ - ȳₜ)²`
* **Mean Absolute Error (MAE):** Can also be used, less sensitive to outliers.

## 4. Controlling Tree Growth & Preventing Overfitting

Unconstrained decision trees can easily overfit the training data by creating very complex structures. Techniques to control this include:

* **Pre-pruning (Early Stopping):** Stopping tree growth before it perfectly fits the training data.
    * **Hyperparameters:**
        * `max_depth`: Maximum depth of the tree.
        * `min_samples_split`: Minimum samples required in a node to consider splitting it.
        * `min_samples_leaf`: Minimum samples required in each child (leaf) node after a split.
        * `max_leaf_nodes`: Limit the maximum number of leaf nodes.
        * `min_impurity_decrease`: Minimum impurity reduction required for a split.
* **Post-pruning:** Grow the tree fully, then remove (prune) branches that provide little predictive power or increase complexity unnecessarily.
    * **Cost-Complexity Pruning (`ccp_alpha` in Scikit-learn):** Prunes the tree based on a complexity parameter alpha, balancing error and tree size.

## 5. Visualizing Trees

Visualization is key to understanding decision trees:
* **`sklearn.tree.plot_tree()`:** Uses Matplotlib to render the tree. Shows split conditions, impurity, samples, class distribution/value per node.
* **`sklearn.tree.export_graphviz()`:** Exports the tree to a `.dot` file, which can be converted to an image using Graphviz software for high-quality, customizable diagrams.

## 6. Advantages and Disadvantages

**Advantages:**
* **Simple to understand and interpret (White Box Model).**
* **Easy to visualize.**
* **Requires little data preparation** (e.g., no strict need for feature scaling for basic trees).
* Can handle both **numerical and categorical data** (though Scikit-learn requires numerical input).
* Can capture **non-linear relationships** and feature interactions.
* Performs **implicit feature selection**.
* **Fast prediction** once trained.

**Disadvantages:**
* **Prone to overfitting** if not pruned or constrained.
* **Instability (high variance):** Small changes in data can lead to very different trees.
* **Greedy algorithm:** May not find the globally optimal tree.
* Can be **biased** if classes are imbalanced.
* Struggles with some relationships (e.g., **diagonal decision boundaries** are approximated by "staircases").
* Regression trees produce **piecewise constant predictions**.

## 7. Implementation with Scikit-learn

* **For Classification:** `sklearn.tree.DecisionTreeClassifier`
* **For Regression:** `sklearn.tree.DecisionTreeRegressor`

**Key Hyperparameters in Scikit-learn:**
* `criterion`: The function to measure the quality of a split ("gini" or "entropy" for classification; "squared_error", "friedman_mse", "absolute_error", "poisson" for regression).
* `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `min_impurity_decrease`, `ccp_alpha`.
* `random_state`: For reproducibility.

**General Workflow:**
1.  Load and split data.
2.  Instantiate `DecisionTreeClassifier` or `DecisionTreeRegressor` with desired (or tuned) hyperparameters.
3.  Train the model: `model.fit(X_train, y_train)`.
4.  Make predictions: `model.predict(X_test)`.
5.  Evaluate performance.
6.  Visualize the tree to understand its decision logic.
7.  Use techniques like `GridSearchCV` to tune hyperparameters for optimal performance and generalization.