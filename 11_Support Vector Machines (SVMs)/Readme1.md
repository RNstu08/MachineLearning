# Topic 11: Support Vector Machines (SVMs)

## Overview

Support Vector Machines (SVMs) are a powerful and versatile class of supervised learning algorithms used for **classification (SVC)**, **regression (SVR)**, and outlier detection. The core idea behind SVMs is to find an optimal hyperplane that best separates data points into different classes (for classification) or best fits the data within a certain margin (for regression).

SVMs are particularly effective in high-dimensional spaces, can model non-linear relationships using the "kernel trick," and are known for good generalization performance due to the concept of margin maximization.

**Key Prerequisite: Feature Scaling**
Feature scaling (e.g., using `StandardScaler`) is crucial for SVMs, as their underlying optimization and distance calculations (implicit in kernels like RBF) are sensitive to the scale of input features.

## 1. Support Vector Classification (SVC)

### a. Linear SVC: Maximal Margin Classifier

* **Goal:** For linearly separable data, find the hyperplane that not only separates the classes but also has the **largest possible margin** (distance) between the hyperplane and the nearest data points (support vectors) of any class.
* **Margin:** The "street" between the classes. A wider margin generally leads to better generalization.
* **Support Vectors:** The data points that lie closest to the decision boundary (on the edges of the margin). These points "support" or define the hyperplane.
* **Hard Margin:** Assumes perfect linear separability. Sensitive to outliers.
* **Soft Margin (More Practical):** Allows for some misclassifications or points within the margin to handle noise and non-separable data.
    * **Hyperparameter `C` (Regularization):** Controls the trade-off between maximizing the margin and minimizing margin violations.
        * Small `C`: Wider margin, more margin violations allowed (stronger regularization, softer margin).
        * Large `C`: Narrower margin, fewer margin violations (weaker regularization, harder margin, can overfit).

### b. Non-Linear SVC: The Kernel Trick

* **Problem:** Real-world data is often not linearly separable in its original feature space.
* **Solution:** SVMs can map the input features into a higher-dimensional space where a linear separation might be possible. This mapping is done implicitly using the **kernel trick**.
* **Kernel Trick:** Allows computation of dot products in a high-dimensional feature space without explicitly transforming the data, using a **kernel function** $K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$.
* **Common Kernels:**
    * **`linear`**: $K(x_i, x_j) = x_i^T x_j$. (Standard linear SVM).
    * **`poly` (Polynomial)**: $K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$.
        * Hyperparameters: `degree (d)`, `gamma ($\gamma$)`, `coef0 (r)`.
    * **`rbf` (Radial Basis Function)**: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$. (Popular default).
        * Hyperparameter: `gamma ($\gamma$)`. Small gamma = wider influence (smoother boundary); Large gamma = narrower influence (can overfit).
    * **`sigmoid`**: $K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)$.
        * Hyperparameters: `gamma ($\gamma$)`, `coef0 (r)`.

## 2. Support Vector Regression (SVR)

* **Goal:** Instead of separating classes, SVR aims to find a function that fits the data such that most data points lie *within* a specified margin (the **$\epsilon$-insensitive tube**) around this function. The model also tries to be as "flat" (simple) as possible.
* **$\epsilon$-Insensitive Tube:** Defined by the hyperparameter `epsilon ($\epsilon$)`. Errors for points falling *inside* this tube (width $2\epsilon$) are ignored in the loss function.
* **Support Vectors (in SVR):** Data points that lie on the boundary of or outside the $\epsilon$-insensitive tube.
* **Hyperparameters:**
    * `C`: Regularization parameter (trade-off between model complexity/flatness and allowing points outside the tube).
    * `epsilon ($\epsilon$)`: Defines the width of the "no-penalty" tube.
    * Kernel parameters (`kernel`, `gamma`, `degree`, `coef0`) also apply.

## 3. Key Hyperparameters (Consolidated)

* **`C`**: Regularization parameter (for both SVC and SVR). Inverse of regularization strength.
* **`kernel`**: Type of kernel to use ('linear', 'poly', 'rbf', 'sigmoid').
* **`gamma`**: Kernel coefficient for 'rbf', 'poly', 'sigmoid'. Controls the influence of a single training example.
* **`degree`**: Degree for the 'poly' kernel.
* **`coef0`**: Independent term in 'poly' and 'sigmoid' kernels.
* **`epsilon`**: (SVR only) Defines the width of the $\epsilon$-insensitive tube.

Tuning these hyperparameters, typically using **cross-validation** (e.g., `GridSearchCV`), is crucial for SVM performance.

## 4. Mathematical Intuition (Simplified)

* **SVC:** A constrained optimization problem. Aims to minimize model complexity (represented by $||w||^2$, where $w$ is the weight vector normal to the hyperplane) subject to the constraint that data points are correctly classified with a margin of at least 1 (for normalized data). This leads to maximizing the margin.
* **SVR:** Also a constrained optimization problem. Aims to minimize model complexity ($||w||^2$) subject to the constraint that the errors for most data points $|y_i - f(x_i)|$ are within $\pm\epsilon$.

## 5. Implementation with Scikit-learn

* **For Classification:** `sklearn.svm.SVC`
* **For Regression:** `sklearn.svm.SVR`

**General Workflow:**
1.  Load and preprocess data.
2.  **Scale features** (e.g., `StandardScaler`). This is highly recommended.
3.  Split data into training and testing sets.
4.  Instantiate `SVC` or `SVR` with chosen/tuned hyperparameters.
5.  Train the model: `model.fit(X_train_scaled, y_train)`.
6.  Make predictions: `model.predict(X_test_scaled)`.
7.  Evaluate performance (classification metrics for SVC, regression metrics for SVR).
8.  Hyperparameter tuning is often done using `GridSearchCV`.

## 6. Pros and Cons of SVMs

**Pros:**
* Effective in high-dimensional spaces (even when features > samples).
* Memory efficient in its decision function (uses only support vectors).
* Versatile due to different kernel functions, allowing modeling of non-linear relationships.
* Good generalization performance due to margin maximization / $\epsilon$-insensitivity.
* Relatively robust to outliers (with soft margin / $\epsilon$-tube).

**Cons:**
* Computationally intensive to train on very large datasets.
* Performance is highly dependent on the choice of kernel and hyperparameter tuning (`C`, `gamma`, etc.).
* Less interpretable ("black box") for non-linear kernels compared to simpler models.
* SVC does not directly provide probability estimates (requires an extra calibration step).

---
This README provides a comprehensive overview of Support Vector Machines.