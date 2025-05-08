# Topic 19: Dimensionality Reduction - Principal Component Analysis (PCA)

## 1. Overview of Dimensionality Reduction

Dimensionality reduction is the process of reducing the number of features (dimensions) in a dataset while aiming to preserve essential information and structure. High-dimensional data can lead to issues like increased computational cost, model overfitting, and the "curse of dimensionality" (where data becomes sparse and patterns harder to find).

**Main Approaches:**
* **Feature Selection:** Selecting a subset of the original features.
* **Feature Extraction:** Creating new, fewer features by combining or transforming the original ones. PCA is a feature extraction technique.

## 2. Principal Component Analysis (PCA)

PCA is a widely used **linear dimensionality reduction** technique. It transforms the data into a new coordinate system such that the greatest variance by any projection of the data comes to lie on the first coordinate (called the first principal component, PC1), the second greatest variance on the second coordinate (PC2), and so on. The principal components are **uncorrelated** with each other.

**Goal:**
* To identify the directions (principal components) in the feature space along which the data varies the most.
* To project the original data onto a lower-dimensional subspace formed by these principal components, thereby reducing dimensionality while retaining maximum variance (information).

## 3. Key Concepts in PCA

* **Principal Components (PCs):** New, uncorrelated features that are linear combinations of the original features. They are ordered by the amount of variance they explain (PC1 explains the most).
* **Covariance Matrix:** A $p \times p$ matrix (where $p$ is the number of original features) that describes the variance of each feature and the covariance between pairs of features. PCA analyzes this structure.
* **Eigenvectors and Eigenvalues:**
    * The **eigenvectors** of the covariance matrix define the directions of the principal components.
    * The corresponding **eigenvalues** represent the amount of variance captured by each principal component. Larger eigenvalues mean more variance explained.
* **Explained Variance Ratio:** The percentage of the total variance in the original dataset that is captured by each principal component.
    `Explained Variance Ratio (PCᵢ) = Eigenvalue of PCᵢ / Sum of all Eigenvalues`
* **Cumulative Explained Variance:** The sum of the explained variance ratios of the principal components kept. This helps decide how many components are needed to retain a desired percentage of the total variance.

## 4. How PCA Works (Mathematical Steps)

1.  **Standardize the Data (Feature Scaling):** This is a **crucial** step. PCA is sensitive to feature scales, so features must be standardized to have zero mean and unit variance (e.g., using `StandardScaler`).
2.  **Compute the Covariance Matrix:** Calculate the covariance matrix of the standardized features.
3.  **Perform Eigendecomposition (or SVD):** Find the eigenvectors and eigenvalues of the covariance matrix. (In practice, Singular Value Decomposition - SVD - is often used on the data matrix for numerical stability and efficiency).
4.  **Sort Eigenvectors by Eigenvalues:** Sort the eigenvectors in descending order of their corresponding eigenvalues. The eigenvector with the largest eigenvalue is PC1, the next is PC2, and so on.
5.  **Select Principal Components:** Choose the top $k$ eigenvectors (principal components) that capture a significant amount of variance.
6.  **Transform the Data:** Project the original standardized data onto the subspace defined by the selected $k$ principal components. The result is a new dataset with $k$ dimensions.
    `X_pca = X_scaled @ W` (where `W` is the matrix of the top $k$ eigenvectors).

## 5. Choosing the Number of Principal Components (k)

* **Explained Variance Ratio Plot (Cumulative):** Plot the cumulative explained variance against the number of components. Choose $k$ where a desired percentage of variance is explained (e.g., 95%).
* **Scree Plot:** Plot the eigenvalues in descending order. Look for an "elbow" point where the eigenvalues start to level off. Components before the elbow are typically kept.
* **Application-Specific Needs:** For visualization, $k=2$ or $k=3$ is chosen.

## 6. Use Cases of PCA

* **Dimensionality Reduction for Machine Learning:** Speeds up training, reduces overfitting, combats the curse of dimensionality.
* **Data Visualization:** Reduces high-dimensional data to 2D or 3D for plotting and visual pattern discovery.
* **Noise Reduction/Denoising:** Discarding low-variance components can filter out noise.
* **Feature Engineering/Extraction:** Creates new, uncorrelated features.
* **Image Compression.**
* **Anomaly Detection.**

## 7. Pros and Cons

**Pros:**
* Reduces dimensionality, leading to computational and memory efficiency.
* Removes multicollinearity (principal components are uncorrelated).
* Can reduce noise.
* Useful for data compression and visualization.
* Unsupervised (does not require target labels).

**Cons:**
* **Information Loss:** Some information is lost during dimensionality reduction.
* **Reduced Interpretability:** Principal components are linear combinations of original features and can be harder to interpret directly.
* **Assumes Linearity:** May not capture non-linear structures well (alternatives: Kernel PCA, t-SNE, UMAP).
* **Sensitive to Feature Scaling:** Standardization is essential.
* **Variance might not always equate to predictive importance** for a specific supervised task.
* Can be influenced by outliers.

## 8. Implementation with Scikit-learn

* Use `sklearn.decomposition.PCA`.

**Key Parameters for `PCA`:**
* `n_components`:
    * Integer: Number of components to keep.
    * Float (between 0.0 and 1.0): Select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified.
    * `None`: All components are kept.
    * `'mle'`: Uses Minka's MLE to guess the dimension.
* `svd_solver`: Method for SVD computation ('auto', 'full', 'arpack', 'randomized').
* `random_state`: For reproducibility with solvers like 'arpack' or 'randomized'.

**General Workflow:**
1.  Load and preprocess data.
2.  **Standardize features** (`StandardScaler`).
3.  Split data (if PCA is part of a supervised learning pipeline, fit PCA *only* on training data).
4.  Instantiate `PCA` with the desired `n_components` (or use explained variance to determine it).
5.  Fit PCA on the (scaled) training data: `pca_model.fit(X_train_scaled)`.
6.  Transform training and test data:
    `X_train_pca = pca_model.transform(X_train_scaled)`
    `X_test_pca = pca_model.transform(X_test_scaled)`
7.  Use the transformed data (`X_train_pca`, `X_test_pca`) for subsequent tasks (e.g., training a classifier, visualization).
8.  Examine `pca_model.explained_variance_ratio_` to understand information retention.

---
