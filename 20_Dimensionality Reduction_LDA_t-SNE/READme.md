# Topic 20: Dimensionality Reduction - Other Techniques (Briefly)

Besides Principal Component Analysis (PCA), there are other notable techniques for dimensionality reduction, each with different goals and characteristics. This section briefly covers Linear Discriminant Analysis (LDA) and t-Distributed Stochastic Neighbor Embedding (t-SNE).

## 1. Linear Discriminant Analysis (LDA)

* **Type:** **Supervised** dimensionality reduction technique.
    * Unlike PCA (unsupervised), LDA utilizes class labels during the reduction process.
* **Primary Goal:** To find a lower-dimensional subspace that **maximizes the separability between classes**.
    * It projects data onto axes (linear discriminants) that maximize the ratio of between-class variance to within-class variance.
* **How it Works (Conceptual):**
    1.  Computes the mean for each class.
    2.  Calculates the within-class scatter matrix ($S_W$ - how spread out each class is internally) and the between-class scatter matrix ($S_B$ - how spread out the class means are from each other).
    3.  Solves an eigenvalue problem (related to $S_W^{-1}S_B$) to find the linear discriminants (eigenvectors) that maximize class separation.
* **Number of Components:** The maximum number of linear discriminants is $c-1$, where $c$ is the number of classes. For binary classification, LDA finds 1 discriminant.
* **Use Cases:**
    * Primarily used as a feature extraction technique **before classification tasks** to improve class separability and potentially model performance.
    * Can be used for visualization (if reduced to 2 or 3 components) to show class separation.
* **Contrast with PCA:**
    * PCA finds directions of maximum variance (unsupervised).
    * LDA finds directions of maximum class separability (supervised).
* **Assumptions:** Assumes features are normally distributed and classes have identical covariance matrices (though often works reasonably well if not perfectly met).
* **Scikit-learn:** `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`

## 2. t-Distributed Stochastic Neighbor Embedding (t-SNE)

* **Type:** **Non-linear** dimensionality reduction technique.
* **Primary Goal:** **Visualization** of high-dimensional datasets in low-dimensional space (typically 2D or 3D). It excels at revealing local structure and clusters.
* **How it Works (Conceptual):**
    1.  **High-Dimensional Similarities:** Models pairwise similarities between high-dimensional data points as conditional probabilities using a Gaussian distribution (similar points have higher probability).
    2.  **Low-Dimensional Similarities:** Models pairwise similarities between corresponding low-dimensional points using a t-distribution (which has heavier tails, helping to separate dissimilar points and reduce crowding).
    3.  **Minimizing Divergence:** Iteratively adjusts the positions of points in the low-dimensional embedding to minimize the Kullback-Leibler (KL) divergence between the high-dimensional and low-dimensional similarity distributions.
* **Key Characteristics & Considerations:**
    * **Preserves Local Structure:** Excellent at showing which points are "neighbors" in high dimensions, often revealing distinct clusters clearly.
    * **Global Structure Not Always Meaningful:** The relative sizes of clusters and distances *between* clusters in a t-SNE plot should not be over-interpreted. Focus on which points group together.
    * **Computationally Intensive:** Can be slow on large datasets.
    * **Sensitive to Hyperparameters:**
        * `perplexity`: Roughly related to the number of nearest neighbors considered (typical values: 5-50). Output can vary significantly with perplexity.
        * `n_iter`: Number of optimization iterations.
        * `learning_rate`: Step size for optimization.
        * `init`: Initialization method ('random', 'pca'). 'pca' is often more stable.
    * **Not for Preprocessing for ML Models:** Primarily a visualization tool. The transformed features are generally not suitable for input to most supervised learning models due to the complex, non-linear mapping and lack of global structure preservation.
* **Use Cases:** Visualizing complex datasets in fields like bioinformatics, image analysis, and NLP to explore potential clusters or manifold structures.
* **Scikit-learn:** `sklearn.manifold.TSNE`

## Summary Table

| Feature          | PCA (Principal Component Analysis)                 | LDA (Linear Discriminant Analysis)              | t-SNE (t-Distributed Stochastic Neighbor Embedding) |
| :--------------- | :------------------------------------------------- | :---------------------------------------------- | :------------------------------------------------ |
| **Type** | Unsupervised, Linear                             | Supervised, Linear                              | Unsupervised, Non-linear                          |
| **Goal** | Maximize variance                                  | Maximize class separability                     | Preserve local neighborhood structure             |
| **Uses Labels?** | No                                                 | Yes                                             | No                                                |
| **Primary Use** | Dimensionality reduction for ML, visualization     | Dimensionality reduction for classification     | Visualization of high-dimensional data            |
| **Output** | Principal Components (uncorrelated)                | Linear Discriminants (max class separation)     | Low-dimensional embedding (2D/3D)                 |
| **Scaling** | Crucial (Standardization)                        | Often recommended                               | Often recommended                                 |

---