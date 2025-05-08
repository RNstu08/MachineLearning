# Topic 16: Clustering - K-Means

## 1. Overview of Unsupervised Learning & Clustering

* **Unsupervised Learning:** A type of machine learning where the algorithm learns patterns from **unlabeled data** (data without predefined target outcomes). The goal is to discover intrinsic structures, groupings, or relationships within the data.
* **Clustering:** A common unsupervised learning task that involves grouping a set of data points into subsets, called **clusters**.
    * Points within the same cluster should be highly **similar** to each other.
    * Points in different clusters should be highly **dissimilar**.
    * Similarity/dissimilarity is typically measured using distance metrics (e.g., Euclidean distance).

## 2. K-Means Clustering

K-Means is a popular, simple, and efficient **centroid-based** clustering algorithm.

* **Goal:** To partition $N$ data points into $K$ distinct, non-overlapping clusters, where $K$ is a user-specified number. Each data point belongs to the cluster whose mean (centroid) is nearest.
* **Centroids:** The center point of a cluster, calculated as the mean of all data points belonging to that cluster.

### How K-Means Algorithm Works (Iterative Process):

1.  **Initialization:**
    a.  **Choose K:** Specify the desired number of clusters.
    b.  **Initialize Centroids:** Randomly select $K$ data points as initial centroids, or use a smarter method like **K-Means++** (default in Scikit-learn), which tries to spread out initial centroids.
2.  **Assignment Step (Expectation):**
    * For each data point, calculate its distance to all $K$ centroids.
    * Assign the data point to the cluster whose centroid is closest.
3.  **Update Step (Maximization):**
    * Recalculate the position of each of the $K$ centroids by taking the mean of all data points currently assigned to that cluster.
4.  **Repeat:** Iterate between the Assignment Step and Update Step until convergence criteria are met (e.g., centroids no longer move significantly, cluster assignments don't change, or a maximum number of iterations is reached).

## 3. Inertia (Within-Cluster Sum of Squares - WCSS)

* K-Means aims to minimize **Inertia (WCSS)**, which is the sum of squared distances between each data point and its assigned cluster's centroid.
    `Inertia = Σ Σ ||xᵢ - μₖ||²` (sum over all clusters $k$, and all points $x_i$ in cluster $C_k$).
* Lower inertia generally indicates more compact, well-defined clusters.
* Inertia always decreases as $K$ increases. If $K=N$ (number of data points), inertia is 0.

## 4. Choosing the Optimal Number of Clusters (K)

Since $K$ is a hyperparameter, choosing it appropriately is crucial.

* **Elbow Method:**
    1.  Run K-Means for a range of $K$ values.
    2.  Plot Inertia (WCSS) against $K$.
    3.  Look for an "elbow" point in the plot where the rate of decrease in inertia slows down significantly. This point is often considered a good estimate for $K$.
    * Limitation: The elbow can sometimes be ambiguous.
* **Silhouette Analysis:** Measures how similar a point is to its own cluster compared to other clusters. Provides a score between -1 and 1 (higher is better).
* **Domain Knowledge:** Often the most reliable way to choose $K$.

## 5. Key Hyperparameters & Considerations

* **`n_clusters` (K):** The number of clusters to form.
* **`init`:** Method for centroid initialization.
    * `'k-means++'` (default): Smartly selects initial centroids to speed up convergence and improve results.
    * `'random'`: Chooses $K$ observations randomly.
* **`n_init`:** Number of times the K-Means algorithm will be run with different centroid seeds. The final result will be the best output of these runs in terms of inertia. (Default is 10).
* **`max_iter`:** Maximum number of iterations for a single run of K-Means.

## 6. Assumptions and Limitations of K-Means

* **Assumes Spherical/Globular Clusters:** Works best when clusters are roughly round and of similar sizes.
* **Assumes Clusters of Similar Size/Density:** Tends to create clusters of roughly equal size.
* **Sensitive to Feature Scaling:** Distances are affected by feature scales. **Scaling features (e.g., `StandardScaler`) is almost always necessary.**
* **Need to Specify K in Advance:** Choosing K can be challenging.
* **Sensitive to Initial Centroid Placement:** Can converge to local optima if not using K-Means++ or multiple initializations (`n_init`).
* **Struggles with Non-Globular Shapes:** Has difficulty with elongated clusters, clusters of irregular shapes, or concentric circles.
* **Impact of Outliers:** Centroids (being means) can be pulled by outliers.

## 7. Implementation with Scikit-learn

* Use `sklearn.cluster.KMeans`.

**General Workflow:**
1.  Load and preprocess data.
2.  **Scale features** (e.g., `StandardScaler`).
3.  Determine an appropriate $K$ (e.g., using the Elbow Method or Silhouette Analysis).
4.  Instantiate `KMeans` with the chosen `n_clusters` and other parameters (e.g., `init='k-means++'`, `n_init=10`).
5.  Train the model: `kmeans_model.fit(X_scaled)`.
    * The "training" primarily involves finding the optimal centroid locations.
6.  Get cluster assignments: `cluster_labels = kmeans_model.labels_` or `kmeans_model.predict(X_scaled)`.
7.  Get centroid locations: `centroids = kmeans_model.cluster_centers_`.
8.  Analyze and visualize the resulting clusters (e.g., using PCA for 2D projection if more than 2 features).