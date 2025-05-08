# Topic 17: Clustering - Hierarchical Clustering

## 1. Overview

Hierarchical Clustering is an unsupervised learning algorithm that builds a hierarchy of clusters. Unlike K-Means, it does not require the number of clusters to be specified beforehand. The primary output is a **dendrogram**, a tree-like diagram that visualizes the nested cluster structure and allows for choosing the number of clusters after the hierarchy is built.

There are two main approaches:
* **Agglomerative (Bottom-up):** Starts with each data point as an individual cluster and iteratively merges the closest pairs of clusters until only one cluster remains. This is the most common approach.
* **Divisive (Top-down):** Starts with all data points in a single cluster and recursively splits them into smaller clusters. This is less common due to higher computational complexity.

## 2. Agglomerative Hierarchical Clustering

The agglomerative approach involves the following steps:
1.  **Initialization:** Treat each data point as a separate cluster.
2.  **Iteration (Merging):**
    a.  **Calculate Proximity Matrix:** Compute the distances (dissimilarities) between all pairs of current clusters.
    b.  **Find Closest Pair:** Identify the two clusters that are closest to each other based on a chosen **linkage criterion**.
    c.  **Merge:** Combine these two clusters into a new, single cluster.
    d.  **Update Proximity Matrix:** Reflect the new cluster and its distances to other clusters.
3.  **Repeat:** Continue merging until all data points belong to a single cluster.

### Linkage Criteria (Defining Inter-Cluster Distance)

The method used to calculate the distance between two clusters (which may contain multiple points) is crucial and defined by the linkage criterion:
* **Single Linkage (MIN):** Distance between the closest pair of points, one from each cluster. Tends to produce long, chain-like clusters and can be sensitive to outliers.
    `D(A, B) = min(dist(a,b)) for a in A, b in B`
* **Complete Linkage (MAX):** Distance between the farthest pair of points, one from each cluster. Tends to produce compact, spherical clusters.
    `D(A, B) = max(dist(a,b)) for a in A, b in B`
* **Average Linkage (UPGMA):** Average distance between all pairs of points, one from each cluster. Often a good balance.
    `D(A, B) = (1 / (|A|*|B|)) * Σ Σ dist(a,b) for a in A, b in B`
* **Ward's Linkage:** Merges clusters in a way that minimizes the increase in the total within-cluster variance (Sum of Squared Errors - SSE or WCSS). Tends to produce compact, spherical clusters of roughly equal sizes. Often used with Euclidean distance.

## 3. Dendrograms

* A **dendrogram** is a tree diagram that visualizes the hierarchical clustering process.
    * Leaves represent individual data points.
    * Branches merge as clusters are combined.
    * The height of a merge indicates the distance (or dissimilarity) at which the clusters were joined. Longer vertical lines signify merges between more dissimilar clusters.
* **Choosing the Number of Clusters:**
    * By "cutting" the dendrogram horizontally at a certain height. The number of vertical lines intersected by the cut indicates the number of clusters.
    * A common heuristic is to make a cut where it crosses the longest vertical lines that are not yet merged, suggesting a natural separation.

## 4. Feature Scaling

Hierarchical clustering algorithms that use distance metrics (most common linkage methods) are **sensitive to feature scaling**. If features are on different scales, features with larger values can dominate the distance calculations. It's crucial to **scale or normalize features** (e.g., using `StandardScaler`) before applying hierarchical clustering.

## 5. Advantages and Disadvantages

**Advantages:**
* **No need to pre-specify the number of clusters (K).**
* **Provides a rich visualization (dendrogram)** showing the hierarchy and structure.
* Can discover clusters of **arbitrary shapes** (depending on the linkage criterion, e.g., single linkage).
* **Deterministic** for a given linkage and distance metric.

**Disadvantages:**
* **Computationally intensive ($O(N^2 \log N)$ to $O(N^3)$)**, making it slow for large datasets.
* **Decisions are irreversible (greedy nature):** Early merges/splits cannot be undone, potentially leading to suboptimal global clustering.
* **Sensitive to the choice of linkage criterion and distance metric.**
* Some linkage methods can be **sensitive to outliers**.
* **Dendrogram interpretation can be subjective** for choosing the number of clusters.
* **Scalability issues** for very large datasets.

## 6. Implementation with Scikit-learn

* Use `sklearn.cluster.AgglomerativeClustering` for performing the clustering once the number of clusters or a distance threshold is decided.
* Use `scipy.cluster.hierarchy` (specifically `shc.linkage` and `shc.dendrogram`) for generating and plotting dendrograms to help determine the number of clusters.

**Key Parameters for `AgglomerativeClustering`:**
* `n_clusters`: The number of clusters to find. (Set to `None` if using `distance_threshold`).
* `affinity`: Metric used to compute linkage (e.g., 'euclidean', 'manhattan', 'cosine'). Default is 'euclidean'.
* `linkage`: Which linkage criterion to use ('ward', 'complete', 'average', 'single'). 'ward' is a common default and requires 'euclidean' affinity.
* `distance_threshold`: The linkage distance threshold; clusters will not be merged above this threshold. (Set to `None` if using `n_clusters`).

**General Workflow:**
1.  Load and preprocess data.
2.  **Scale features.**
3.  Generate and plot a dendrogram using `scipy.cluster.hierarchy.linkage` and `scipy.cluster.hierarchy.dendrogram` to help decide on the number of clusters or a distance threshold.
4.  Instantiate `AgglomerativeClustering` with the chosen `n_clusters` (or `distance_threshold`) and `linkage` method.
5.  Fit the model and get cluster labels: `cluster_labels = model.fit_predict(X_scaled)`.
6.  Analyze and visualize the resulting clusters.

---
