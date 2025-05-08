# Topic 18: Clustering - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## 1. Overview

DBSCAN is a **density-based** unsupervised learning algorithm used for clustering. Unlike K-Means, it does not require the number of clusters to be specified beforehand. DBSCAN can find **arbitrarily shaped clusters** and is effective at identifying **outliers (noise points)**.

The core idea is to group together points that are closely packed in high-density regions, separated by low-density regions.

## 2. Key Concepts

DBSCAN defines clusters based on the density of points, using two main parameters:

* **`eps` ($\epsilon$ or Epsilon):** A distance value that defines the radius of a neighborhood around a data point. Any point within this radius is considered a neighbor.
* **`min_samples` (MinPts):** The minimum number of data points (including the point itself) that must be within a point's `eps`-neighborhood for that point to be considered a **core point**.

Based on these parameters, points are classified as:

1.  **Core Point:** A point that has at least `min_samples` points within its `eps`-neighborhood. Core points are in the interior of a dense cluster.
2.  **Border Point:** A point that is not a core point itself but falls within the `eps`-neighborhood of a core point. Border points are on the edges of clusters.
3.  **Noise Point (Outlier):** A point that is neither a core point nor a border point. These are isolated points in low-density regions.

**Connectivity:**
* **Directly Density-Reachable:** Point `Q` is directly density-reachable from core point `P` if `Q` is in `P`'s `eps`-neighborhood.
* **Density-Reachable:** Point `R` is density-reachable from `P` if there's a chain of directly density-reachable points from `P` to `R`.
* **Density-Connected:** Points `P` and `Q` are density-connected if there's a core point `O` from which both `P` and `Q` are density-reachable.

A **cluster** in DBSCAN is a set of density-connected points (all core points and the border points reachable from them).

## 3. The DBSCAN Algorithm Steps

1.  Label all points as unvisited.
2.  Iterate through all unvisited data points ($P$):
    a.  Mark $P$ as visited.
    b.  Find all points in the `eps`-neighborhood of $P$ (`Neighbors`).
    c.  If `len(Neighbors) < min_samples`, mark $P$ as noise (it might later become a border point).
    d.  If `len(Neighbors) >= min_samples` ($P$ is a core point):
        i.  Create a new cluster $C$ and add $P$ to $C$.
        ii. **Expand Cluster:** For each point $Q$ in `Neighbors`:
            1.  If $Q$ is unvisited, mark it visited. If $Q$ is also a core point, add its neighbors to the list of points to explore for cluster $C$.
            2.  If $Q$ is not yet a member of any cluster, add $Q$ to cluster $C$.
3.  Repeat until all points are visited.

## 4. Choosing Hyperparameters (`eps` and `min_samples`)

The choice of `eps` and `min_samples` is crucial and data-dependent.

* **`min_samples`:**
    * A common rule of thumb: `min_samples >= D + 1` (where D is dimensionality), or `min_samples = 2 * D` for higher dimensions.
    * Larger `min_samples` leads to more robust clusters and fewer noise points being included in clusters.
* **`eps`:**
    * Often determined using a **k-distance plot** (or k-NN distance plot):
        1.  Fix `min_samples`. Let $k = \text{min\_samples} - 1$ (or $k = \text{min\_samples}$).
        2.  For each point, find the distance to its $k^{th}$ nearest neighbor.
        3.  Sort these $k$-distances and plot them.
        4.  Look for an "elbow" or "knee" in the plot. The distance value at this point is a good candidate for `eps`. This value represents a threshold where density significantly changes.

**Important Note on Feature Scaling:** DBSCAN uses distance (`eps`), so it's **highly sensitive to feature scaling**. Features must be scaled (e.g., using `StandardScaler`) before applying DBSCAN.

## 5. Advantages and Disadvantages

**Advantages:**
* **Does not require specifying the number of clusters beforehand.**
* Can find **arbitrarily shaped clusters.**
* **Robust to outliers** (has a built-in mechanism for noise detection).
* Parameters (`eps`, `min_samples`) can be somewhat intuitive if domain knowledge about density exists.

**Disadvantages:**
* **Struggles with clusters of varying densities.** A single `eps` and `min_samples` may not work well for all parts of the data.
* **Sensitive to hyperparameter choice.** Finding optimal `eps` and `min_samples` can be challenging.
* **"Curse of Dimensionality":** Performance can degrade in high-dimensional spaces as distance/density becomes less meaningful.
* **Border point assignment** can sometimes be ambiguous if a border point is reachable from multiple clusters (though implementations are deterministic).

## 6. Implementation with Scikit-learn

* Use `sklearn.cluster.DBSCAN`.

**Key Parameters for `DBSCAN`:**
* `eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
* `min_samples`: The number of samples in a neighborhood for a point to be considered as a core point.
* `metric`: The distance metric to use (default is 'euclidean').

**General Workflow:**
1.  Load and preprocess data.
2.  **Scale features.**
3.  Determine appropriate `eps` and `min_samples` (e.g., using k-distance plot for `eps`).
4.  Instantiate `DBSCAN` with chosen parameters.
5.  Fit the model and get cluster labels: `cluster_labels = model.fit_predict(X_scaled)`.
    * Labels will be integers for clusters, and **-1 for noise points**.
6.  Analyze and visualize the resulting clusters and noise points.
7.  Evaluate clustering using metrics like Silhouette Score (if appropriate, usually calculated on non-noise points).

---
