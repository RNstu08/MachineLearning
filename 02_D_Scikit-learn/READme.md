# Scikit-learn Learning Roadmap

This roadmap outlines the essential concepts, modules, and techniques within the Scikit-learn library for performing machine learning tasks in Python.

## I. Introduction & Core Concepts

* **What is Scikit-learn?** Purpose (machine learning toolkit), strengths (algorithms, preprocessing, evaluation, consistent API), dependencies (`NumPy`, `SciPy`).
* **Installation:** `pip install scikit-learn`.
* **Key Design Principles:** Consistency (Estimator API), Inspection, Sensible Defaults, Composition (Pipelines).
* **The Estimator API:** The core interface:
    * `estimator.fit(X, y)`: Train the model (Supervised).
    * `estimator.fit(X)`: Train the model (Unsupervised).
    * `estimator.predict(X)`: Make predictions (Classification, Regression).
    * `estimator.predict_proba(X)`: Predict class probabilities (Classification).
    * `estimator.transform(X)`: Transform data (Preprocessing, Dimensionality Reduction).
    * `estimator.fit_transform(X)`: Fit and transform in one step (often more efficient).
    * `estimator.score(X, y)` / `estimator.score(X)`: Evaluate model performance (default metrics).
* **Data Representation:** Input data (Features `X`, Target `y`) typically as `NumPy` arrays, `Pandas` `DataFrames`, or `SciPy` sparse matrices.
* **Loading Example Datasets:** Using `sklearn.datasets` (e.g., `load_iris()`, `load_digits()`, `make_classification()`).

## II. Data Splitting & Model Evaluation Fundamentals

* **The Need for Splitting:** Training set vs. Test set (evaluating generalization).
* **Splitting Data:** `sklearn.model_selection.train_test_split`. Key parameters: `test_size`, `train_size`, `random_state`, `shuffle`, `stratify` (for classification).
* **Basic Evaluation Concepts:** Overfitting, Underfitting, Bias-Variance Tradeoff (conceptual).
* **Introduction to Metrics (`sklearn.metrics`):**
    * Classification: Accuracy (`accuracy_score`).
    * Regression: Mean Squared Error (`mean_squared_error`), R-squared (`r2_score`).

## III. Data Preprocessing & Feature Engineering

* **Importance:** Why preprocessing is crucial for many algorithms.
* **Scaling Numerical Features (`sklearn.preprocessing`):**
    * `StandardScaler`: Standardize features (zero mean, unit variance).
    * `MinMaxScaler`: Scale features to a given range (e.g., [0, 1]).
    * `RobustScaler`: Uses median and IQR, robust to outliers.
    * `Normalizer`: Scales individual samples (rows) to unit norm.
* **Encoding Categorical Features (`sklearn.preprocessing`):**
    * `OneHotEncoder`: Convert categorical features into one-hot numeric arrays. Handling unknown categories (`handle_unknown`).
    * `LabelEncoder`: Encode target labels with values between 0 and n_classes-1 (use mainly for `y`).
    * `OrdinalEncoder`: Encode categorical features as an integer array based on order.
* **Handling Missing Values (`sklearn.impute`):**
    * `SimpleImputer`: Strategies like `mean`, `median`, `most_frequent`, `constant`.
* **Feature Engineering (`sklearn.preprocessing`):**
    * `PolynomialFeatures`: Generate polynomial and interaction features.
* **Feature Extraction (Text - `sklearn.feature_extraction.text`):**
    * `CountVectorizer`: Convert text documents to a matrix of token counts.
    * `TfidfVectorizer`: Convert text documents to a matrix of TF-IDF features.
* **Feature Selection (Introduction - `sklearn.feature_selection`):**
    * `VarianceThreshold`: Remove low-variance features.
    * Univariate selection (`SelectKBest`, `SelectPercentile`).
    * Recursive Feature Elimination (`RFE`).

## IV. Supervised Learning: Regression

* **Concept:** Predicting a continuous target variable.
* **Linear Models (`sklearn.linear_model`):**
    * `LinearRegression`: Ordinary Least Squares (OLS).
    * `Ridge`: OLS with L2 regularization.
    * `Lasso`: OLS with L1 regularization (can perform feature selection).
    * `ElasticNet`: Combination of L1 and L2 regularization.
* **Other Common Regressors:**
    * `KNeighborsRegressor` (`sklearn.neighbors`).
    * `DecisionTreeRegressor` (`sklearn.tree`).
    * `RandomForestRegressor`, `GradientBoostingRegressor` (`sklearn.ensemble`).
    * `SVR` (Support Vector Regressor) (`sklearn.svm`).
* **Evaluation Metrics (`sklearn.metrics`):** `mean_squared_error`, `mean_absolute_error`, `r2_score`.

## V. Supervised Learning: Classification

* **Concept:** Predicting a categorical target variable.
* **Common Classifiers:**
    * `LogisticRegression` (`sklearn.linear_model`): Handles binary/multinomial classification, includes regularization.
    * `SVC` (Support Vector Classifier) (`sklearn.svm`): Effective in high dimensions, different kernels (`linear`, `poly`, `rbf`).
    * `KNeighborsClassifier` (`sklearn.neighbors`): Instance-based learning.
    * `DecisionTreeClassifier` (`sklearn.tree`).
    * `RandomForestClassifier`, `GradientBoostingClassifier` (`sklearn.ensemble`).
    * Naive Bayes (`sklearn.naive_bayes`): `GaussianNB`, `MultinomialNB`, `BernoulliNB`.
    * `SGDClassifier` (`sklearn.linear_model`): Efficient for large datasets.
* **Evaluation Metrics (`sklearn.metrics`):**
    * Accuracy: `accuracy_score`.
    * Confusion Matrix: `confusion_matrix`, `ConfusionMatrixDisplay`.
    * Precision, Recall, F1-score: `precision_score`, `recall_score`, `f1_score` (and averaging options: `binary`, `micro`, `macro`, `weighted`).
    * Classification Report: `classification_report`.
    * ROC Curve & AUC: `roc_curve`, `roc_auc_score` (primarily for binary).
    * Precision-Recall Curve: `precision_recall_curve`.

## VI. Model Selection & Hyperparameter Tuning (`sklearn.model_selection`)

* **Cross-Validation (CV):** More robust model evaluation than a single train-test split.
    * `cross_val_score`: Evaluate a metric using CV.
    * `cross_validate`: Evaluate multiple metrics using CV.
    * CV Iterators: `KFold`, `StratifiedKFold` (preserves class distribution), `ShuffleSplit`, `LeaveOneOut`, etc.
* **Hyperparameter Tuning:** Finding optimal model parameters.
    * `GridSearchCV`: Exhaustive search over a specified parameter grid using CV.
    * `RandomizedSearchCV`: Samples parameters from distributions using CV (often more efficient).
* **Model Comparison:** Using CV results to select the best model/parameters.
* **Learning Curves & Validation Curves:** Diagnosing model performance (bias/variance).

## VII. Unsupervised Learning

* **Concept:** Finding patterns in unlabeled data.
* **Clustering (`sklearn.cluster`):** Grouping similar data points.
    * `KMeans`: Partitioning data into K clusters based on centroids. Choosing K (Elbow method, Silhouette score).
    * `DBSCAN`: Density-based clustering, finds arbitrarily shaped clusters.
    * `AgglomerativeClustering`: Hierarchical clustering.
    * Evaluation: Silhouette Coefficient (`silhouette_score`), Adjusted Rand Index (`adjusted_rand_score` - requires true labels).
* **Dimensionality Reduction (`sklearn.decomposition`, `sklearn.manifold`):** Reducing the number of features.
    * `PCA` (Principal Component Analysis): Linear transformation to find principal components capturing maximum variance.
    * `NMF` (Non-Negative Matrix Factorization).
    * `TSNE`: T-distributed Stochastic Neighbor Embedding (primarily for visualization).

## VIII. Pipelines & Composite Estimators

* **Pipelines (`sklearn.pipeline`):**
    * `Pipeline`, `make_pipeline`: Chaining multiple steps (e.g., scaler + classifier) into one estimator.
    * Benefits: Convenience, prevents data leakage during cross-validation.
* **Column Transformer (`sklearn.compose`):**
    * `ColumnTransformer`, `make_column_transformer`: Applying different preprocessing steps to different columns (e.g., scaling numerical, one-hot encoding categorical).
* **Feature Union (`sklearn.pipeline.FeatureUnion` - Less common now).**

## IX. Model Persistence

* **Saving and Loading Models:** Using `joblib` (`dump`, `load`) or `pickle` for saving trained models and pipelines. Security and versioning considerations.

## X. Further Topics

* Handling Imbalanced Datasets (brief mention of techniques like resampling, class weighting).
* Feature Importance (accessing `feature_importances_` or `coef_`).
* Partial Fit (for algorithms that support online/incremental learning).
* Brief comparison/mention of when to use libraries like `XGBoost`, `LightGBM`, `CatBoost`, `TensorFlow`, `PyTorch`.