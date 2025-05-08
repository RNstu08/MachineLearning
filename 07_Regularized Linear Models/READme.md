# Topic 7: Regularized Linear Models

## Overview

Regularized Linear Models are extensions of standard Linear Regression (Ordinary Least Squares - OLS) designed to address common issues like **overfitting** and **multicollinearity**. Overfitting occurs when a model learns the training data too well, including its noise, leading to poor performance on unseen data. This is especially problematic with a large number of features.

Regularization techniques introduce a **penalty term** to the model's cost function. This penalty discourages overly complex models (typically by shrinking the magnitude of the coefficients), leading to better generalization and more robust performance.

**Key Prerequisite: Feature Scaling**
It is crucial to scale features (e.g., using `StandardScaler`) before applying regularized linear models, as the penalty terms are sensitive to the scale of the coefficients.

## Core Idea: Modified Cost Function

The general form of the cost function for regularized linear models is:

`J_Regularized(b) = MSE + Penalty_Term`

Where `MSE` is the Mean Squared Error, and the `Penalty_Term` depends on the type of regularization used. The hyperparameter `alpha` (α) controls the strength of this penalty.

## Types of Regularized Linear Models

### 1. Ridge Regression (L2 Regularization)

* **Penalty Term:** Adds the sum of the squares of the coefficient magnitudes (L2 norm) to the cost function. The intercept (`b₀`) is typically not regularized.
    `Penalty_L2 = α * Σ(bⱼ)²` (for j=1 to p features)
* **Cost Function:** `J_Ridge(b) = MSE + α * Σ(bⱼ)²`
* **Hyperparameter `alpha` (α):**
    * Controls regularization strength. `α ≥ 0`.
    * `α = 0`: Equivalent to OLS Linear Regression.
    * `α → ∞`: Coefficients shrink closer to zero.
* **Effects:**
    * Shrinks all coefficients towards zero, but **rarely to exactly zero**.
    * Reduces model variance and helps prevent overfitting.
    * Particularly effective when dealing with multicollinearity (distributes coefficient weights among correlated features).
    * Keeps all features in the model but moderates their influence.
* **Scikit-learn:** `sklearn.linear_model.Ridge`, `sklearn.linear_model.RidgeCV` (for finding optimal `alpha` via cross-validation).

### 2. Lasso Regression (L1 Regularization)

* **Penalty Term:** Adds the sum of the absolute values of the coefficient magnitudes (L1 norm).
    `Penalty_L1 = α * Σ|bⱼ|` (for j=1 to p features)
* **Cost Function:** `J_Lasso(b) = MSE + α * Σ|bⱼ|`
* **Hyperparameter `alpha` (α):** Similar role as in Ridge.
* **Effects:**
    * Shrinks coefficients towards zero.
    * Crucially, it can shrink some coefficients to **exactly zero**, performing **automatic feature selection**.
    * Useful when many features are suspected to be irrelevant.
    * If features are highly correlated, Lasso tends to arbitrarily select one and zero out others.
* **Scikit-learn:** `sklearn.linear_model.Lasso`, `sklearn.linear_model.LassoCV`.

### 3. Elastic Net Regression (L1 + L2 Regularization)

* **Penalty Term:** A linear combination of L1 and L2 penalties, controlled by `alpha` and `l1_ratio`.
    Scikit-learn's cost function (simplified): `J_ElasticNet(b) = MSE + α * l1_ratio * Σ|bⱼ| + α * 0.5 * (1 - l1_ratio) * Σ(bⱼ)²`
* **Hyperparameters:**
    * `alpha` (α): Overall regularization strength.
    * `l1_ratio` (ρ, rho): The mixing parameter ($0 \le \text{l1\_ratio} \le 1$).
        * `l1_ratio = 1`: Elastic Net behaves like Lasso.
        * `l1_ratio = 0`: Elastic Net behaves like Ridge.
        * `0 < l1_ratio < 1`: A combination of L1 and L2 effects.
* **Effects:**
    * Combines the benefits of Ridge and Lasso.
    * Can perform feature selection.
    * More robust than Lasso when dealing with groups of highly correlated features (tends to select or discard them together).
* **Scikit-learn:** `sklearn.linear_model.ElasticNet`, `sklearn.linear_model.ElasticNetCV`.

## How Regularization Prevents Overfitting

1.  **Bias-Variance Trade-off:**
    * Regularization introduces a small amount of bias into the coefficient estimates (they are shrunk from their OLS values).
    * This leads to a significant reduction in the model's variance (its sensitivity to the specific training data).
    * The goal is to find an `alpha` that optimizes this trade-off, resulting in lower overall error on unseen data.
2.  **Constraining Model Complexity:**
    * By penalizing large coefficient values, regularization prevents the model from becoming overly complex and fitting noise in the training data.
    * Simpler models (with smaller coefficient magnitudes or fewer features, as with Lasso/Elastic Net) often generalize better.

## Choosing Regularization Strength (`alpha` and `l1_ratio`)

* These hyperparameters are crucial and must be tuned for optimal performance.
* **Cross-Validation (CV)** is the standard method:
    * The training data is split into multiple folds.
    * The model is trained and validated on different combinations of these folds for various hyperparameter values.
    * The hyperparameter set yielding the best average validation performance is chosen.
* **Scikit-learn tools:**
    * `RidgeCV`, `LassoCV`, `ElasticNetCV`: Estimators with built-in cross-validation to find the best `alpha` (and `l1_ratio` for `ElasticNetCV`).
    * `GridSearchCV`, `RandomizedSearchCV`: More general tools for hyperparameter tuning via cross-validation.

## When to Choose Which Model

* **Ridge Regression:**
    * Good default, especially if you believe most features are relevant.
    * Effective for multicollinearity.
    * When you want to reduce model variance without eliminating features.
* **Lasso Regression:**
    * When you suspect many features are irrelevant and desire automatic feature selection.
    * Aims for sparse models (many zero coefficients), which can improve interpretability.
    * Can be unstable with highly correlated features.
* **Elastic Net Regression:**
    * A good compromise when you want both feature selection and better handling of correlated features than Lasso.
    * Often performs well when there are many features, or $p > n$ (more features than samples).
    * Requires tuning two hyperparameters (`alpha` and `l1_ratio`).

**General Advice:** It's often beneficial to try all three (or at least Ridge and one of Lasso/Elastic Net), tune their hyperparameters using cross-validation, and select the model that performs best on your validation metric for your specific dataset.

---
This README should provide a solid summary of Topic 7.