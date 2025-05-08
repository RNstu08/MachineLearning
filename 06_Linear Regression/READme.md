# Topic 6: Linear Regression

## Overview

Linear Regression is a fundamental supervised learning algorithm used for **regression tasks**, meaning its primary goal is to predict a **continuous numerical target variable**. It aims to model the linear relationship between a dependent variable (target, `y`) and one or more independent variables (features or predictors, `X`). The core idea is to find the "best-fitting" straight line (in Simple Linear Regression) or hyperplane (in Multiple Linear Regression) that describes the data.

It's known for its simplicity, interpretability (when assumptions are met), and computational efficiency, often serving as a crucial baseline model.

## Key Concepts Covered

### 1. Types of Linear Regression

* **Simple Linear Regression (SLR):**
    * Involves a single independent variable (`x`).
    * Prediction Equation: `ŷ = b₀ + b₁x`
        * `ŷ`: Predicted value.
        * `b₀`: Intercept (value of `ŷ` when `x=0`).
        * `b₁`: Coefficient/Slope (change in `ŷ` for a one-unit change in `x`).
* **Multiple Linear Regression (MLR):**
    * Involves two or more independent variables (`x₁, x₂, ..., xₚ`).
    * Prediction Equation: `ŷ = b₀ + b₁x₁ + b₂x₂ + ... + bₚxₚ`
        * `b₀`: Intercept (value of `ŷ` when all `xⱼ=0`).
        * `bⱼ`: Coefficient for feature `xⱼ` (change in `ŷ` for a one-unit change in `xⱼ`, holding all other features constant).

### 2. Assumptions of Linear Regression

For the model's estimates and statistical inferences (like p-values, confidence intervals) to be reliable, several assumptions about the error term `ε` (estimated by residuals `y - ŷ`) should hold:

* **Linearity:** The relationship between `X` and the mean of `y` is linear.
* **Independence:** Errors are independent of each other (no autocorrelation).
* **Homoscedasticity:** Errors have constant variance across all levels of `X` (or `ŷ`).
* **Normality of Errors:** Errors are normally distributed with a mean of zero (especially important for small samples and inference).
* **No (or Little) Multicollinearity (for MLR):** Independent variables are not highly correlated with each other.

### 3. Cost Function (How "Good" is the Fit?)

* To find the "best" parameters (`b₀, b₁, ...`), we need to minimize a **cost function** that measures the discrepancy between actual (`y`) and predicted (`ŷ`) values.
* **Mean Squared Error (MSE)** is standard for linear regression:
    `J(b) = MSE = (1/n) * Σ(yᵢ - ŷᵢ)²`
    This function is convex, ensuring a single global minimum.

### 4. Optimization: Finding the Best Parameters

How do we find the `b` values that minimize MSE?

* **Normal Equation (Analytical Solution):**
    * A direct, one-step calculation using matrix algebra:
        `b = (XᵀX)⁻¹ Xᵀy`
    * Requires `XᵀX` to be invertible.
    * Computationally expensive for a very large number of features (`p`).
* **Gradient Descent (Iterative Solution):**
    * An iterative algorithm that starts with initial guesses for `b` and takes steps in the direction of the steepest descent of the cost function.
    * Update Rule: `b := b - α * ∇J(b)`
        * `α` (alpha): **Learning Rate** (step size), a crucial hyperparameter.
        * `∇J(b)`: Gradient of the cost function.
    * **Feature Scaling** (e.g., Standardization) is essential for efficient convergence of Gradient Descent.
    * Types: Batch, Stochastic (SGD), Mini-Batch Gradient Descent.

### 5. Mathematical Intuition

* Linear regression seeks to minimize a convex (bowl-shaped) cost function (MSE).
* This minimization can be achieved either by directly solving for where the gradient of the cost function is zero (Normal Equation) or by iteratively stepping down the gradient (Gradient Descent).

### 6. Implementation with Scikit-learn

Scikit-learn provides convenient tools for linear regression:

* **`sklearn.linear_model.LinearRegression`:**
    * Typically uses an analytical solver (related to the Normal Equation, often via SVD).
    * Does not require explicit feature scaling for correctness but can benefit numerically.
    * Few hyperparameters (e.g., `fit_intercept`).
    * Workflow:
        ```python
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(model.intercept_, model.coef_)
        ```
* **`sklearn.linear_model.SGDRegressor`:**
    * Implements linear regression using Stochastic Gradient Descent.
    * **Requires Feature Scaling** (e.g., `StandardScaler`).
    * Suitable for very large datasets (many samples/features) and online learning.
    * More hyperparameters to tune (e.g., `loss`, `penalty`, `alpha`, `learning_rate`, `eta0`, `max_iter`, `tol`).
    * Workflow:
        ```python
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
        sgd_model.fit(X_train_scaled, y_train)
        predictions_sgd = sgd_model.predict(X_test_scaled)
        print(sgd_model.intercept_, sgd_model.coef_)
        ```

### 7. Model Evaluation

Common metrics for regression (covered in more detail in Topic 5):

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R-squared (R²)**: Proportion of variance in `y` explained by the model.

## Pros and Cons of Linear Regression

**Pros:**
* **Simple & Fast:** Easy to implement and computationally inexpensive.
* **Interpretable:** Coefficients have a clear meaning (if assumptions hold and multicollinearity is low).
* **No Hyperparameters (for basic `LinearRegression`):** The Normal Equation provides a direct solution.
* **Good Baseline:** Excellent starting point for regression tasks.

**Cons:**
* **Linearity Assumption:** Assumes the relationship between features and target is linear; performs poorly if not.
* **Sensitivity to Outliers:** MSE cost function makes it sensitive to extreme values.
* **Multicollinearity Issues:** High correlation between predictors makes coefficient estimates unstable and unreliable.
* **Assumption Dependency:** Reliability of statistical inference (p-values, CIs) heavily depends on the model assumptions holding true.

## Key Takeaways

After this topic, one should understand:
* The fundamental principles of fitting a linear model to data.
* The distinction and application of Simple and Multiple Linear Regression.
* The importance of checking model assumptions and their implications.
* How the model parameters are learned (minimizing MSE via Normal Equation or Gradient Descent).
* How to implement and interpret Linear Regression models using Scikit-learn.

---