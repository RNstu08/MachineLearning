# Topic 8: Polynomial Regression

## Overview

Polynomial Regression is a type of regression analysis used when the relationship between the independent variable(s) `X` and the dependent variable `y` is **non-linear**. While standard linear regression models a straight-line relationship, polynomial regression extends this by fitting the data to an $n^{th}$-degree polynomial.

The core idea is to **transform the original features** by creating new features that are powers of the original ones (e.g., $x^2, x^3$) and their interaction terms. Then, a standard **Multiple Linear Regression** model is fitted to these engineered polynomial features. Thus, it's considered a "linear model" because it is linear in its *coefficients*, even though it models a non-linear relationship with the original features.

## Key Concepts Covered

### 1. Capturing Non-Linear Relationships

* When scatter plots or domain knowledge suggest a curved relationship between `X` and `y`, standard linear regression will underfit.
* Polynomial regression allows the model to fit a wider range of curvatures.

### 2. The Polynomial Equation

For a single independent variable `x`, a polynomial regression model of degree `d` takes the form:

`ŷ = b₀ + b₁x + b₂x² + ... + b_dx^d`

Where:
* `ŷ` is the predicted value.
* `b₀, b₁, ..., b_d` are the coefficients learned by the model.
* `d` is the degree of the polynomial.

If there are multiple features, polynomial features will also include interaction terms (e.g., for features `x₁`, `x₂` and degree 2: `x₁`, `x₂`, `x₁²`, `x₁x₂`, `x₂²`).

### 3. Creating Polynomial Features

* In Scikit-learn, the `sklearn.preprocessing.PolynomialFeatures` transformer is used.
* **Key Parameters:**
    * `degree`: The maximum degree of the polynomial terms.
    * `include_bias`: If `True` (default), includes a column of ones (bias term). Often set to `False` if the subsequent linear model handles the intercept.
    * `interaction_only`: If `True`, only interaction terms are produced (e.g., `x₁x₂`), not power terms like `x₁²`.

### 4. Degree of the Polynomial (`d`) - Hyperparameter

* The `degree` is a crucial hyperparameter that dictates the model's flexibility.
    * `d=1`: Standard linear regression.
    * `d=2`: Quadratic (fits a parabola-like curve).
    * Higher `d`: Allows for more complex, "wiggly" curves.
* Choosing the optimal `d` often involves techniques like cross-validation or visual inspection (for simple cases).

### 5. Relationship with Overfitting

* **Low Degree:** May lead to **underfitting** if the true relationship is more complex than the chosen degree (high bias).
* **Optimal Degree:** Captures the underlying trend well without fitting noise (good bias-variance balance).
* **High Degree:** Can lead to **overfitting**. The model becomes overly flexible, fitting the noise in the training data. This results in excellent performance on training data but poor generalization to unseen test data. Overfit polynomial models often exhibit wild oscillations, especially at the boundaries of the data.

### 6. Implementation with Scikit-learn

The typical workflow involves two main steps:
1.  Transform the input features using `PolynomialFeatures`.
2.  Fit a `LinearRegression` model on these transformed polynomial features.

Using `sklearn.pipeline.Pipeline` is highly recommended to chain these steps:
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Example for degree 2
poly_pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear_regression", LinearRegression())
])
# poly_pipeline.fit(X_train, y_train)
# predictions = poly_pipeline.predict(X_test)
```s
### Feature Scaling: 
When polynomial features are generated, especially with higher degrees, their values can become very large. It's often beneficial to scale these features (e.g., using StandardScaler) after the PolynomialFeatures transformation and before fitting LinearRegression, particularly if regularization is also applied or if the solver is sensitive to feature scales.

### Pros and Cons
Pros:

Allows linear models to capture non-linear relationships.
Relatively simple to implement and understand.
Can model a wide variety of curvatures.
The resulting linear model on polynomial features can still be somewhat interpretable for low degrees.
Cons:

Choosing the optimal degree d can be challenging.
Highly prone to overfitting if d is too large.
Polynomials can behave erratically at the boundaries of the data range (poor extrapolation).
Combinatorial explosion of features: With multiple original features, the number of generated polynomial features (including interactions) can grow very rapidly with the degree, leading to high dimensionality (curse of dimensionality), increased computational cost, and higher risk of overfitting.

### When to Use / Alternatives
Use when a clear, relatively simple non-linear trend is observed in the data.
Good for exploratory analysis to understand the nature of non-linearity.
For more complex non-linear relationships or high-dimensional feature spaces, consider alternatives:
Splines (e.g., in Generalized Additive Models - GAMs)
Tree-based models (Decision Trees, Random Forests, Gradient Boosting)
Support Vector Machines (SVMs) with non-linear kernels
Neural Networks