# QuantGaussVector

**QuantGaussVector** is a Python library designed for advanced Gaussian Process Regression (GPR) and Classification with integrated quantization and vectorization techniques. This library includes models optimized for parallel processing, making it suitable for high-performance machine learning tasks.

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
   - [Gaussian Process Regression (GPR)](#gaussian-process-regression-gpr)
   - [Quantization](#quantization)
   - [Vectorization](#vectorization)
   - [Parallel Processing](#parallel-processing)
3. [Installation](#installation)
4. [Usage](#usage)
   - [QGVR Model](#qgvr-model)
   - [QVGC Model](#qvgc-model)
   - [Parameters](#parameters)
5. [Examples](#examples)
   - [Regression with QGVR](#regression-with-qgvr)
   - [Classification with QVGC](#classification-with-qvgc)
6. [API Reference](#api-reference)
7. [License](#license)

## Overview

**QuantGaussVector** offers two main models:

1. **QGVR (Quantized Gaussian Vector Regression)**: A regression model based on Gaussian Processes that incorporates quantization and vectorization techniques for enhanced performance.
2. **QVGC (Quantized Vector Gaussian Classification)**: A classification model that leverages Gaussian Processes for probabilistic predictions, incorporating quantization and vectorization.

These models are designed to handle large datasets efficiently, utilizing parallel processing to accelerate computation.

## Theoretical Background

### Gaussian Process Regression (GPR)

Gaussian Process Regression (GPR) is a non-parametric, Bayesian approach to regression that models the distribution over functions. It is particularly useful for making predictions with uncertainty estimates. The core idea is to place a Gaussian Process prior over the function to be predicted, and use observed data to update this prior.

### Quantization

Quantization in machine learning involves reducing the precision of the numbers used to represent data. This technique can reduce memory usage and computational cost, especially when working with large datasets. In **QuantGaussVector**, quantization is applied dynamically, with options for min-max scaling to fit data into a lower precision range.

### Vectorization

Vectorization is a method of optimizing computations by replacing explicit loops with array operations. This is particularly powerful in Python, where operations on NumPy arrays can be executed in C, leading to significant performance improvements. Both **QGVR** and **QVGC** models are fully vectorized to maximize efficiency.

### Parallel Processing

Parallel processing divides tasks into smaller subtasks that can be processed simultaneously on multiple cores. In **QuantGaussVector**, the `joblib` library is used to parallelize the sampling and prediction steps, enabling the models to handle large datasets more efficiently.

## Installation

You can install **QuantGaussVector** using pip:

```bash
pip install QuantGaussVector
```
### Alternatively, you can clone the repository and install the package manually:
```bash
git clone https://github.com/MM21B038/QuantGaussVector.git
cd QuantGaussVector
pip install .
```

## Usage

### QGVR Model
#### The QGVR model is used for regression tasks. Here's how you can use it:
```bash
from QuantGaussVector import QGVR

# Initialize the model
model = QGVR(kernel='rbf', length_scale=1.0, sigma_f=1.0, sigma_n=0.1, quantize=True)

# Fit the model on training data
model.fit(X_train, y_train)

# Make predictions
y_mean, y_cov = model.predict(X_test)
```

### QVGC Model
#### The QVGC model is used for classification tasks. Here's how you can use it:
```bash
from QuantGaussVector import QVGC

# Initialize the classifier
classifier = QVGC(kernel='linear', length_scale=1.0, sigma_f=1.0, sigma_n=0.1, quantize=True)

# Fit the model on training data
classifier.fit(X_train, y_train)

# Predict probabilities
probabilities, variances = classifier.predict_proba(X_test)

# Predict class labels
predictions = classifier.predict(X_test)
```

## Parameters
- `kernel`: Type of kernel to use ('rbf' or 'linear').
- `length_scale`: Length scale for the kernel.
- `sigma_f`: Signal variance.
- `sigma_n`: Noise variance.
- `alpha`: Regularization parameter.
- `n_jobs`: Number of parallel jobs.
- `dtype`: Data type for computation.
- `quantize`: Whether to enable dynamic range quantization.

## Examples

### Regression with QGVR
```bash
import numpy as np
from QuantGaussVector import QGVR

# Example dataset
X_train = np.random.rand(100, 3)
y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])

X_test = np.random.rand(20, 3)

# Initialize and fit model
model = QGVR(kernel='rbf', length_scale=1.0, sigma_f=1.0, sigma_n=0.1, quantize=True)
model.fit(X_train, y_train)

# Predict
y_mean, y_cov = model.predict(X_test)
print("Predicted mean:", y_mean)
print("Predicted covariance:", y_cov)
```

### Classification with QVGC
```bash
import numpy as np
from QuantGaussVector import QVGC

# Example dataset
X_train = np.random.rand(100, 3)
y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)

X_test = np.random.rand(20, 3)

# Initialize and fit classifier
classifier = QVGC(kernel='linear', length_scale=1.0, sigma_f=1.0, sigma_n=0.1, quantize=True)
classifier.fit(X_train, y_train)

# Predict probabilities and classes
probabilities, variances = classifier.predict_proba(X_test)
predictions = classifier.predict(X_test)
print("Predicted probabilities:", probabilities)
print("Predicted classes:", predictions)
```

## API Reference

### QGVR
- __init__(self, kernel='rbf', length_scale=1.0, sigma_f=1.0, sigma_n=0.1, alpha=1e-10, n_jobs=1, dtype=np.float64, quantize=False): Initializes the QGVR model.
- fit(self, X_train, y_train): Fits the model to the training data.
- predict(self, X_test): Predicts the mean and covariance for the test data.
- sample_y(self, X_test, n_samples=3): Generates samples from the posterior distribution.
- log_marginal_likelihood(self): Computes the log marginal likelihood.
- get_params(self): Returns the model parameters.
- set_params(self, **params): Sets the model parameters.
- score(self, X_test, y_true): Computes the mean squared error of the predictions.

### QVGC
- __init__(self, kernel='rbf', length_scale=1.0, sigma_f=1.0, sigma_n=0.1, alpha=1e-10, n_jobs=1, dtype=np.float64, quantize=False): Initializes the QVGC classifier.
- fit(self, X_train, y_train): Fits the classifier to the training data.
- predict_proba(self, X_test): Predicts the probabilities and variances for the test data.
- predict(self, X_test): Predicts the class labels for the test data.
- log_marginal_likelihood(self): Computes the log marginal likelihood.
- get_params(self): Returns the classifier parameters.
- set_params(self, **params): Sets the classifier parameters.
- score(self, X_test, y_test): Computes the accuracy of the predictions.
- decision_function(self, X_test): Computes the decision function for the test data.

## License
This project is licensed under the MIT License. See the LICENSE file for details.