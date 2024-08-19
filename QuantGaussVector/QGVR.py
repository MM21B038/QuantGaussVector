import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve
from joblib import Parallel, delayed

class QGVR:
    def __init__(self, kernel='rbf', length_scale=1.0, sigma_f=1.0, sigma_n=0.1, alpha=1e-10, n_jobs=1, dtype=np.float64, quantize=False):
        self.params = {
            'kernel': kernel,
            'length_scale': length_scale,
            'sigma_f': sigma_f,
            'sigma_n': sigma_n,
            'alpha': alpha,
            'n_jobs': n_jobs,
            'dtype': dtype,
            'quantize': quantize  # Enable/disable dynamic range quantization
        }
        self.kernel = self.choose_kernel(kernel, length_scale, sigma_f)
        self.sigma_n = sigma_n
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.dtype = dtype
        self.quantize = quantize

    def choose_kernel(self, kernel, length_scale, sigma_f):
        if kernel == 'rbf':
            return lambda X1, X2: sigma_f**2 * np.exp(-0.5 / length_scale**2 * self.squared_distance(X1, X2))
        elif kernel == 'linear':
            return lambda X1, X2: np.dot(X1, X2.T) * sigma_f**2
        else:
            raise ValueError("Unsupported kernel type")

    def squared_distance(self, X1, X2):
        # Applying quantization if enabled
        if self.quantize:
            X1 = self.quantize_features(X1)
            X2 = self.quantize_features(X2)
        return np.sum((X1[:, np.newaxis] - X2)**2, axis=2, dtype=self.dtype)

    def quantize_features(self, X):
        # Simple min-max scaling to [0, 1], then to [0, 255] if dtype is integer
        X_scaled = (X - np.min(X)) / (np.max(X) - np.min(X))
        if np.issubdtype(self.dtype, np.integer):
            return (X_scaled * 255).astype(self.dtype)
        return X_scaled.astype(self.dtype)

    def fit(self, X_train, y_train):
        self.X_train = np.atleast_2d(X_train).astype(self.dtype)
        self.y_train = np.array(y_train, dtype=self.dtype)
        if self.quantize:
            self.X_train = self.quantize_features(self.X_train)
        K = self.kernel(self.X_train, self.X_train) + (self.sigma_n**2 + self.alpha) * np.eye(len(self.X_train), dtype=self.dtype)
        self.L = cholesky(K, lower=True)
        self.alpha_y = solve_triangular(self.L, self.y_train, lower=True)

    def predict(self, X_test):
        X_test = np.atleast_2d(X_test).astype(self.dtype)
        if self.quantize:
            X_test = self.quantize_features(X_test)
        K_trans = self.kernel(self.X_train, X_test)
        v = solve_triangular(self.L, K_trans, lower=True)
        y_mean = np.dot(v.T, self.alpha_y)
        K_ss = self.kernel(X_test, X_test)
        y_cov = K_ss - np.dot(v.T, v)
        return y_mean, y_cov

    def sample_y(self, X_test, n_samples=3):
        mean, cov = self.predict(X_test)
        results = Parallel(n_jobs=self.n_jobs)(delayed(np.random.multivariate_normal)(mean, cov) for _ in range(n_samples))
        return np.array(results).T

    def log_marginal_likelihood(self):
        log_likelihood = -0.5 * np.dot(self.y_train.T, self.alpha_y)
        log_likelihood -= np.sum(np.log(np.diagonal(self.L)))
        log_likelihood -= len(self.y_train) / 2 * np.log(2 * np.pi)
        return log_likelihood

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.params:
                setattr(self, key, value)
                self.params[key] = value
        self.kernel = self.choose_kernel(self.params['kernel'], self.params['length_scale'], self.params['sigma_f'])

    def get_params(self):
        return self.params

    def score(self, X_test, y_true):
        y_pred, _ = self.predict(X_test)
        return np.mean((y_true - y_pred)**2)

    def set_score_request(self, X, y):
        self.score_request = {'X': X, 'y': y}
        return self.score_request

    def set_predict_request(self, X):
        self.predict_request = {'X': X}
        return self.predict_request

    def get_metadata_routing(self):
        return {
            'class_name': self.__class__.__name__,
            'module': self.__module__,
            'params': self.get_params()
        }