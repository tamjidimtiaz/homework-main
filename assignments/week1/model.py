import numpy as np


class LinearRegression:
    """
    A linear regression model to fit the training data.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = []
        self.b = []
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        fit the model for a given training dataset.
        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output.
        Returns:
            np.ndarray: weight and bias terms.
        """
        # define a new matrix comprising of a column of ones and append it with the original 'X'
        X_new = np.append(np.ones((X.shape[0], 1)), X, 1)
        # Calculate the weights
        weights = np.linalg.pinv(X_new.T @ X_new) @ (X_new.T @ y)

        # calculate the bias and weights from the vector 'weights'
        self.b = weights[0]
        self.w = weights[1:]
        return self.w, self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.
        Arguments:
            X (np.ndarray): The input data.
        Returns:
            np.ndarray: The predicted output.
        """
        return X @ self.w.T + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        fit the model for a given training dataset.
        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output.
        Returns:
            np.ndarray: weight and bias terms.
        """
        self.lr = lr
        self.epochs = epochs

        y_new = np.reshape(y, (len(y), 1))
        X = np.append(np.ones((X.shape[0], 1)), X, 1)
        N = X.shape[0]
        print(N)
        w = np.zeros((X.shape[1], 1))
        for i in range(epochs):
            gradients = (2 / N) * X.T @ (X @ w - y_new)
            w = w - self.lr * np.clip(gradients, -1, 1)

        self.w = w[1:]
        self.b = w[0]

        return self.w, self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.
        Arguments:
            X (np.ndarray): The input data.
        Returns:
            np.ndarray: The predicted output.
        """
        return X @ self.w + self.b
