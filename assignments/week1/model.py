import numpy as np


class LinearRegression:
    """
    A linear regression model to fit the training data.
    """

    # w: np.ndarray
    # b: float

    def __init__(self):
        self.w = []
        self.b = []
        return None

        # self.w = np.randn(self.X.shape[0])
        # self.b = np.random.random_sample()
        # print(self.X.shape)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        fit the model for a given training dataset.
        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output.
        Returns:
            np.ndarray: The predicted output.

        """
        X_new = np.append(np.ones((X.shape[0], 1)), X, 1)
        y = y
        weights = np.linalg.pinv(X_new.T @ X_new) @ (X_new.T @ y)
        print(weights.shape)
        self.b = weights[0]
        self.w = weights[1:]
        print(self.b)
        print(self.w)
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
            lr (float): learning rate.
            epochs (int): Number of Epochs.
        Returns:
            np.ndarray: The predicted output.

        """
        X = np.append(np.ones((X.shape[0], 1)), X, 1)
        w = np.zeros(X.shape[1])
        for i in range(epochs):
            y_hat = X @ w.T
            df_dm = (-2 / X.shape[0]) * (X.T @ (y - y_hat))
            # df_dm = df_dm.reshape(len(df_dm),-1)
            w = w - df_dm * lr
            self.b = np.array(w[0])
            self.w = np.array(w[1:])

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
