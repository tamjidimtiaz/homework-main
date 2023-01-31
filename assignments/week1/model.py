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
        X_new = np.append(np.ones((X.shape[0],1)), X, 1)
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
            lr (float): learning rate
            epochs (int): number of epochs.
        Returns:
            np.ndarray: weight and bias terms.

        """   
        # define a new matrix comprising of a column of ones and append it with the original 'X'
        X = np.append(np.ones((X.shape[0],1)), X, 1)
        # initialize the weight matrix
        w = np.zeros(X.shape[1])
        for i in range(epochs):
            y_hat = X @ w.T
            df_dm =  (-2/X.shape[0]) * (X.T @ (y - y_hat))
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
