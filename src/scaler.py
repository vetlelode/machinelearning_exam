import numpy as np
from preprocessing import get_dataset


class StandardScaler:
    """
    Our implementation of an Sklearn style scaler
    Scales the dataset using the formula:
    z = (x - u) / s
    where x is the original sample, u is the mean value of the column,
    and s is the standard deviation of the column
    """

    def __init__(self):
        self.var = 0
        self.mean = 0

    def fit(self, X: np.ndarray, y=None):
        """
        Fit dataset X into the scaler
        """
        X = np.array(X)
        self._reset()
        for i in range(len(X[0])):
            column = X[:, i]
            # U
            self.mean = np.mean(column)
            # S
            self.var = np.std(column)

    def transform(self, X):
        """
        Transform the provided dataset X using the earlier fit
        """
        X = np.array(X)
        for i in range(len(X)):
            for j in range(len(X[0])):
                X[i, j] = (
                    X[i, j]-self.mean)/self.var
        return X

    def fit_transform(self, X):
        """
        QOL method combines both the
        fit and transform functions into one
        """
        self.fit(X=X)
        return self.transform(X=X)

    def _reset(self):
        "Reset everything"
        del self.var
        del self.mean


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_dataset(
        sample=1000, pollution=0.7, train_size=0.9)
    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    scaler.transform(X_test)
