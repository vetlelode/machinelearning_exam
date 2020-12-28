import numpy as np
from preprocessing import get_dataset
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler as Skaler
from math import sqrt


class StandardScaler:
    """
    Our implementation of an Sklearn style scaler
    Scales the dataset using the formula:
    z = (x - u) / s
    where x is the original sample, u is the mean value of the column,
    and s is the standard deviation of the column
    """

    def __init__(self):
        self.scaler_vars = np.empty((30, 2))

    def fit(self, X: np.ndarray, y=None):
        """
        Fit dataset X into the scaler
        """
        X = np.array(X)
        self._reset()
        for i in range(len(X[0])):
            column = X[:, i]
            # U
            self.scaler_vars[i, 0] = np.mean(column)
            # S
            self.scaler_vars[i, 1] = np.std(column, ddof=0)**2

    def transform(self, X):
        """
        Transform the provided dataset X using the earlier fit
        """
        X = np.array(X)
        X_new = np.empty(X.shape)
        #z = (x - u) / s
        X -= self.scaler_vars[:, 0]
        X /= np.sqrt(self.scaler_vars[:, 1])

        return X

    def fit_transform(self, X, y=None):
        """
        QOL method combines both the
        fit and transform functions into one
        """
        self.fit(X=X)
        return self.transform(X=X)

    def _reset(self):
        "Reset everything"
        self.scaler_vars = np.empty((30, 2))


if __name__ == "__main__":
    """
    This is just testing code to make sure the scaler gives the same results as the SKlearn one 
    """
    X_train, Y_train, X_test, Y_test = get_dataset(
        sample=1000, pollution=0.7, train_size=0.9)
    # Test our awsome scaler
    scaler = StandardScaler()
    X_train_0 = X_train
    X_test_0 = X_test
    X_train_1 = X_train
    X_test_1 = X_test
    scaler.fit(X_train_0)
    X_train_0 = scaler.transform(X_train_0)
    X_test_0 = scaler.transform(X_test_0)
    # Test the boring Sklearn scaler
    skaler = Skaler()
    X_train_1 = skaler.fit_transform(X_train_1)
    X_test_1 = skaler.transform(X_test_1)

    # Check for deviations from the standard Sklearn scaler
    failed = False
    for i in range(len(X_train_0[0])):
        if(round(scaler.scaler_vars[i][0], 4) != round(skaler.mean_[i], 4)):
            failed = True
            print(scaler.scaler_vars[i][0], skaler.mean_[i])
        elif (round(scaler.scaler_vars[i][1], 4) != round(skaler.var_[i], 4)):
            failed = True
            print(scaler.scaler_vars[i][1], skaler.var_[i])

    for i in range(len(X_train_0[0])):
        if(round(X_train_0[0][i], 4) != round(X_train_1[0][i], 4)):
            failed = True
            print(X_train_0[0][i], X_train_1[0][i])
        if(round(X_test_0[0][i], 4) != round(X_test_1[0][i], 4)):
            failed = True
            print(X_test_0[0][i], X_test_1[0][i])

    # If there are no major deviations from the SKlearn scaler print a happy message
    if failed == False:
        print("Huzzah, the two scalers are more or less the same!")

    # Since we are using the scaler in a pipeline make sure it works
    pipe = make_pipeline(StandardScaler())
    pipe.fit_transform(X_train)
    pipe.transform(X_test)
