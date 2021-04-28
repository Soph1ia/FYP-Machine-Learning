from sklearn import svm
from DataPreperation import *


def SVMRegression():
    # Get the training data
    X_train, y_train = prepare()

    # x_test = [[1, 2, 0, 128, 1],[1, 2, 0, 128, 1],[1, 2, 0, 128, 1],[1, 2, 0, 128, 1],[1, 2, 0, 128, 1]]
    # features = ['programming_language', 'cpu_intensity', 'memory_intensity', 'memory_size', 'provider']
    # x_test = pd.DataFrame(x_test, columns=features)

    # Process both x and y using standard scalar
    sc_X = StandardScaler()
    sc_Y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_Y.fit_transform(y_train)

    # Create The Model
    regr = svm.SVR(kernel='rbf')

    # Fit the model
    regr.fit(X, y.ravel())

    y_pred = regr.predict(X_train)
    y_pred = sc_Y.inverse_transform(y_pred)

    print("The predictions are:", y_pred)


if __name__ == '__main__':
    SVMRegression()
