import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils import shuffle
from DataPreperation import *


def knn():
    x, y = prepare()

    # Scale The feature
    ss_x = StandardScaler()
    ss_y = StandardScaler()
    # X_train = ss_x.fit_transform(x)
    # y_train = ss_y.fit_transform(y)

    rbs_x = RobustScaler()
    rbs_y = RobustScaler()
    # X_train = rbs_x.fit_transform(x)
    # y_train = rbs_y.fit_transform(y)

    mnm_x = MinMaxScaler()
    mnm_y = MinMaxScaler()
    X_train = mnm_x.fit_transform(x)
    y_train = mnm_y.fit_transform(y)

    y_train = y

    # # Split the test data
    # x_test = scaled_test_data[features]
    # y_test = scaled_test_data[label]
    #
    # # split the data to predict and scale it.
    # x_predicting_data = data_to_predict[features]
    # scaled_data_to_predict = preprocessing.normalize(x_predicting_data)
    # X_scaled_data_to_predict = pd.DataFrame(scaled_data_to_predict, columns=features)

    # find best K value for neighbours
    neighbour_options = {'n_neighbors': range(1, 150)}
    knn_function = KNeighborsRegressor()
    model = GridSearchCV(knn_function, neighbour_options, cv=5)
    model.fit(X_train, y_train)
    print(' The best K for KNN is: {}'.format(model.best_params_))

    # train the model using the best K value calculated above
    regressor = KNeighborsRegressor(n_neighbors=model.best_params_['n_neighbors'])
    regressor.fit(X_train, y_train)

    # Calculate accuracy with training data.
    accuracy = regressor.score(X_train, y_train)
    print('The R2 score is : {}'.format(accuracy))

    # # Predict using Test data
    # y_predictions = regressor.predict(x_test)
    #
    # # Calculate the RMSE
    # rmse = mean_squared_error(y_test, y_predictions)
    # print('The RMSE is: {}'.format(rmse))
    #
    # # Calculate the MAE
    # mae = mean_absolute_error(y_test, y_predictions)
    # print('The MAE is: {}'.format(mae))

    # Cross Validation Score
    cross_val_result = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10)
    print('The cross validation score average is : {}'.format(cross_val_result.mean()))

    # Predict using Test data
    # y_predict = regressor.predict(X_scaled_data_to_predict)
    #
    # print("0 = Java, 1 = Python, 2 = Ruby, 3 = NodeJs, 4 = Go")
    # j = 0
    # for i in y_predict:
    #     programming_language = X_scaled_data_to_predict['programming_language'][j]
    #     print("The prediction for language: {} is {} ops/ms".format(programming_language, i))
    #     j = j + 1


if __name__ == '__main__':
    knn()
