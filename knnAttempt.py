import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils import shuffle


def knn():
    col_names = ['programming_language', 'cpu_intensity', 'memory_intensity', 'memory_size', 'provider', 'throughput']
    data = pd.read_csv('Machine-Learning-Data.csv', header=None, names=col_names)
    data_to_predict = pd.read_csv('data-to-predict.txt', header=None, names=col_names)
    testing_data = pd.read_csv('hidden-data-to-test.txt', header=None, names=col_names)

    # split into feature and label
    features = ['programming_language', 'cpu_intensity', 'memory_intensity', 'memory_size', 'provider']
    label = ['throughput']

    # Format the data to map categorical to numerical values
    map_language_to_number = {'Java ': 0, 'Python': 1, 'Ruby': 2, 'NodeJs': 3, 'Go': 4}
    map_mem_intensity = {'no': 0, 'yes': 1}
    map_cpu_intensity_to_number = {'low': 1, 'medium': 2, 'high': 3, 'no': 0}

    # Format the main data
    data['programming_language'] = data['programming_language'].map(map_language_to_number)
    data['cpu_intensity'] = data['cpu_intensity'].map(map_cpu_intensity_to_number)
    data['memory_intensity'] = data['memory_intensity'].map(map_mem_intensity)

    # Format the users predicting data
    data_to_predict['programming_language'] = data_to_predict['programming_language'].map(map_language_to_number)
    data_to_predict['cpu_intensity'] = data_to_predict['cpu_intensity'].map(map_cpu_intensity_to_number)
    data_to_predict['memory_intensity'] = data_to_predict['memory_intensity'].map(map_mem_intensity)

    # format the testing data
    testing_data['programming_language'] = testing_data['programming_language'].map(map_language_to_number)
    testing_data['cpu_intensity'] = testing_data['cpu_intensity'].map(map_cpu_intensity_to_number)
    testing_data['memory_intensity'] = testing_data['memory_intensity'].map(map_mem_intensity)

    # Shuffle the data
    data = shuffle(data)
    data = data.sample(frac=1)

    # Normalise the Data
    scaled_data = MinMaxScaler().fit_transform(data)
    # scaled_data = preprocessing.normalize(data)
    scaled_data = pd.DataFrame(scaled_data, columns=col_names)

    # scaled_test_data_1 = preprocessing.normalize(testing_data)
    # scaled_test_data = pd.DataFrame(scaled_test_data_1, columns=col_names)


    # Split into features and labels
    X_train = scaled_data[features]
    y_train = scaled_data[label]

    # # Split the test data
    # x_test = scaled_test_data[features]
    # y_test = scaled_test_data[label]
    #
    # # split the data to predict and scale it.
    # x_predicting_data = data_to_predict[features]
    # scaled_data_to_predict = preprocessing.normalize(x_predicting_data)
    # X_scaled_data_to_predict = pd.DataFrame(scaled_data_to_predict, columns=features)

    # find best K value for neighbours
    neighbour_options = {'n_neighbors': range(1, 40)}
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
    #
    # # Cross Validation Score
    # cross_val_result = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10)
    # print('The cross validation score average is : {}'.format(cross_val_result.mean()))

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
