import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np


def linearRegression():
    col_names = ['programming_language', 'cpu_intensity', 'memory_intensity', 'memory_size', 'provider', 'throughput']
    data = pd.read_csv('../Machine-Learning-Data.csv', header=None, names=col_names)
    data_to_predict = pd.read_csv('../data-to-predict.txt', header=None, names=col_names)

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

    # Shuffle the data
    data = shuffle(data)
    data = data.sample(frac=1)

    # Normalise the Data
    scaled_data = preprocessing.normalize(data)
    scaled_data = pd.DataFrame(scaled_data, columns=col_names)

    # get features of the test data
    data_to_predict = data_to_predict[features]

    # Normalise the data for predicting
    scaled_test_data = preprocessing.normalize(data_to_predict)
    x_test = pd.DataFrame(scaled_test_data, columns=features)

    # Split into features and labels
    X_train = scaled_data[features]
    y_train = scaled_data[label]

    # get the regression
    regressor = LinearRegression(normalize=True)

    # train the model, get a hypothesis
    regressor.fit(X_train, y_train)

    # Get Accuracy for model using training data
    accuracy = regressor.score(X_train, y_train)
    print("[Training Data] >> Accuracy for linear regression model is: {}".format(accuracy))

    # Test model with training data
    y_prediction = regressor.predict(X_train)
    print("The predicted values are: ", y_prediction)

    l2 = np.linalg.norm(data, ord=2, axis=1)

    un_normalised = y_prediction * l2[:, None]
    print("The un_normalised result is:", un_normalised)

    # get the accuracy for the testing data ( this is R^2 )
    # accuracy = regressor.score(x_test, y_test)
    # print("[Testing Data] >> The accuracy achieved for the testing data is {} ".format(accuracy))

    # # Calculate RMSE
    # print('The RMSE is: {}'.format(mean_squared_error(y_test, y_prediction)))
    #
    # # Calculate MAE .
    # print('The MAE is: {}'.format(mean_absolute_error(y_test, y_prediction)))

    cross_val_score_predictions = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10)
    print('The cross validation score {}'.format(cross_val_score_predictions.mean()))

    # the stats model
    x1 = sm.add_constant(X_train)
    result = sm.OLS(y_train, x1).fit()
    print('Total statsmodel results: {}'.format(result.summary()))


if __name__ == '__main__':
    linearRegression()
