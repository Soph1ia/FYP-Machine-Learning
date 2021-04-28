import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def linearRegression():
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
    scaled_data = preprocessing.normalize(data)
    # scaled_data = MinMaxScaler().fit_transform(data)
    # scaled_data = StandardScaler().fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=col_names)

    scaled_test_data = preprocessing.normalize(testing_data)
    # scaled_test_data = MinMaxScaler().fit_transform(testing_data)
    # scaled_test_data = StandardScaler().fit_transform(testing_data)
    scaled_test_data = pd.DataFrame(scaled_test_data, columns=col_names)

    # Split into features and labels
    X_train = scaled_data[features]
    y_train = scaled_data[label]

    # Split the test data and get only features
    x_test = scaled_test_data[features]
    y_test = scaled_test_data[label]

    # get the regression
    regressor = LinearRegression(normalize=True)

    # train the model, get a hypothesis
    regressor.fit(X_train, y_train)

    # Get Accuracy for model using training data
    accuracy = regressor.score(X_train, y_train)
    print("[Training Data] >> Accuracy for linear regression model is: {}".format(accuracy))

    # Test model with testing data
    y_prediction = regressor.predict(x_test)

    # get the accuracy for the testing data ( this is R^2 )
    accuracy = regressor.score(x_test, y_test)
    print("[Testing Data] >> The accuracy achieved for the testing data is {} ".format(accuracy))

    # Calculate RMSE
    print('The RMSE is: {}'.format(mean_squared_error(y_test, y_prediction)))

    # Calculate MAE .
    print('The MAE is: {}'.format(mean_absolute_error(y_test, y_prediction)))

    cross_val_score_predictions = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10)
    print('The cross validation score {}'.format(cross_val_score_predictions.mean()))

    # the statsmodel
    x1 = sm.add_constant(X_train)
    result = sm.OLS(y_train, x1).fit()
    print('Total statsmodel results: {}'.format(result.summary()))


if __name__ == '__main__':
    linearRegression()
