import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np


def prepare():
    '''
    This method prepares the training data. This is done by
    shuffling the data mapping categorical features to numerical values
    processing the data and splitting the training data into features and labels.
    :return:
    '''
    col_names = ['programming_language', 'cpu_intensity', 'memory_intensity', 'memory_size', 'provider', 'throughput']
    data = pd.read_csv('Machine-Learning-Data.csv', header=None, names=col_names)

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

    # Shuffle the data
    data = shuffle(data)
    data = data.sample(frac=1)

    # Normalise ONLY the feature keep label unnormalised
    # X_train = normalise_input_data(data[features], features)
    # y_train = data[label]
    X_train = data[features]
    y_train = data[label]

    return X_train, y_train


def normalise_input_data(data, col_names):
    '''
    This method normalises the data provided the data and the columns in order to create the dataframe

    :param data:
    :param col_names:
    :return: DataFrame
    '''
    return pd.DataFrame(preprocessing.normalize(data), columns=col_names)


def scale_input_data(data):
    return "TODO"


def prepare_test_data():
    col_names = ['programming_language', 'cpu_intensity', 'memory_intensity', 'memory_size', 'provider', 'throughput']
    data = pd.read_csv('testing-data.txt', header=None, names=col_names)

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

    x_train = data[features]
    y_train = data[label]

    return x_train, y_train
