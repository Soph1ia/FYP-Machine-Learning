from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.tree import DecisionTreeRegressor

from DataPreperation import *


def source():
    # Get the Data
    X_train, y_train = prepare()
    x_test, y_test = prepare_test_data()

    # Create a Pipeline
    pipeline_lr = Pipeline([
        ('scalar1', StandardScaler()),
        ('lr_model', LinearRegression(normalize=True))
    ])
    pipeline_knn = Pipeline([
        ('scalar2', StandardScaler()),
        ('knn_model', KNeighborsRegressor(n_neighbors=4))
    ])
    pipeline_SVR = Pipeline([
        ('scalar3', StandardScaler()),
        ('svr_model', svm.SVR(kernel='rbf'))
    ])
    pipeline_decisionTree = Pipeline([
        ('scalar4', StandardScaler()),
        ('DTR_model', DecisionTreeRegressor(max_depth=5))
    ])

    # list of all pipelines
    pipelines = [pipeline_lr, pipeline_knn, pipeline_SVR, pipeline_decisionTree]

    best_accuracy = 0
    best_regr = 0
    best_pipeline = ""

    pipeline_dict = {0: 'Linear Regression', 1: 'Knn', 2: 'SVR', 3: 'Decision Tree Regression'}

    for pipe in pipelines:
        pipe.fit(X_train, y_train)

    # Training Data Results
    print(" >> Results on Training Data ")
    for i, model in enumerate(pipelines):
        print("{} Test Accuracy: {}".format(pipeline_dict[i], model.score(X_train, y_train)))

    # Results on Testing Data
    print(">> Results on Testing Data ")
    for i, model in enumerate(pipelines):
        print("{} Test Accuracy: {}".format(pipeline_dict[i], model.score(x_test, y_test)))

    for i, model in enumerate(pipelines):
        if model.score(x_test, y_test) > best_accuracy:
            best_accuracy = model.score(x_test, y_test)
            best_regr = model
            best_pipeline = i
    print("The Regressor with best accuracy is {}, with accuracy of {}".format(pipeline_dict[best_pipeline],
                                                                               best_accuracy))

    # Now We know the best model . Print out the prediction
    y_predictions = best_regr.predict(x_test)

    print("The predictions are: ", y_predictions)

if __name__ == '__main__':
    source()
