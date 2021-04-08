import joblib
import pandas as pd
import scipy.stats as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor


def data_preprocessing(column_name):
    le = LabelEncoder()
    le.fit(data_frame[column_name])
    data_frame[column_name] = le.transform(data_frame[column_name])


# CRIM - per capita crime rate by town
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS - proportion of non-retail business acres per town.
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX - nitric oxides concentration (parts per 10 million)
# RM - average number of rooms per dwelling
# AGE - proportion of owner-occupied units built prior to 1940
# DIS - weighted distances to five Boston employment centres
# RAD - index of accessibility to radial highways
# TAX - full-value property-tax rate per $10,000
# PTRATIO - pupil-teacher ratio by town
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT - % lower status of the population
# MEDV - Median value of owner-occupied homes in $1000's

data_frame = pd.read_excel('./boston.xls')
X = data_frame[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PT', 'B', 'LSTAT']]
y = data_frame['MV']
print('X', X)
print('y', y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
print("X_train=", X_train.shape)
print("y_train=", y_train.shape)
print("X_test=", X_test.shape)
print("y_test=", y_test.shape)
one_to_left = st.beta(10, 1)
from_zero_positive = st.expon(0, 50)
params = {
    "n_estimators": st.randint(3, 50),
    "max_depth": st.randint(3, 50),
    "learning_rate": st.uniform(0.0001, 0.5),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive
}
if __name__ == "__main__":
    xgbreg = XGBRegressor(nthread=8)
    rsCV = RandomizedSearchCV(xgbreg, params, n_jobs=16)
    rsCV.fit(X_train, y_train)
    clf = XGBRegressor(**rsCV.best_params_)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "./PriceModel.sav")
    prediction = clf.predict(X_test)
    print(prediction)
    print("MAE: %.4f" % mean_absolute_error(y_test, prediction))
    print("MSE: %.4f" % mean_squared_error(y_test, prediction))
    result = clf.score(X_test, y_test)
    print('Score::', result)
