import joblib
from sklearn.preprocessing import LabelEncoder


def data_preprocessing(data_frame, column_name):
    le = LabelEncoder()
    le.fit(data_frame[column_name])
    data_frame[column_name] = le.transform(data_frame[column_name])


def perform_prediction(data_frame):
    y_test = data_frame.values
    model_load = joblib.load("./PriceModel.sav")
    prediction = model_load.predict(y_test)
    return prediction
