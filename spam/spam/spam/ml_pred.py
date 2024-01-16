import joblib
from .utils import extract_embeddings

logistic_regression = joblib.load('logistic_regression_model.joblib')

def prediction(text):
    processed_text = extract_embeddings(text)
    predictions = logistic_regression.predict(processed_text)
    if predictions == 0:
        print('the email is not spam')
    else:
        print('the email is spam')
    return predictions
