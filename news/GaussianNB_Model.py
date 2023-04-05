import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from utils import *



csv_filepath = "articles_labels.csv"

def run_GaussianNB():
    dataset = load_data(csv_filepath)
    encoded_labels = encode_labels(dataset)
    encoded_features = encode_features_countvectorizer(dataset)
    X_train, X_test, y_train, y_test = split(encoded_features, encoded_labels)

    classifier = GaussianNB();
    classifier.fit(X_train, y_train)
 
    # predicting test set results
    y_pred = classifier.predict(X_test)

    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="weighted")
    print("Accuracy:", accuray)
    print("F1 Score:", f1)

run_GaussianNB()