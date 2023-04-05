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



csv_filepath = "articles_labels.csv"


def load_data(filepath):
    dataset = pd.read_csv(filepath)
    return dataset


def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size = 0.25, random_state = 0)

    return X_train, X_test, y_train, y_test

def encode_labels(dataset):
    # Encode labels
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(dataset['labels'])

def encode_features_countvectorizer(dataset):
    cv = CountVectorizer(max_features = 1500)
    encoded_features = cv.fit_transform(dataset['features']).toarray()
    return encoded_features



def run():
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




run()