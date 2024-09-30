import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score

try:
    data = pd.read_csv("email.tsv", sep="\t")
except FileNotFoundError:
    print("the file 'email.tsv' was not found.")
    exit()
print(data.columns)


def spam_keywords(row):
    viagra_present = row["viagra"]
    winner_present = 1 if row["winner"] == "yes" else 0
    return viagra_present + winner_present


def content_density(row):
    return (
        row["num_char"] / (row["line_breaks"] + 1)
        if row["line_breaks"] > 0
        else row["num_char"]
    )


def sender_credibility(row):
    email = row["from"]
    if isinstance(email, str):
        return 1 if email.endswith("noreply") or email.endswith(".com") else 0
    return 0


def urgency_indicator(row):
    return row["urgent_subj"] + row["exclaim_subj"] + (1 if row["dollar"] > 0 else 0)


def password_mentioned(row):
    return row["password"]


def multiple_recipients(row):
    return 1 if row["to_multiple"] > 1 else 0


data["spam_keywords"] = data.apply(spam_keywords, axis=1)
data["content_density"] = data.apply(content_density, axis=1)
data["sender_credibility"] = data.apply(sender_credibility, axis=1)
data["urgency_indicator"] = data.apply(urgency_indicator, axis=1)
data["password_mentioned"] = data.apply(password_mentioned, axis=1)
data["multiple_recipients"] = data.apply(multiple_recipients, axis=1)

data["time"] = pd.to_datetime(data["time"]).astype(int) / 10**9

feature_columns = [
    "spam_keywords",
    "content_density",
    "sender_credibility",
    "urgency_indicator",
    "password_mentioned",
    "multiple_recipients",
]

X = data[feature_columns]
y = data["spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

classifierKNN = KNeighborsClassifier(n_neighbors=3)
classifierKNN.fit(X_train, y_train)
KNN_predictions = classifierKNN.predict(X_test)
print(
    "K-Nearest Neighbor Test set score: {:.2f}".format(
        accuracy_score(y_test, KNN_predictions)
    )
)

classifierRndForest = RandomForestClassifier()
classifierRndForest.fit(X_train, y_train)
RndForest_predictions = classifierRndForest.predict(X_test)
print(
    "Random Forest Test set score: {:.2f}".format(
        accuracy_score(y_test, RndForest_predictions)
    )
)

classifierDTree = DecisionTreeClassifier()
classifierDTree.fit(X_train, y_train)
DTree_predictions = classifierDTree.predict(X_test)
print(
    "Decision Tree Test set score: {:.2f}".format(
        accuracy_score(y_test, DTree_predictions)
    )
)

classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)
NB_predictions = classifierNB.predict(X_test)
print(
    "Gaussian Naive Bayes Test set score: {:.2f}".format(
        accuracy_score(y_test, NB_predictions)
    )
)

classifierSVM = svm.LinearSVC()
classifierSVM.fit(X_train, y_train)
SVM_predictions = classifierSVM.predict(X_test)
print("SVM Test set score: {:.2f}".format(accuracy_score(y_test, SVM_predictions)))
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns




