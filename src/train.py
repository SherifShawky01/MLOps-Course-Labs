"""
This module trains a churn prediction model and logs all necessary information to MLflow.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import mlflow
import mlflow.sklearn
import logging

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    Args:
        data (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]

    churn_maj_downsample = resample(
        churn_0, n_samples=len(churn_1), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_1])

def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        tuple: Preprocessed training and test sets
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]

    data = df.loc[:, filter_feat]
    data_bal = rebalance(data)

    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )

    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_test = col_transf.transform(X_test)

    return col_transf, X_train, X_test, y_train, y_test

def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test, **params):
    """
    Train a model and log to MLflow.

    Args:
        model_name (str): Name of the model
        model: Model instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    """
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_name", model_name)
        for param, value in params.items():
            mlflow.log_param(param, value)

        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        logger.info(f"{model_name} accuracy: {accuracy:.4f}")

        # Log model
        mlflow.sklearn.log_model(model, f"{model_name}_model")

        # Log confusion matrix as artifact
        conf_mat = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        disp.plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Churn Prediction")

    logger.info("Loading dataset...")
    df = pd.read_csv("D:/Python Labs ITI/mlops/MLOps-Course-Labs/dataset/Churn_Modelling.csv")

    col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    # Train Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    train_and_log_model(
        "Logistic Regression", log_reg, X_train, y_train, X_test, y_test, max_iter=1000
    )

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_log_model(
        "Random Forest", rf_model, X_train, y_train, X_test, y_test, n_estimators=100
    )

if __name__ == "__main__":
    main()
