import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='kaustubh-rathod8', repo_name='MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/kaustubh-rathod8/MLFlow.mlflow")

# Load Wine Dataset
wine = load_wine()
x = wine.data
y = wine.target

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# Define the params for RF Model
max_depth = 10
n_estimators = 10

mlflow.set_experiment('MLOPS-Exp1')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Creating a Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save Plot
    plt.savefig("Confusion-matrix.png")

    # Log Artifacts using MlFlow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # Tags
    mlflow.set_tags({"Author": 'Kaustubh', "Project": 'Wine Classification'})

    # Log the Model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    print(accuracy)