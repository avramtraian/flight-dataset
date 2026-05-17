from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@dataclass
class Dataset:
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.DataFrame

def load_dataset_from_csv(file_path: str) -> Dataset:
    df = pd.read_csv(file_path)
    return Dataset(
        df = df,
        X = df[[
            # "month",
            "is_holiday",
            "departure_congestion",
            "arrival_congestion",
            # "departure_weather",
            # "arrival_weather",
            "ticket_price",
        ]],
        y = df["is_delayed"]
    )

def model_analysis_confusion_matrix(confusion_matrix, plot_file_path: str):
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=["Fara intarziere", "Cu intarziere"]
    )
    display.plot()
    plt.title("Matrice de confuzie")
    plt.savefig(plot_file_path, dpi=300, bbox_inches="tight")

def model_analysis_performance_metrics(expected_y, predicted_y, plot_file_path: str):
    metrics = {
        "Acuratete": accuracy_score(expected_y, predicted_y),
        "Precizie": precision_score(expected_y, predicted_y),
        "Recall": recall_score(expected_y, predicted_y),
        "F1-score": f1_score(expected_y, predicted_y),
    }

    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.title("Grafic de performanta a modelului")
    plt.ylabel("Score")
    plt.savefig(plot_file_path, dpi=300, bbox_inches="tight")

def build_model(train_dataset_file_path: str):
    train_dataset = load_dataset_from_csv(train_dataset_file_path)
    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    model.fit(train_dataset.X, train_dataset.y)
    return model

def main():
    model = build_model("train.csv")
    
    test_dataset = load_dataset_from_csv("test.csv")
    predicted_y = model.predict(test_dataset.X)

    conf_mat = confusion_matrix(test_dataset.y, predicted_y)
    model_analysis_confusion_matrix(conf_mat, "confusion_matrix.png")
    model_analysis_performance_metrics(test_dataset.y, predicted_y, "performance.png")

if __name__ == "__main__":
    main()
