import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.scaling_experiment import load_split
best_k = 19
if __name__ == "__main__":
    X_train, y_train = load_split("dataset/processed/train.csv")
    X_test, y_test = load_split("dataset/processed/test.csv")
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = best_k))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(classification_report(y_test, pred, digits = 3))
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, pred, labels = labels)
    print(pd.DataFrame(cm, index = labels, columns = labels))
    total_error = (pred != y_test).sum()
    i, j = labels.index("SIRA"), labels.index("DERMASON")
    pair_errors = cm[i, j] + cm[j, i]
    print(f"Total error: {total_error}, Pair errors: {pair_errors}, Ratio: {pair_errors}/{total_error}")
    fig, ax = plt.subplots(figsize = (8, 7))
    ConfusionMatrixDisplay(cm, display_labels = labels).plot(ax = ax, cmap = "Blues", colorbar = False)
    plt.xticks(rotation = 45)
    ax.set_title("Confusion Matrix: Scaled kNN, k = 19 (test set)")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi = 150)
    plt.show()