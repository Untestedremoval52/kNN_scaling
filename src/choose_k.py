import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from src.scaling_experiment import load_split, evaluate
k_values = list(range(1, 52, 2))
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
def score_k_values(X, y):
    train_means = []
    cv_means = []
    cv_stds = []
    for k in k_values:
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = k))
        result = cross_validate(model, X, y, cv = cv, return_train_score = True)
        train_means.append(result["train_score"].mean())
        cv_means.append(result["test_score"].mean())
        cv_stds.append(result["test_score"].std())
    return np.array(train_means), np.array(cv_means), np.array(cv_stds)
def plot_k_curve(train_means, cv_means, cv_stds, best_k):
    fig, ax = plt.subplots(figsize = (9, 5))
    ax.plot(k_values, train_means, marker = "o", label = "train accuracy")
    ax.plot(k_values, cv_means, marker = "o", label = "5-fold CV accuracy")
    ax.fill_between(k_values, cv_means - cv_stds, cv_means + cv_stds, alpha = 0.2)
    ax.axvline(best_k, linestyle = "--", color = "black", label = f"best k = {best_k}")
    plt.xlabel("k (number of neighbours)")
    plt.ylabel("accuracy")
    plt.title("Choosing k: Train vs 5-fold CV accuracy (scaled kNN)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/k_selection.png", dpi = 150)
    plt.show()
if __name__ == "__main__":
    X_train, y_train = load_split("dataset/processed/train.csv")
    X_test, y_test = load_split("dataset/processed/test.csv")
    train_means, cv_means, cv_stds = score_k_values(X_train, y_train)
    best_k = k_values[np.argmax(cv_means)]
    print(f"best k: {best_k}, CV accuracy: {cv_means.max():.4f}, avg CV std: {cv_stds.mean():.4f}")
    final_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
    acc, f1 = evaluate(final_model, X_train, y_train, X_test, y_test)
    print(f"Test accuracy: {acc:.4f}, macro F1: {f1:.4f}")
    plot_k_curve(train_means, cv_means, cv_stds, best_k)