import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
target = "Class"
k = 5
size_features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "ConvexArea", "EquivDiameter"]
def load_split(path):
    df = pd.read_csv(path)
    X = df.drop(columns = [target])
    y = df[target]
    return X, y
def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average = "macro")
if __name__ == "__main__":
    X_train, y_train = load_split("dataset/processed/train.csv")
    X_test, y_test = load_split("dataset/processed/test.csv")
    experiments = {"Unscaled, all 16": (KNeighborsClassifier(n_neighbors=k), X_train.columns), "Unscaled, 6 size only": (KNeighborsClassifier(n_neighbors=k), size_features), "StandardScaler, all 16": (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), X_train.columns), "MinMaxScaler, all 16": (make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=k)), X_train.columns),}
    for name, (model, cols) in experiments.items():
        acc, f1 = evaluate(model, X_train[cols], y_train, X_test[cols], y_test)
        print(f"{name:<25} acc = {acc:.4f}  macroF1 = {f1:.4f}")