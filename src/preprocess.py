import pandas as pd
from sklearn.model_selection import train_test_split
from src.load_data import load_raw_dataset, raw_path, target
random_state = 42
test_size = 0.2
def drop_duplicates(df):
    before = len(df)
    print(df[df.duplicated()][target].value_counts())
    df = df.drop_duplicates()
    print(f"Number of rows removed: {before - len(df)}")
    return df
def split_data(df):
    X = df.drop(columns = [target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = y, random_state = random_state)
    return X_train, X_test, y_train, y_test
if __name__ == "__main__":
    df = load_raw_dataset(raw_path)
    df = drop_duplicates(df)
    X_train, X_test, y_train, y_test = split_data(df)
    train_df = pd.concat([X_train, y_train], axis = 1)
    test_df = pd.concat([X_test, y_test], axis = 1)
    train_df.to_csv("dataset/processed/train.csv", index=False)
    test_df.to_csv("dataset/processed/test.csv", index=False)
    print(df.shape)
    print(X_train.shape, X_test.shape)
    print(y_train.value_counts(normalize = True).round(3))
    print(y_test.value_counts(normalize = True).round(3))
    print(train_df.columns[-1])