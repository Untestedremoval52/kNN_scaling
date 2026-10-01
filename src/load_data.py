import pandas as pd
raw_path = "dataset/raw/Dry_Bean_Dataset.xlsx"
target = "Class"
def load_raw_dataset(path):
    df = pd.read_excel(path)
    return df
if __name__ == "__main__":
    df = load_raw_dataset(raw_path)
    print(df.shape)
    print(df.columns.tolist())
    print(df[target].value_counts())
    print(df.isna().sum().sum())
    print(df.duplicated().sum())
    print(df.describe().loc[["min", "max"]].T)