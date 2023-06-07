import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split(data_dir, train_file_path, test_file_path, split):
    data = pd.read_csv(data_dir)

    train_df, test_df = train_test_split(data, test_size=split)

    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)
