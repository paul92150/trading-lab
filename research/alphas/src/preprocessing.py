import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_data(df, date_column="Date"):
    """
    Split a dataframe into train, validation, and test sets based on date ranges.

    Args:
        df (pd.DataFrame): Input dataframe with a date column.
        date_column (str): Name of the date column.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    train = df[df[date_column] < "2016-01-01"].reset_index(drop=True)
    val   = df[(df[date_column] >= "2016-01-01") & (df[date_column] < "2017-01-01")].reset_index(drop=True)
    test  = df[df[date_column] >= "2017-01-01"].reset_index(drop=True)

    return train, val, test


def scale_features(train_df, val_df, test_df, feature_cols):
    """
    Apply z-score standardization to features using statistics from the training set only.

    Args:
        train_df (pd.DataFrame): Training set.
        val_df (pd.DataFrame): Validation set.
        test_df (pd.DataFrame): Test set.
        feature_cols (list): List of feature column names to standardize.

    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
