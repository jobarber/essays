import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)


def get_splits(trait, lambda_apply_fn=None):
    """Get the train-validation splits for X and y.

    Parameters
    ----------
    trait : int or str
        The number of the trait on which to train the model.
    lambda_apply_fn : a lambda function or None
        A lambda function to reduce the number of dimensions
        of the problem. For example, to convert the task into
        a binary task along the threshold where all models
        have trouble, you could use
        `lambda_apply_fn=lambda x: int(x >= 4)`.

    Returns
    -------
    X_test : np.array
    X_train : np.array
    y_test : np.array
    y_train : np.array
    """
    df = pd.read_csv('data/sample_essay_dataset_clean.csv')
    df[f'trait{trait}_final_score'] = df[f'trait{trait}_final_score'].astype(int) - 1

    # Convert the number of classes if desired.
    if lambda_apply_fn:
        df[f'trait{trait}_final_score'] = df[f'trait{trait}_final_score'].apply(lambda_apply_fn)

    # Split the data
    X, y = df['response_text'], df[f'trait{trait}_final_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True, stratify=y)

    # Add the generated augmentation samples.
    supplemental_df = pd.read_csv('data/sample_essay_dataset_supplement.csv')
    X_train = pd.concat([X_train, supplemental_df['response_text']],
                        ignore_index=True)
    y_train = pd.concat([y_train, supplemental_df[f'trait{trait}_final_score']],
                        ignore_index=True)

    return X_train, X_test, y_train, y_test
