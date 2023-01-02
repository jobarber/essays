import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from pycm import ConfusionMatrix
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier, GradientBoostingClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

np.random.seed(42)


def run_random_forest_trial(trait=1):
    """Run trial using an logistic regression baseline.

    Parameters
    ----------
    trait : int or str
        The trait to use as the target.
    """
    X_test, X_train, y_test, y_train = get_splits(trait)

    # Perform TFIDF vectorization followed by random oversampling
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), stop_words=None)  # 1, 2 and 1, 3
    vectorized_X_train = tfidf_vectorizer.fit_transform(X_train)
    X_resampled, y_resampled = do_random_oversampling(vectorized_X_train, y_train)

    # Select the feature from TFIDF that will most help the model
    selector = SelectFromModel(estimator=LinearSVC(penalty="l1", dual=False)).fit(X_resampled, y_resampled)
    X_train_new = selector.transform(X_resampled)

    # Fit the classifier
    classifier = BaggingClassifier(RandomForestClassifier(),
                                      max_samples=0.5, max_features=0.5,
                                      random_state=47)  # 42
    classifier.fit(X_train_new, y_resampled)

    # Evaluate the classifier
    vectorized_X_test = tfidf_vectorizer.transform(X_test)
    X_test_new = selector.transform(vectorized_X_test)
    y_pred = classifier.predict(X_test_new)

    # Get the metrics
    cm = ConfusionMatrix(y_test.values, y_pred)
    for metric in ['Overall_MCC', 'ACC', 'MCC']:
        metric_value = getattr(cm, metric)
        print(trait, metric, metric_value)


def do_random_oversampling(vectorized_X_train, y_train):
    ros = RandomOverSampler(random_state=42, sampling_strategy='not majority')
    X_resampled, y_resampled = ros.fit_resample(vectorized_X_train, y_train)
    return X_resampled, y_resampled


def get_splits(trait, lambda_apply_fn=lambda x: int(x >= 4)):
    """Get the train-validation splits for X and y.

    Parameters
    ----------
    trait : int or str
        The number of the trait on which to train the model.
    lambda_apply_fn : a lambda function or None
        A lambda function to reduce the number of dimensions
        of the problem.

    Returns
    -------
    X_test : np.array
    X_train : np.array
    y_test : np.array
    y_train : np.array
    """
    df = pd.read_csv('data/sample_essay_dataset_clean.csv')
    if lambda_apply_fn:
        df[f'trait{trait}_final_score'] = df[f'trait{trait}_final_score'].apply(lambda_apply_fn)
    X, y = df['response_text'], df[f'trait{trait}_final_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True, stratify=y)

    # Add the generated augementation samples.
    supplemental_df = pd.read_csv('data/sample_essay_dataset_supplement.csv')
    X_train = pd.concat([X_train, supplemental_df['response_text']],
                        ignore_index=True)
    y_train = pd.concat([y_train, supplemental_df[f'trait{trait}_final_score']],
                        ignore_index=True)

    return X_test, X_train, y_test, y_train


if __name__ == '__main__':
    for trait in [1, 2]:
        run_random_forest_trial(trait)
    #     with mlflow.start_run(run_name=f'sklearn_trait{trait}'):
    #         run_baseline_trial()
