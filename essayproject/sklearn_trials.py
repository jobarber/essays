import numpy as np
from pycm import ConfusionMatrix
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

from transformdata import get_splits
from utils.datatools import get_oversampled_embeddings

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
    X_resampled, y_resampled = get_oversampled_embeddings(vectorized_X_train, y_train,
                                                          method='smote')

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


if __name__ == '__main__':
    for trait in [1, 2]:
        run_random_forest_trial(trait)
    #     with mlflow.start_run(run_name=f'sklearn_trait{trait}'):
    #         run_baseline_trial()
