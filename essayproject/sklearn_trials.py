import mlflow
import numpy as np
from pycm import ConfusionMatrix
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils.erroranalysis import log_sample_explanations
from transformdata import get_splits
from utils.datatools import get_oversampled_embeddings, Word2VecEmbedder

np.random.seed(42)


def run_sklearn_trial(trait=1):
    """Run trial using an logistic regression baseline.

    Parameters
    ----------
    trait : int or str
        The trait to use as the target.
    """
    X_train, X_test, y_train, y_test = get_splits(trait)

    # Perform TFIDF vectorization followed by random oversampling
    vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), stop_words=None)  # 1, 2 and 1, 3
    vectorized_X_train = vectorizer.fit_transform(X_train)
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

    pipeline = Pipeline([('vectorizer', vectorizer),
                         ('selector', selector),
                         ('classifier', classifier)])

    # Evaluate the classifier
    y_pred = pipeline.predict(X_test)

    log_sample_explanations(pipeline.predict_proba, X_test, y_test)

    # Get the metrics
    cm = ConfusionMatrix(y_test.values, y_pred)
    metric_values = []
    for metric in ['Overall_MCC', 'ACC', 'MCC']:
        metric_value = getattr(cm, metric)
        metric_values.append([metric, metric_value])

    # Log the metric values
    with mlflow.start_run(nested=True, run_name=f'trait{trait}_metrics'):
        mlflow.log_dict(metric_values)


if __name__ == '__main__':
    for trait in [1, 2]:
        with mlflow.start_run(run_name=f'sklearn_trait{trait}'):
            run_sklearn_trial(trait)
