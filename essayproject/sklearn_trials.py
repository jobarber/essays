import numpy as np
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from trialmodels.sklearntrial import run_sklearn_trial
from trialmodels.utils.datatools import get_oversampled_embeddings

np.random.seed(42)


def create_random_forest_pipeline(X_train, y_train):
    """Create a specific pipeline for evaluation.

    Parameters
    ----------
    X_train : np.array
    y_train : np.array

    Returns
    -------
    pipeline : A trained sklearn.pipeline.Pipeline object
    """
    # Perform TFIDF vectorization followed by random oversampling
    vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), stop_words=None)  # 1, 2 and 1, 3
    vectorized_X_train = vectorizer.fit_transform(X_train)
    X_resampled, y_resampled = get_oversampled_embeddings(vectorized_X_train, y_train,
                                                          method='smote')

    # Select the feature from TFIDF that will most help the model
    selector = SelectFromModel(estimator=LinearSVC(penalty="l1", dual=False, max_iter=3_000)).fit(X_resampled,
                                                                                                  y_resampled)
    X_train_new = selector.transform(X_resampled)

    # Fit the classifier. Random forest is already a bagging classifier.
    # But by adding the bagging classifier wrapper around it,
    # I am getting better results.
    classifier = BaggingClassifier(RandomForestClassifier(),
                                   max_samples=0.5, max_features=0.5,
                                   random_state=47)  # 42
    classifier.fit(X_train_new, y_resampled)
    pipeline = Pipeline([('vectorizer', vectorizer),
                         ('selector', selector),
                         ('classifier', classifier)])
    return pipeline


def create_gradient_boosting_pipeline(X_train, y_train):
    """Create a specific pipeline for evaluation.

    Parameters
    ----------
    X_train : np.array
    y_train : np.array

    Returns
    -------
    pipeline : A trained sklearn.pipeline.Pipeline object
    """
    # Perform TFIDF vectorization followed by random oversampling
    vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), stop_words=None)  # 1, 2 and 1, 3
    vectorized_X_train = vectorizer.fit_transform(X_train)
    X_resampled, y_resampled = get_oversampled_embeddings(vectorized_X_train, y_train,
                                                          method='smote')

    # Select the feature from TFIDF that will most help the model
    selector = SelectFromModel(estimator=LinearSVC(penalty="l1", dual=False, max_iter=3_000)).fit(X_resampled,
                                                                                                  y_resampled)
    X_train_new = selector.transform(X_resampled)

    # Fit the classifier. Random forest is already a bagging classifier.
    # But by adding the bagging classifier wrapper around it,
    # I am getting better results.
    classifier = GradientBoostingClassifier(learning_rate=5e-2,
                                            n_estimators=200,
                                            subsample=0.5,
                                            random_state=47)  # 42
    classifier.fit(X_train_new, y_resampled)
    pipeline = Pipeline([('vectorizer', vectorizer),
                         ('selector', selector),
                         ('classifier', classifier)])
    return pipeline


if __name__ == '__main__':
    for trait in [1, 2]:
        run_sklearn_trial(trait, create_random_forest_pipeline, explain=True)
        run_sklearn_trial(trait, create_gradient_boosting_pipeline, explain=True)
