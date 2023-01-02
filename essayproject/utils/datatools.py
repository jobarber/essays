import gensim
import nltk
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split

np.random.seed(42)

nltk.download('punkt')

def get_oversampled_embeddings(X_train_vectors, y_train, method='smote'):
    """Do oversampling from embedding vectors.

    Parameters
    ----------
    X_train_vectors : np.array
    y_train : pd.Series or np.array or list
    method : str or None
        The method to use to do the oversampling. Can be
        adasyn, randomoversampler, smote, or None (for
        no oversampling).

    Returns
    -------
    oversampled_X : np.array with shape (batch size, embedding size)
        The augmented embeddings.
    oversampled y : np.array with shape (batch size,)
        The target labels.
    """
    method_classes = {'adasyn': ADASYN,
                      'randomoversampler': RandomOverSampler,
                      'smote': SMOTE}
    if method:
        oversampler = method_classes[method](random_state=42,
                                             sampling_strategy='not majority')
        X_train_vectors, y_train = oversampler.fit_resample(X_train_vectors, y_train)

    return X_train_vectors, y_train


class Word2VecEmbedder:
    """
    A class for managing word2vec embeddings.
    This class is similar to TfidfVectorizer.
    """
    def __init__(self):
        self.model = None

    def _construct_dataset(self, X):
        """Construct the dataset to train word2vec.

        Parameters
        ----------
        X : A sequence
            A sequence of texts.

        Returns
        -------
        tokenized_corpus : list of list of str
            The tokenized words within the tokenized
            sentences.
        """
        tokenized_corpus = []
        for text_sample in X:
            # Tokenize the sentences.
            tokenized_text = simple_preprocess(text_sample)
            tokenized_corpus.append(tokenized_text)
        return tokenized_corpus

    def fit(self, X):
        """Fit the word2vec model.

        Parameters
        ----------
        X : A sequence
            The text samples to be fit.

        Returns
        -------
        self.model : A gensim model
        """
        dataset = self._construct_dataset(X)
        self.model = gensim.models.Word2Vec(dataset,
                                            window=5,
                                            min_count=2,
                                            workers=4)
        self.model.save('modeldata/word2vec.kvmodel')
        return self.model

    def transform(self, X):
        """Transform samples X with word2vec.

        We will use the mean of all text vectors as
        the final embedding.

        Parameters
        ----------
        X : A sequence of texts
            The text samples to transform.

        Returns
        -------
        sample_embeddings : np.array (batch size, embedding size)
        """
        dataset = self._construct_dataset(X)
        self.model = gensim.models.Word2Vec(dataset, window=5, min_count=2)

        # Get embedding for each sample.
        sample_embeddings = []

        # For each sample ...
        for tokenized_text in dataset:
            text_embeddings = []

            # For each sentence in the sample ...
            for tokenized_sent in tokenized_text:

                # For each token in the sentence ...
                for token in tokenized_text:

                    # Be sure we actually have a vector for the word!
                    if token in self.model.wv.index_to_key:
                        vector = self.model.wv[token]
                        text_embeddings.append(vector)

            # Get the mean embedding for all sentence in
            # this particular sample.
            text_mean = np.mean(text_embeddings, axis=0)
            sample_embeddings.append(text_mean)
        return np.stack(sample_embeddings)