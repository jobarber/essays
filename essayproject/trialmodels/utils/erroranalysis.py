import mlflow
import numpy as np
from lime.lime_text import LimeTextExplainer


def log_sample_explanations(predict_fn, X, y):
    """Create LIME explanation for all samples and log.

    Explanations are logged to the mlflow run.

    Parameters
    ----------
    predict_fn : A function
        It should take X as inputs and return y_pred.
    X : Sequence of texts
        These will be an input for the predict_fn above.
    y : Sequence of target values (int)
        The outputs of predict_fn(X) will be compared to
        y. Values are assumed to correspond to indices
        for the predicted vector.
    """
    explainer = LimeTextExplainer()
    sample_explanations = {'correct': [], 'wrong': []}
    for sample, target in zip(X, y):
        explanation = explainer.explain_instance(sample, predict_fn,
                                                 labels=y.unique().astype(int),
                                                 num_features=15)

        # Figure out whether this prediction belongs in the
        # correct or wrong bin.
        if np.argmax(explanation.predict_proba).item() == target:
            bin = sample_explanations['correct']
        else:
            bin = sample_explanations['wrong']

        # Log the attributes to the dict
        data = {}
        data['predict_proba'] = explanation.predict_proba.tolist()
        data['raw_prediction'] = explanation.score
        data['y_true'] = target
        data['explanation'] = explanation.as_list()
        bin.append(data)

    return sample_explanations
