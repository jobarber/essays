from pprint import pformat

import mlflow
import pandas as pd
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from pycm import ConfusionMatrix
from pycm.pycm_output import table_print
from sklearn.model_selection import train_test_split


class AutoSklearnTrial:
    """A single automl trial on a single dataset for a specific feature."""

    def __init__(self, df_path='data/sample_essay_dataset_clean.csv',
                 input_column='response_text',
                 target_column='trait1_final_score',
                 task='classification',
                 pycm_metrics=['Overall_MCC', 'ACC', 'MCC'],
                 trial_name='trial',
                 train_size=0.8,
                 dataset_name='trait_n',
                 **automl_kwargs):
        self.df = pd.read_csv(df_path)
        self.input_column = input_column
        self.target_column = target_column
        df_split = train_test_split(self.df[input_column].values.tolist(),
                                    self.df[target_column].to_list(),
                                    random_state=42,
                                    stratify=self.df[target_column].to_list(),
                                    train_size=train_size)
        self.X_train, self.X_test, self.y_train, self.y_test = df_split
        if task == 'classification':
            self.estimator = AutoSklearnClassifier
        else:
            self.estimator = AutoSklearnRegressor
        self.pycm_metrics = pycm_metrics
        self.trial_name = trial_name
        self.dataset_name = dataset_name
        self.automl_kwargs = automl_kwargs

    def eval(self, test=True):
        """Evaluate the model using the metric chosen for the trial.

        Parameters
        ----------
        test : bool
            Whether to run eval on the test or train dataset.

        Returns
        -------
        metrics : dict
            The chosen {metric: value} for the final model.
        """
        y_pred = self.automl.predict(self.X_test if test else self.X_train)

        # If we use a regressor, make the predictions discrete like the targets.
        if self.estimator is AutoSklearnRegressor:
            y_pred = self._discretize(y_pred)

        # Calculate the metric
        cm = ConfusionMatrix(self.y_test if test else self.y_train, y_pred)
        metric_values = {}
        split = 'test' if test else 'train'
        for metric in self.pycm_metrics:
            metric_value = getattr(cm, metric)
            if isinstance(metric_value, dict):
                for k, v in metric_value.items():
                    if str(v) != 'None':
                        metric_values[f'{metric}_{int(k)}_{split}'] = v
            elif metric:
                metric_values[f'{metric}_{split}'] = metric_value

        return {'metric_values': metric_values,
                'confusion_matrix': table_print(cm.classes, cm.table)}

    def train(self):
        """Conduct trials and create sklearn models."""

        # Set the defaults.
        automl_kwargs = dict(time_left_for_this_task=60 * 5,
                             per_run_time_limit=30,
                             n_jobs=-1,
                             resampling_strategy="cv",
                             resampling_strategy_arguments={"folds": 5})

        # Overwrite the defaults with user defined values.
        automl_kwargs.update(self.automl_kwargs)

        # Create and fit the estimator.
        self.automl = self.estimator(**automl_kwargs)
        self.automl.fit(self.X_train, self.y_train, dataset_name=self.dataset_name)

    def log(self):
        """Log model parameters nd metrics and register the model."""

        with mlflow.start_run(nested=True, run_name=self.trial_name):
            mlflow.log_param('classifier', self.trial_name)

            # Log the model dict in pretty fashion.
            mlflow.log_text(pformat(self.automl.show_models(), indent=4),
                            f'models_{self.trial_name}.txt')

            # Log the leaderboard as a text table.
            mlflow.log_text(self.automl.leaderboard(detailed=True,
                                                    ensemble_only=False,
                                                    top_k=10).to_string(),
                            f'leaderboard_{self.trial_name}.txt')

            # Log the optimizing metric.
            for test in [True, False]:
                test_eval = self.eval(test=test)
                mlflow.log_metrics(test_eval['metric_values'])
                mlflow.log_text(test_eval['confusion_matrix'],
                                f'confusion_matrix_{"test" if test else "train"}_{self.trial_name}.txt')

            # Log and register the model.
            mlflow.sklearn.log_model(self.trial_name, 'model',
                                     registered_model_name=self.trial_name)

    def _discretize(self, y_pred):
        """Convert float predictions into discrete predictions.

        Parameters
        ----------
        y_pred : numpy.ndarray (dtype('float64/32/16'))
            Predictions from a regression model.

        Returns
        -------
        y_pred : numpy.ndarray (dtype('int64/32/16'))
        """
        y_pred = y_pred.round()

        # For some regression models, predictions may be higher or lower
        # than the highest or lowest allowable values.
        y_pred[y_pred > self.df[self.target_column].max()] = self.df[self.target_column].max()
        y_pred[y_pred < self.df[self.target_column].min()] = self.df[self.target_column].min()

        return y_pred
