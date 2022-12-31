from pprint import pformat

import mlflow
import pandas as pd
from pycm import ConfusionMatrix
from pycm.pycm_output import table_print
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class LogisticRegressionTrial:
    """A baseline trial class."""

    def __init__(self, df_path='data/sample_essay_dataset_clean.csv',
                 input_column='response_text',
                 target_column='trait1_final_score',
                 pycm_metrics=['Overall_MCC', 'ACC', 'MCC'],
                 trial_name='logistic regression_trial_trait1',
                 train_size=0.8,
                 dataset_name='trait_1',
                 lowercase=False):
        self.df = pd.read_csv(df_path)
        self.input_column = input_column
        self.target_column = target_column
        # We need to do repeatable stratified random sampling.
        train_df = self.df.groupby(self.target_column,
                                   group_keys=False).apply(lambda x: x.sample(frac=train_size, random_state=42))
        supplement = pd.read_csv('data/sample_essay_dataset_supplement.csv')
        train_df = pd.concat([train_df, supplement], ignore_index=True)
        # Be sure that there is no data leakage.
        merged_df = self.df.merge(train_df.drop_duplicates(),
                                  on=[input_column, target_column],
                                  how='left', indicator=True)
        # Filter out the duplicates
        valid_df = merged_df[merged_df['_merge'] == 'left_only']

        self.X_train = train_df[input_column].to_list()
        self.X_test = valid_df[input_column].to_list()
        self.y_train = train_df[target_column].values
        self.y_test = valid_df[target_column].values

        self.pycm_metrics = pycm_metrics
        self.trial_name = trial_name
        self.dataset_name = dataset_name

        tfidf = TfidfVectorizer(lowercase=lowercase, ngram_range=(1, 4))
        model = LogisticRegression(class_weight='balanced')
        self.pipeline = Pipeline(steps=[('vectorizer', tfidf),
                                        ('classifier', model)])

    def train(self):
        """Create and fit the LogisticRegression model."""
        self.pipeline = self.pipeline.fit(self.X_train, self.y_train)

    def eval(self):
        """Evaluate the sklearn model."""
        y_pred = self.pipeline.predict(self.X_test)

        # Calculate the metric
        cm = ConfusionMatrix(self.y_test, y_pred)
        metric_values = {}

        for metric in self.pycm_metrics:
            metric_value = getattr(cm, metric)
            if isinstance(metric_value, dict):
                for k, v in metric_value.items():
                    if str(v) != 'None':
                        metric_values[f'{metric}_{int(k)}'] = v
            elif metric:
                metric_values[f'{metric}_test'] = metric_value

        return {'metric_values': metric_values,
                'confusion_matrix': table_print(cm.classes, cm.table)}

    def log(self):
        """Log model parameters nd metrics and register the model."""
        with mlflow.start_run(nested=True, run_name=self.trial_name):
            mlflow.log_param('classifier', self.trial_name)

            # Log the optimizing metric.
            for test in [True, False]:
                test_eval = self.eval()
                mlflow.log_metrics(test_eval['metric_values'])
                mlflow.log_text(test_eval['confusion_matrix'],
                                f'confusion_matrix_{"test" if test else "train"}_{self.trial_name}.txt')

            # Log and register the model.
            mlflow.sklearn.log_model(self.pipeline, 'logistic regression_model',
                                     registered_model_name=self.trial_name)
