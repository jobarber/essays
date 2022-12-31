import mlflow

from trial_utils.autosklearntrial import AutoSklearnTrial
from trial_utils.sklearnbaseline import LogisticRegressionTrial


def run_baseline_trial(trait=1):
    """Run trial using an logistic regression baseline.

    Parameters
    ----------
    trait : int or str
        The trait to use as the target.
    """
    for trait in [1, 2]:
        trial = LogisticRegressionTrial(df_path='data/sample_essay_dataset.csv',
                                        input_column='response_text',
                                        target_column=f'trait{trait}_final_score',
                                        pycm_metrics=['Overall_MCC', 'ACC', 'MCC'],
                                        trial_name=f'logistic regression_trial_trait{trait}',
                                        train_size=0.8,
                                        dataset_name=f'trait_{trait}',
                                        lowercase=True if trait == 1 else False)
        trial.train()
        trial.log()


def run_autosklearn_trial(trait=1):
    """Run trial without using auto-sklearn ensembles.

    Run for both classification and regression.
    """
    for task in ('classification', 'regression'):
        trial = AutoSklearnTrial(target_column=f'trait{trait}_final_score',
                                 task=task,
                                 pycm_metrics=['Overall_MCC', 'ACC', 'MCC'],
                                 trial_name='baseline_trial',
                                 dataset_name=f'trait{trait}',
                                 ensemble_kwargs={'ensemble_size': 1},
                                 time_left_for_this_task=60 * 5,
                                 per_run_time_limit=30)
        trial.train()
        trial.log()


def run_ensemble_trial(trait=1):
    """Run trial using auto-sklearn ensembles.

    Run for both classification and regression.
    """
    for task in ('classification', 'regression'):
        trial = AutoSklearnTrial(target_column=f'trait{trait}_final_score',
                                 task=task,
                                 pycm_metrics=['Overall_MCC', 'ACC', 'MCC'],
                                 trial_name='ensemble_trial',
                                 dataset_name=f'trait{trait}',
                                 time_left_for_this_task=60 * 5,
                                 per_run_time_limit=30)
        trial.train()
        trial.log()


if __name__ == '__main__':
    for trait in [1, 2]:
        with mlflow.start_run(run_name=f'sklearn_trait{trait}'):
            run_baseline_trial()
            run_autosklearn_trial()
            run_ensemble_trial()
