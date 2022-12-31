import mlflow

from essayproject.trialutils.autosklearntrial import AutoSklearnTrial


def run_baseline_trial(trait=1):
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
            run_ensemble_trial()
