import mlflow

from trialmodels.transformertrial import TransformerTrial


def run_baseline_trial(trait=1):
    """Run Transformer trial using reasonable params."""
    trial = TransformerTrial(roberta_model='roberta-base',
                             target_column=f'trait{trait}_final_score',
                             num_layers=12,
                             optimizer_name='AdamW',
                             criterion_name='CrossEntropyLoss',
                             epochs=10,
                             lr=1.5e-5,
                             batch_size=8,
                             trial_name=f'roberta_baseline_trial_trait_{trait}',
                             weight_decay=3e0)
    trial.train()
    trial.evaluate(test=True)
    trial.evaluate(test=False)
    del trial


def run_cased_trial(trait=1):
    """Run Transformer trial using reasonable params."""
    trial = TransformerTrial(roberta_model='roberta-base',
                             target_column=f'trait{trait}_final_score',
                             num_layers=12,
                             optimizer_name='AdamW',
                             criterion_name='CrossEntropyLoss',
                             epochs=10,
                             lr=1.5e-5,
                             batch_size=8,
                             trial_name=f'roberta_baseline_trial_trait_{trait}',
                             weight_decay=3e0)
    trial.train()
    trial.evaluate(test=True)
    trial.evaluate(test=False)
    del trial


def run_base_small_trial(trait=1):
    """Run Transformer trial using reasonable params."""
    trial = TransformerTrial(roberta_model='roberta-base',
                             target_column=f'trait{trait}_final_score',
                             num_layers=6,
                             optimizer_name='AdamW',
                             criterion_name='CrossEntropyLoss',
                             epochs=10,
                             lr=1.5e-5,
                             batch_size=8,
                             trial_name=f'roberta_baseline_trial_trait_{trait}',
                             weight_decay=3e0)
    trial.train()
    trial.evaluate(test=True)
    trial.evaluate(test=False)
    del trial


def run_cased_small_trial(trait=1):
    """Run Transformer trial using reasonable params."""
    trial = TransformerTrial(roberta_model='roberta-base',
                             target_column=f'trait{trait}_final_score',
                             num_layers=6,
                             optimizer_name='AdamW',
                             criterion_name='CrossEntropyLoss',
                             epochs=10,
                             lr=1.5e-5,
                             batch_size=8,
                             trial_name=f'roberta_baseline_trial_trait_{trait}',
                             weight_decay=3e0)
    trial.train()
    trial.evaluate(test=True)
    trial.evaluate(test=False)
    del trial


if __name__ == '__main__':
    for trait in [1, 2]:
        with mlflow.start_run(run_name=f'transformer_trait{trait}'):
            run_baseline_trial(trait=trait)
            run_base_small_trial(trait=trait)
