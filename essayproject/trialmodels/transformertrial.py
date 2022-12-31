import os
from pprint import pformat

import mlflow
import numpy as np
import torch
import torch.nn as nn
from pycm import ConfusionMatrix
from pycm.pycm_output import table_print
from sklearn.utils.class_weight import compute_class_weight
from torch import optim
from torch.utils.data import DataLoader

from trialmodels.transformercomponents.torchmodel import EssayModel
from trialmodels.transformercomponents.torchdataset import TorchDataset

torch.manual_seed(42)


class TransformerTrial:
    """A trial to train and test a transformer model."""

    def __init__(self,
                 dataset=TorchDataset,
                 input_column='response_text',
                 target_column='trait1_final_score',
                 bert_model='bert-base-uncased',
                 num_layers=12,
                 optimizer_name='AdamW',
                 criterion_name='CrossEntropyLoss',
                 epochs=3,
                 lr=2e-5,
                 batch_size=16,
                 pycm_metrics=['Overall_MCC', 'ACC', 'MCC'],
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 trial_name='bert_trial_1',
                 **optimizer_kwargs):
        """Constructor.

        Parameters
        ----------
        dataset : A torch Dataset user-defined class
            The dataset should define __getitem__ and __len__ or
            be an IterableDataset. This dataset should also take
            a kwarg train=True|False in its __init__ method.
            For convenience, let's also include the unique
            `labels` and `targets` as an attribute of the dataset
            as well for loss weight calculations.
        input_column : str
            The name of the column holding input text.
        target_column : str
            The name of the column holding the target labels.
        bert_model : A BertForSequenceClassification model
            A huggingface model from which we can load a pretrained
            model for finetuning.
        num_layers : int
            The number of hidden layers in the BERT model.
        optimizer_name : str
            The torch.optim optimizer to use for training.
        criterion_name : str
            The torch.optim optimizer to use for training.
        pycm_metrics : list of str
            The relevant pycm metrics to calculate.
        epochs : int
            Number of epochs to train for.
        lr : float
            Learning rate to apply in optimizer.
        batch_size : int
            The number of samples per training batch.
        device : A `torch.device` object or str
            This will allow us to run on a GPU when available.
        optimizer_kwargs : dict
            Any extra kwargs for the optimizer.
        """
        self.device = device
        self.train_dataset = dataset(train=True,
                                     input_column=input_column,
                                     target_column=target_column)
        self.batch_size = batch_size
        # Let's make the batch_size 1 for now, since we will be converting
        # each essay into multiple samples, effectively increasing the batch
        # to 25-50 sentences per batch.
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           # drop_last=True,
                                           shuffle=True)
        self.valid_dataset = dataset(train=False,
                                     input_column=input_column,
                                     target_column=target_column)
        self.valid_dataloader = DataLoader(self.valid_dataset,
                                           batch_size=self.batch_size,
                                           # drop_last=True,
                                           )

        num_labels = len(self.train_dataset.labels)
        model = EssayModel(bert_model=bert_model,
                           num_labels=num_labels,
                           num_layers=num_layers,
                           device=device)
        self.model = model.to(self.device)

        optimizer = getattr(optim, optimizer_name)
        self.optimizer = optimizer(self.model.parameters(), lr=lr, **optimizer_kwargs)
        criterion = getattr(nn, criterion_name)
        weights = compute_class_weight(class_weight='balanced',
                                       classes=np.unique(self.train_dataset.targets),
                                       y=self.train_dataset.targets)
        self.train_criterion = criterion(weight=torch.tensor(weights).float().to(self.device))
        self.valid_criterion = criterion()
        self.epochs = epochs
        self.metrics = pycm_metrics
        self._current_epoch = 1
        self.trial_name = trial_name

    def evaluate(self, test=True):
        """Evaluate the model on a dataset.

        Parameters
        ----------
        test : bool
            Whether to use test or train dataset for eval. This is
            useful for building learning curves etc.

        Returns
        -------
        metrics : dict
            {'metric_values': metric_values,
             'confusion_matrix': table_print(cm.classes, cm.table)}
        """
        # Make inferences on the dataset
        self.model.eval()
        running_loss = 0.
        y_test, y_pred = [], []
        for b, batch in enumerate(self.valid_dataloader):
            inputs, targets = batch
            with torch.no_grad():
                outputs = self.model(inputs)
                argmaxes = torch.argmax(outputs, dim=-1)
                for yt, yp in zip(targets, argmaxes):
                    y_test.append(yt.item())
                    y_pred.append(yp.item())
                loss = self.valid_criterion(outputs, targets.to(self.device).long())
                running_loss += loss.item()

            print(f'VALID Epoch: {self._current_epoch} Batch: {b + 1} '
                  f'Running Loss: {running_loss / (b + 1)} '
                  f'Loss: {loss.item()}')

        # Calculate the metrics
        cm = ConfusionMatrix(y_test, y_pred)
        metric_values = {}
        split = 'test' if test else 'train'
        for metric in self.metrics:
            metric_value = getattr(cm, metric)
            if isinstance(metric_value, dict):
                for k, v in metric_value.items():
                    k = float(k)
                    if str(v) != 'None':
                        metric_values[f'{metric}_{int(k)}_{split}'] = v
            elif metric:
                metric_values[f'{metric}_{split}'] = metric_value

        print({'metric_values': metric_values})
        print(table_print(cm.classes, cm.table))

        metrics = {'metric_values': metric_values,
                   'confusion_matrix': table_print(cm.classes, cm.table)}
        return metrics

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.
        for b, batch in enumerate(self.train_dataloader):
            inputs, targets = batch
            outputs = self.model(inputs)
            loss = self.train_criterion(outputs, targets.to(self.device).long())
            running_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            print(f'TRAIN Epoch: {self._current_epoch}'
                  f' Batch: {b + 1}/{(len(self.train_dataset) + 1) // self.batch_size}'
                  f' Running Loss: {running_loss / (b + 1)}'
                  f' Loss: {loss.item()}')

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch()
            validation_results = self.evaluate()
            self._log_model(validation_results)

        # Log the model (just once per run due to size limitations).
        with mlflow.start_run(nested=True, run_name=self.trial_name):
            artifact_dir = os.path.join('transformerdata', f'{self.trial_name}.pt')
            if not os.path.exists(artifact_dir):
                os.mkdir(artifact_dir)

            self.model.model.save_pretrained(artifact_dir)
            self.model.tokenizer.save_pretrained(artifact_dir)

            # # Log and register the model.
            # mlflow.pyfunc.log_model(self.model,
            #                         registered_model_name=self.trial_name)

    def _log_model(self, validation_results):
        with mlflow.start_run(nested=True, run_name=self.trial_name):
            mlflow.log_param('classifier', f'{self.trial_name}_{self._current_epoch}')
            mlflow.log_text(pformat(self.model.model.config),
                            f'config_{self.trial_name}_{self._current_epoch}.txt')

            # Log the optimizing metric.
            for test in [True, False]:
                mlflow.log_metrics(validation_results['metric_values'])
                mlflow.log_text(validation_results['confusion_matrix'],
                                f'confusion_matrix_{"test" if test else "train"}_'
                                f'{self.trial_name}_{self._current_epoch}.txt')

        self._current_epoch += 1
