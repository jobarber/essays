"""This module explores the text of the dataset in order to
identify patterns that give context to the predictive
task.

Much more could be added here in terms of both analysis
and optimization, but this is merely a start.
"""

import os
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import spacy


class Exploration:
    """Explores the text of a single dataset and a single target attribute.

    All visualizations are saved to the visualizations/ folder.

    Example:

    >>> exploration = Exploration()
    >>> exploration.analyze()
    """

    def __init__(self, dataset_path='data/sample_essay_dataset_clean.csv',
                 input_column='response_text',
                 target_column='trait1_final_score',
                 exploration_name='trait_1'):
        """The constructor.

        Parameters
        ----------
        dataset_path : str
            The path to a dataset.
        input_column : str
            The name of the column in the dataset that contains the
            input text.
        target_column : str
            The name of the column in the dataset that contains the
            target variables.
        exploration_name : str
            The name to be used in visualization titles and filenames.
        """
        self.df = pd.read_csv(dataset_path)
        self.input_column = input_column
        self.target_column = target_column
        self.exploration_name = exploration_name
        self.nlp = spacy.load("en_core_web_sm")
        path_stem = Path(dataset_path).stem
        self.dir_ = os.path.join('visualizations', path_stem)

        if not os.path.exists(self.dir_):
            os.mkdir(self.dir_)

    def analyze(self):
        """Run a full analysis."""
        self.visualize_tokens(attribute='POS')
        self.visualize_tokens(attribute='DEP')
        self.visualize_response_lengths()

    def visualize_response_lengths(self):
        """Visualize response length in tokens for each label."""
        # Find the length of tokens for each response
        df = self.df.copy()
        df['Token Count'] = df[self.input_column].apply(lambda x: len(re.findall(r'[\w-]+', x)))
        df['Label'] = df[self.target_column]

        # Now visualize the dataframe.
        plt.figure(figsize=(12, 6))
        sns.violinplot(df, x='Label', y='Token Count')
        plt.xticks(rotation=90)
        plt.title(f'Token Counts for {self.exploration_name.replace("_", " ").title()}')
        plt.tight_layout()
        path = os.path.join(self.dir_, f'token_counts_{self.exploration_name}.png')
        plt.savefig(path)
        mlflow.log_artifact(path)
        plt.clf()

    def visualize_tokens(self, attribute='POS'):
        """Visualize a SpaCy token attribute with a bar plot.

        Parameters
        ----------
        attribute : str
            A SpaCy token attribute.
        """
        # Accumulate the data for all the labels into the same dataframe.
        attribute_df = pd.DataFrame()
        for label in self.df[self.target_column].unique():
            subdf = self._get_linguistic_counts(label, attribute=attribute)
            attribute_df = pd.concat([attribute_df, subdf], sort=False)

        # 'INTJ', 'X', 'NUM', 'SYM' are technically POS codes, but I don't
        # want to get too much into the weeds now.
        attribute_df = attribute_df[~attribute_df[attribute].isin(['INTJ', 'X', 'NUM', 'SYM'])]
        attribute_df = attribute_df.sort_values(by=['Label', 'Proportion'],
                                                ascending=[True, False])

        # Now visualize the dataframe.
        plt.figure(figsize=(12, 6))
        sns.barplot(attribute_df, x=attribute, y='Proportion', hue='Label')
        plt.xticks(rotation=90)
        plt.title(f'{attribute} Proportions for {self.exploration_name.replace("_", " ").title()}')
        plt.tight_layout()
        path = os.path.join(self.dir_, f'{attribute}_proportions_{self.exploration_name}.png')
        plt.savefig(path)
        mlflow.log_artifact(path)
        plt.clf()

    def _get_linguistic_counts(self, label, attribute='POS'):
        """Get the linguistic counts for an attribute by label.

        Parameters
        ----------
        label : str
            The target label for a given text.
        attribute : str
            A property of the SpaCy token (e.g., 'DEP').

        Returns
        -------
        subdf : pd.DataFrame
            A dataframe with attribute counts for the given label.
        """
        attribute_counter = Counter()

        # Reduce the df to a smaller df for this particular label.
        label_texts = self.df[self.df[self.target_column] == label][self.input_column]

        # Create a Counter for each attribute value.
        for text in label_texts:
            doc = self.nlp(text)
            for token in doc:
                attribute_counter[getattr(token, attribute.lower() + '_')] += 1

        # Now create a dataframe for the attribute, counts, and label.
        subdf = pd.DataFrame(attribute_counter.items(), columns=[attribute, 'Count'])
        subdf['Proportion'] = subdf['Count'] / subdf['Count'].sum()
        subdf['Label'] = int(label)
        return subdf


if __name__ == '__main__':
    with mlflow.start_run(run_name='data_exploration'):
        for trait in [1, 2]:
            exploration = Exploration(dataset_path='data/sample_essay_dataset_clean.csv',
                                      input_column='response_text',
                                      target_column=f'trait{trait}_final_score',
                                      exploration_name=f'trait_{trait}')
            exploration.analyze()
