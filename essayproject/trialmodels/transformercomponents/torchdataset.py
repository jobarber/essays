import pandas as pd
from torch.utils.data import Dataset


class TorchDataset(Dataset):

    def __init__(self, dataset_path='data/sample_essay_dataset_clean.csv',
                 input_column='response_text',
                 target_column='trait1_final_score',
                 train=True):
        """Constructor.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame with train and validation data in it.
        input_column : str
            The name of the column holding input text.
        target_column : str
            The name of the column holding the target labels.
        train : bool
            Whether this dataset is a training or validation dataset.
        """
        self.input_column = input_column
        self.target_column = target_column
        df = pd.read_csv(dataset_path).sample(frac=1., random_state=42)
        self.labels = df[target_column].unique()
        self.targets = df[target_column].values
        train_df, valid_df = self._get_train_test_split(df, input_column, target_column)
        self.df = train_df if train else valid_df
        if train:
            supplement = pd.read_csv('data/sample_essay_dataset_supplement.csv')
            self.df = pd.concat([self.df, supplement], ignore_index=True)
            self.df = self.df.sample(frac=1.)

    def _get_train_test_split(self, df, input_column, target_column):
        """Do a repeatable stratified random sample split.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame with train and validation data in it.
        input_column : str
            The name of the column holding input text.
        target_column : str
            The name of the column holding the target labels.

        Returns
        -------
        (train_df, valid_df) : tuple of pd.DataFrames
            The two should have no overlap.
        """
        # We need to do repeatable stratified random sampling.
        train_df = df.groupby(self.target_column,
                              group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=42))

        # Be sure that there is no data leakage.
        merged_df = df.merge(train_df.drop_duplicates(),
                             on=[input_column, target_column],
                             how='left', indicator=True)
        # Filter out the duplicates
        valid_df = merged_df[merged_df['_merge'] == 'left_only']

        return train_df, valid_df

    def __getitem__(self, item):
        """Get the corresponding input and target for the item index.

        Parameters
        ----------
        item : int
            An index in the dataset.

        Returns
        -------
        input_text, target : text, torch.LongTensor
            The input text and target corresponding to this index.
        """
        input_text = self.df.iloc[item][self.input_column]
        target = self.df.iloc[item][self.target_column]
        return input_text, int(target - 1)

    def __len__(self):
        return len(self.df)
