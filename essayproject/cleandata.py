import os
import re

import pandas as pd

def clean(text):
    """Clean up the text.

    Some text cleaning is done by the sklearn or transformer
    algorithms themselves and does not need to be duplicated
    here.

    Parameters
    ----------
    text : str
        A single text that needs to be cleaned.

    Returns
    -------
    clean_text : str
    """
    # Assume that 3 or more spaces is a new paragraph.
    newline_text = re.sub(r'\s{3,}', '\n\n', text)

    # Some people use 2 spaces between sentences.
    spaced_text = re.sub(r' {2}', ' ', newline_text)

    # Add spaces where the space appears to be on the wrong
    # side of the period.
    normal_spaced_text = re.sub(r' .(\w)', r'. \1', spaced_text)

    # Deal with @-tokens. I see vague patterns for these tokens,
    # but no consistent pattern. So for now, let's remove the number
    # at the end of each token to reduce the total number of unique
    # @-tokens. We could use a language model cloze task to fill these.
    clean_text = re.sub(r'(?<!\w)([A-Z]+)\d+', r'\1', normal_spaced_text)

    return clean_text


def clean_column(dataset_path, column_name):
    """Clean a text column of a DataFrame with the `clean` function.

    Parameters
    ----------
    dataset_path : str
        Path to a csv file that can be loaded into Pandas.
    column_name : str
        The name of the text column to clean.

    This function saves the resulting data frame as a new data frame
    appending '_clean' to the filename.
    """
    df = pd.read_csv(dataset_path)
    df[column_name] = df[column_name].apply(clean)
    path, ext = os.path.splitext(dataset_path)
    df.to_csv(path + '_clean.csv')


if __name__ == '__main__':
    clean_column('data/sample_essay_dataset.csv', 'response_text')
