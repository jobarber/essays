# Essays
This project is part of an interview assignment, so I will avoid sharing the data or problem specifics publicly.
The goal of this project is to train 2 models, 1 for each trait, in a given group of essays. I constructed a
solution that tracks, packages, and manages the models made, which are important for machine learning systems.
These models can also be run on other traits without too many modifications.

I currently have several models (in a variety of formats). 

## Current results
The dataset has a lot of overlap between the language that appears for scores 3 and 4--for both traits.
There could be a number of reasons for this. Perhaps annotators/graders did not consistently differentiate
between the two scores. I tried making this a binary problem to see if that would help (scores 1-3 on the one
hand and scores 4-6 for trait 1 or just 4 by itself for trait 2), but it helped only marginally. Here is a typical 
confusion matrix, where you can see the confusion in the center of the matrix.

<img width="469" alt="Screen Shot 2023-01-02 at 9 55 43 PM" src="https://user-images.githubusercontent.com/10589631/210304243-456e9a45-e9a9-42ea-a27e-4ddf8fe3ee4a.png">

I have tried Transformer models, sklearn models, and a number of other models through AWS. None of them 
really break 80% accuracy. I mention accuracy mainly because business stakeholders speak this language,
but the dataset is highly imbalanced and thus needs a better metric. The optimizing metric I chose is Matthews
correlation coefficient (MCC). Unlike the F1-score and other measures, the MCC is symetric and essentially
measures the correlation (in a manner similar to the Pearson correlation coefficient).

There is also a fair bit of noise in the data, which I have questions about. A number of tokens beginning
with an @ sign are scattered throughout the essays.

Here are the current results:

Overall_MCC_test	0.432
| Trait | Model | Ensemble? | Overall MCC Valid. Score | Train Loss | Valid. Loss |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Trait 1 | lda  | No | 0.432  | 0.247843 | 0.303251 |
| Trait 2 | lda  | No | 0.443  | 0.290339 | 0.330013 |
| Trait 1 | lda/mlp | Yes | 0.421 | (multiple) | (multiple) |
| Trait 2 | lda/mlp/gb/knn | Yes | 0.427  | (multiple) | (multiple) |

The models are too big to upload to GitHub, but I can look for another way to share them if desired.

## Entry points
I have created 4 entry points, which can be run from the command line.

First, activate the virtual environment and install the `requirements.txt`.

```
$ source /home/ubuntu/.virtualenvs/<company>/bin/activate
$ pip install -r requirements.txt
```

Then you can run any of the commands in the MLProject file. For example,
you can run any of these.

```
$ mlflow run essayproject -e clean --env-manager=local
$ mlflow run essayproject -e exploration --env-manager=local
$ mlflow run essayproject -e sklearn --env-manager=local
$ mlflow run essayproject -e autosklearn --env-manager=local
$ mlflow run essayproject -e transformers --env-manager=local
```

Results of each run can be seen through the mlflow UI:

```
$ mlflow ui -h 127.0.0.1:5000
```

Or UI results can be served from a remote server:

```
$ mlflow ui -h 0.0.0.0:5000
```

## Some initial analysis
Some trends appear in the data that could potentially be leveraged for a machine learning model. For example, you can see in the 
image below that the highest scored essays have fewer noun subjects and more punctuation than other essays for trait 1.

![DEP_proportions_trait_1](https://user-images.githubusercontent.com/10589631/210122864-12653e8b-5cd6-4d20-8d4d-9a046b30a22e.png)

That same trend also appears for trait 2.

![DEP_proportions_trait_2](https://user-images.githubusercontent.com/10589631/210122865-ee064c08-9cac-4430-bb42-f4222cb8b2fa.png)

In total, though, we see marginally more nouns and punctuation and fewer pronouns and verbs in the higher scored essays for both traits.

![POS_proportions_trait_1](https://user-images.githubusercontent.com/10589631/210122866-3d480dbd-0d3c-4bfb-a0ec-7c42ea549331.png)

![POS_proportions_trait_2](https://user-images.githubusercontent.com/10589631/210122867-ea270e10-d116-4ab2-9f4a-7218aa6f5935.png)

There is a general trend for higher scored essays to be longer as well, though not always.

![token_counts_trait_1](https://user-images.githubusercontent.com/10589631/210122868-85524ace-8c30-4005-bbc0-0c0035396ae0.png)
![token_counts_trait_2](https://user-images.githubusercontent.com/10589631/210122869-e617d8cf-f936-4529-a313-2b079d0c8aee.png)
