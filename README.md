# Essays
This project is part of an interview assignment, so I will avoid sharing the data or problem specifics publicly.
The goal of this project is to train 2 models, 1 for each trait, in a given group of essays. I constructed a
solution that tracks, packages, and manages the models made, which are important for machine learning systems.
These models can also be run on other traits without too many modifications.

I currently have at least 2 models but have not finished exploring the ML algorithms using auto-sklearn.
I will give the algorithm/hyperparameter search more time to run tomorrow.

## Entry points
I have created 4 entry points, which can be run from the command line as follows (once the requisite packages have been installed):

```
$ mlflow run essayproject -e clean --env-manager=local  # (finishing up on my side before I push changes)
$ mlflow run essayproject -e exploration --env-manager=local
$ mlflow run essayproject -e sklearn --env-manager=local
$ mlflow run essayproject -e transformers --env-manager=local
```

## Some initial analysis (more to come)
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


## Work left to do
1. I intend to incorporate error analysis (via LIME) into this ML lifecycle.
2. I have tests to add as well, which I used with Transformers.
3. I have some cleaning up to do.
4. I will make my mlruns folder available somehow in order to share results and model artifacts.
