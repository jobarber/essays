import random
import re

import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

torch.manual_seed(39)
np.random.seed(39)
random.seed(39)


class EssayModel(nn.Module):
    """A convenience class that allows for model experimentation.

    This class separates essays into individual sentences when
    possible. The forward method is run on the sentences of a
    single essay in parallel. We then manually creates the BERT
    pooling layers, so that we can experiment with aggregating
    the sentences in different ways (before the final layer,
    after the final layer, etc.). We could technically use
    BertForSequenceClassification if we only wanted to aggregate
    after the final layer.

    The softmax is left off in case we apply it with the loss
    function.

    Example:
        >>> essay_model = EssayModel()
        >>> essay_model(['Here is a first essay. It has two sentences.',
                         'And here is another essay with only one.'])
        tensor([[ 0.2087, -0.1414,  0.0082,  0.0361, -0.1263,  0.0408],
                [ 0.0044, -0.0420, -0.2267, -0.0189, -0.0484, -0.3093]],
               grad_fn=<StackBackward0>)
    """

    def __init__(self,
                 roberta_model='roberta-base',
                 num_labels=6,
                 num_layers=6,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """Set up BERT model and additional layers for this task.

        bert_model : A BertForSequenceClassification model
            A huggingface model from which we can load a pretrained
            model for finetuning.
        num_labels : int
            The number of labels in the target.
        num_layers : int
            The number of hidden layers in the BERT model.
        device : A `torch.device` object or str
            This will allow us to run on a GPU when available.
        """
        super().__init__()
        self.device = device
        model = RobertaModel.from_pretrained(roberta_model,
                                             num_labels=num_labels,
                                             num_hidden_layers=num_layers)
        self.model = model.to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model)

        # Set up the pooling layers. We need to do this ourselves,
        self.pooler = nn.Linear(in_features=self.model.config.hidden_size,
                                out_features=self.model.config.hidden_size, bias=True)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = nn.Linear(in_features=self.model.config.hidden_size,
                                    out_features=num_labels, bias=True)

    def forward(self, essays):
        """This method makes inferences for essays.

        The method gets outputs for one model at a time,
        since we split each essay into its component
        sentences (as much as possible).

        Parameters
        ----------
        essays : list of str
            The list of essays in a single batch

        Returns
        -------
        classification_distributions : torch.tensor
            This tensor will have shape (number of essays, number of labels).
        """
        essays = [essays] if isinstance(essays, str) else essays
        classification_distributions = []
        sep_token = self.tokenizer.sep_token
        for essay in essays:
            # Add the number of sentences as an input
            num_sentences = len(re.findall(r'(?<![.\d])[.?!](?![.\d])', essay))
            essay = f'{essay}{sep_token}{sep_token}sentences: {num_sentences}'

            # Now get the model outputs
            tokenized_X = self._tokenize(essay, max_length=64).to(self.device)
            outputs = self.model(**tokenized_X)
            last_hidden_state = outputs.last_hidden_state

            # Feed only the CLS token output into the pooler,
            # and mean across the entire batch since it represents
            # one essay.
            reduced = last_hidden_state[:, 0, :].mean(dim=0)
            pooled = self.pooler(reduced)
            activated = self.pooler_activation(pooled)
            with_dropout = self.dropout(activated)
            classified = self.classifier(with_dropout)
            classification_distributions.append(classified)

        return torch.stack(classification_distributions)

    def _tokenize(self, text, max_length=64):
        """Tokenize sentences and maintain a reasonable max length.

        One of the downfalls of Transformers is their latency for
        long sequences, since attention has quadratic complexity
        and since Transformers have a maximum allowable length that
        will fit into the model.

        The idea here is to reduce an essay of, say, 1,000 tokens
        down to 16-20 samples that are no longer than 64 tokens
        each. This ensures (a) that we do not need to truncate the
        essay and (b) that we can pass in samples that are much
        smaller than the original essay and that will be much
        to run.

        Parameters
        ----------
        sentence : str
            A single text sample consisting of multiple sentences.
        max_length : int
            The maximum length allowable before a sample will be
            wrapped.

        Returns
        -------
        tokenized : dict
            The tokenized inputs, which includes input_ids,
            the attention mask, etc.
        """
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        full_stop_id = self.tokenizer.encode('.', add_special_tokens=False)[0]
        batch_sentence_ids = []

        # Sometimes there may not be a full stop in an essay.
        if full_stop_id in input_ids:
            next_full_stop_index = input_ids.index(full_stop_id)
        else:
            next_full_stop_index = max_length - 2

        # Now we need to divide the input_ids into segments not to
        # exceed the maximum length.
        while input_ids:

            # Leave 2 characters for special tokens and 1 for ".".
            if next_full_stop_index <= max_length - 3:
                batch_sentence_ids.append(input_ids[:next_full_stop_index + 1])
                input_ids = input_ids[next_full_stop_index + 1:]
            elif next_full_stop_index > max_length - 3:
                batch_sentence_ids.append(input_ids[:max_length - 2])
                input_ids = input_ids[max_length - 2:]

            # The input_ids have shrunk. Are there any remaining
            # full stops?
            if full_stop_id in input_ids:
                next_full_stop_index = input_ids.index(full_stop_id)
            else:
                next_full_stop_index = max_length - 2

        # We can do better here, but for now we will decode and re-encode
        # the sentences, knowing that we do not have any samples longer
        # than the max length.
        decoded = [self.tokenizer.decode(sentence_ids) for sentence_ids in batch_sentence_ids]
        encoded = self.tokenizer(decoded, return_tensors='pt', padding=True,
                                 max_length=max_length)

        return encoded
