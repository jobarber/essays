import unittest

from essayproject.trialmodels.transformercomponents.torchmodel import EssayModel


class TestEssayModel(unittest.TestCase):
    """Test that the EssayModel behaves as expected."""

    def test_prediction_shape(self):
        """Test that the predictions are the shape we expect."""
        essay_model = EssayModel(num_labels=3, num_layers=2)
        predictions = essay_model(['Here is a first essay. It has two sentences.',
                                   'And here is another essay with only one.'])
        self.assertEqual((2, 3), predictions.shape)

    def test_tokenize_method(self):
        """Test that the tokenize method works."""
        essay_model = EssayModel(num_labels=4, num_layers=1)
        tokenized = essay_model._tokenize('This is a test to see if the tokenization'
                                          ' process works. This is a single text.'
                                          ' How many rows will we get back?',
                                          max_length=5)
        self.assertIn('input_ids', tokenized)
        self.assertGreater(tokenized['input_ids'].shape[0], 2)
        self.assertEqual(5, tokenized['input_ids'].shape[1])
