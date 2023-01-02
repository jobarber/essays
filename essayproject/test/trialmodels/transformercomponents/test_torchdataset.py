import unittest

from essayproject.trialmodels.transformercomponents.torchdataset import TorchDataset


class TestTorchDataset(unittest.TestCase):
    """Test the basic functionality of the TorchDataset class."""

    def test_special_methods(self):
        """Test that __len__ and __getitem__ work properly."""
        dataset = TorchDataset(dataset_path='../../../data/test_sample_essay_dataset.csv',
                               train=True)
        self.assertGreaterEqual(len(dataset), 50)
        self.assertIsInstance(dataset[0], tuple)
        self.assertIsInstance(dataset[0][0], str)
        self.assertIsInstance(dataset[0][1], int)
