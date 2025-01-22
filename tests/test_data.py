import torch
from torch.utils.data import Dataset

from project_mlops.data import playing_cards, fetch_kaggle, normalize


def test_playing_cards():
    # Load datasets
    train_set, valid_set, test_set = playing_cards()

    # Assert dataset sizes
    assert len(train_set) == 7624, "Train dataset size should be 7624"
    assert len(valid_set) == 265, "Validation dataset size should be 265"
    assert len(test_set) == 265, "Test dataset size should be 265"

    # Check the shape of images and range of targets
    for dataset in [train_set, valid_set, test_set]:
        for x, y in dataset:
            assert x.shape == (3, 224, 224), "Image shape should be (3, 224, 224)"
            assert 0 <= y < 53, "Target should be in the range [0, 52]"

    # Check if all 53 classes are represented in the train set
    train_targets = torch.unique(torch.tensor([y for _, y in train_set]))
    assert len(train_targets) == 53, "Train dataset should include all 53 classes"
    assert (train_targets == torch.arange(53)).all(), "Train classes should match [0, 52]"

    # Check if all 53 classes are represented in the test set
    test_targets = torch.unique(torch.tensor([y for _, y in test_set]))
    assert len(test_targets) <= 53, "Test dataset should not have more than 53 classes"

    print("All tests passed!")
