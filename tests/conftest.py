import os

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(REPO_ROOT, "DRDNA", "b10c32")


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def profile():
    from src.detection import Profile

    return Profile.load(PROFILE_DIR)


@pytest.fixture(scope="session")
def resnet18_model(device):
    from src.models.resnet import resnet18

    model = resnet18(pretrained=True, progress=True)
    return model.to(device)


@pytest.fixture(scope="session")
def correct_data_loader():
    """The first N samples of the repo's checked-in correct_data.pt (CIFAR-10
    inputs the un-faulted model already classifies correctly), wrapped as a
    small DataLoader for fast tests."""
    from torch.utils.data import DataLoader, TensorDataset

    data = torch.load(os.path.join(REPO_ROOT, "correct_data.pt"), weights_only=False)
    n = 20
    dataset = TensorDataset(data["inputs"][:n], data["targets"][:n])
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
