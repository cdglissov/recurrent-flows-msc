from sklearn.datasets import make_moons
import torch.utils.data as Data

#Into dataset file
def sample_half_moons(n_train, n_test, noise=0.1):
    train_data, train_labels = make_moons(n_samples=n_train, noise=noise)
    test_data, test_labels = make_moons(n_samples=n_test, noise=noise)
    return train_data, train_labels, test_data, test_labels

# TODO: Implement transform
def loader_half_moons(train_data, test_data, batch_size):
    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
