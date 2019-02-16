
# downloading minst dataset
import torch
import torchvision
import matplotlib as plt
from projPackage.test import batch_size_train, batch_size_test


def downloadDataset():
    print("downloading minst dataset")
    downloadDataset.train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_train, shuffle=True)

    downloadDataset.test_loader= torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True)
    print("finished download ")
    print("running example from the downloaded dataset")
    # testing examples
    examples = enumerate(downloadDataset.test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)
    print("ploting figure ")
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        print(example_data[i][0])
        plt.xticks([])
        plt.yticks([])
    fig


if __name__ == "__main__":
    downloadDataset()
