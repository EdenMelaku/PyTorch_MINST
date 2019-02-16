import torch
import torchvision
import matplotlib.pyplot as plt




#this are the hyper parametrs that we are going to use to train owr model
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# downloading minst dataset
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
        plt.xticks([])
        plt.yticks([])
    fig


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(downloadDataset.train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(downloadDataset.train_loader.dataset),
        100. * batch_idx / len(downloadDataset.train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(downloadDataset.train_loader.dataset)))
      torch.save(network.state_dict(), '/home/eden/newMODEL/results/model.pth')
      torch.save(optimizer.state_dict(), '/home/eden/newMODEL/results/optimizer.pth')


def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in downloadDataset.test_loader:
      print("in loop")
      output = network(data)
      print("output loaded")
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(downloadDataset.test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(downloadDataset.test_loader.dataset),
    100. * correct / len(downloadDataset.test_loader.dataset)))

if __name__ == "__main__":
    print("programm started running ")
    downloadDataset()

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim


    class Net(nn.Module):
        print("defining neural netwok")
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)


    network = Net()
    print("defining optimizer i use SGD ")
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(downloadDataset.train_loader.dataset) for i in range(n_epochs + 1)]

print("runnning pre training test")
test()
print("training model ")
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

print("fininshed training now running examples")
with torch.no_grad():
  examples = enumerate(downloadDataset.test_loader)
  batch_idx, (example_data, example_targets) = next(examples)
  output = network(example_data)
  fig=plt.figure()
  for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], camp='gray', interpolation='none')
      plt.title("prediction:{}".format(
          output.data.max(1, keepdim=True)[1][i].item()
      ))
      plt.xticks([])
      plt.yticks([])
  fig

