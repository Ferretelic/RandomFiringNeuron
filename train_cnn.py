from rfneuron import RandomFiringNeuron
import torch
import torchvision

class NormalConv(torch.nn.Module):
  def __init__(self):
    super(NormalConv, self).__init__()
    self.conv1 = torch.nn.Conv2d(3, 6, 5)
    self.pool = torch.nn.MaxPool2d(2, 2)
    self.conv2 = torch.nn.Conv2d(6, 16, 5)
    self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
    self.fc2 = torch.nn.Linear(120, 84)
    self.fc3 = torch.nn.Linear(84, 10)
    self.activation = RandomFiringNeuron()

  def forward(self, x):
    x = self.pool(self.activation(self.conv1(x)))
    x = self.pool(self.activation(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.fc3(x)
    return x


def main():
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
  testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

  classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
  net = NormalConv()
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if i % 2000 == 1999:
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

  print("Finished Training")

if __name__ == "__main__":
    main()