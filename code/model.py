import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    # Input size is 120

    def __init__(self, name, classes, batch_size, epochs, channels):
        super(Net, self).__init__()
        self.model_name = f'dct_trained_{name}.h5'
        self.classes = classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.conv1 = nn.Conv2d(channels, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, self.classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.sigmoid(self.fc1(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

