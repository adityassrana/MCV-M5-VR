import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils import *

results_dir = '../results/deep_baseline_'
log_dir = '../log/'
model_dir = '../models/'

train_data_dir = '/home/mcv/datasets/MIT_split/train'
test_data_dir = '/home/mcv/datasets/MIT_split/test'

img_size= 64
batch_size = 16
number_of_epoch = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- LOADING DATASET--------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           num_workers=4, shuffle=True)

test_set = torchvision.datasets.ImageFolder(test_data_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          num_workers=4, shuffle=False)

#---- DEFINING MODEL--------
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3)
        self.conv2 = nn.Conv2d(32,32,3)
        self.conv3 = nn.Conv2d(32,64,3)
        self.conv4 = nn.Conv2d(64,64,3)
        self.activ = nn.ReLU()
        self.pool = nn.MaxPool2d((2,2), stride=(2,2))
        self.avgpool = nn.AvgPool2d(4)
        self.fc1 = nn.Linear(64, 8)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm2d(num_features=64, eps=1e-03, momentum=0.99)

    def forward(self,x):
        x = self.pool(self.activ(self.conv1(x)))
        x = self.pool(self.activ(self.conv2(x)))
        x = self.pool(self.activ(self.conv3(x)))
        x = self.dropout(x)
        x = self.activ(self.conv4(x))
        x = self.batchnorm(x)
        x = self.avgpool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.softmax(self.fc1(x))
        x = torch.squeeze(x)
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
                                                                
model = Baseline()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
             
model.apply(weights_init)    
model.to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters for this model: {}'.format(total_params))

summary(model, (3, img_size, img_size))

images, _ = next(iter(train_loader))
writer = SummaryWriter(f'tb/deep_baseline/')
writer.add_graph(model, images.to(device))
writer.close()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)

writer_train = SummaryWriter(f'tb/deep_baseline/train')
writer_test = SummaryWriter(f'tb/deep_baseline/test')

train_acc_hist = []
test_acc_hist = []
train_loss_hist = []
test_loss_hist = []

for epoch in range(number_of_epoch):
    model.train()

    # training statistics
    losses, acc, count = [], [], []
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # transfer data to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculating this way to account for the fact that the
        # last batch may have different batch size
        batch_size = inputs.shape[0]
        # get number of right predictions
        correct_predictions = (outputs.argmax(dim=1) == labels).float().sum()
        # add to list
        losses.append(batch_size * loss.item()), count.append(batch_size), acc.append(correct_predictions)

        writer_train.add_scalar('per_batch/train_loss', loss.item(), epoch * len(train_loader) + batch_idx)

    # accumulate/average statistics
    n = sum(count)
    train_loss_epoch = sum(losses) / n
    train_acc_epoch = sum(acc) / n

    train_loss_hist.append(train_loss_epoch)
    train_acc_hist.append(train_acc_epoch)

    writer_train.add_scalar('per_epoch/losses', train_loss_epoch, epoch)
    writer_train.add_scalar('per_epoch/accuracy', train_acc_epoch, epoch)

    # validation
    model.eval()

    with torch.no_grad():
        losses, acc, count = [], [], []
        for batch_idx, (inputs, labels) in enumerate((test_loader)):
            # transfer data to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_size = inputs.shape[0]
            # get number of right predictions
            correct_predictions = (outputs.argmax(dim=1) == labels).float().sum()
            # add to list
            losses.append(batch_size * loss.item()), count.append(batch_size), acc.append(correct_predictions)

            writer_test.add_scalar('per_batch/test_loss', loss.item(), epoch * len(test_loader) + batch_idx)

    # accumulate/average statistics
    n = sum(count)
    test_loss_epoch = sum(losses) / n
    test_acc_epoch = sum(acc) / n

    test_loss_hist.append(test_loss_epoch)
    test_acc_hist.append(test_acc_epoch)

    writer_test.add_scalar('per_epoch/losses', test_loss_epoch, epoch)
    writer_test.add_scalar('per_epoch/accuracy', test_acc_epoch, epoch)

    print(f"Epoch{epoch}, train_accuracy:{train_acc_epoch:.4f}, test_accuracy:{test_acc_epoch:.4f}, train_loss:{train_loss_epoch:.4f}, test_loss:{test_loss_epoch:.4f}")


# Save model
torch.save(model.state_dict(),model_dir+'/deep_baseline_weights.pkl')

plot_accuracy(train_acc_hist, test_acc_hist, results_dir, baseline=0.78,xmax=number_of_epoch)
plot_loss(train_loss_hist, test_loss_hist, results_dir, baseline=1.2,xmax=number_of_epoch)
