import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as ini 

from utils_pytorch import *
#from torch.utils.tensorboard import SummaryWriter

results_dir = '../results/baseline_'
log_dir = '../log/'
weights_dir = '../weights/'
model_dir = '../models/'

train_data_dir = '/home/mcv/datasets/MIT_split/train'
val_data_dir = '/home/mcv/datasets/MIT_split/test'
test_data_dir = '/home/mcv/datasets/MIT_split/test'

img_size= 32
batch_size = 16
number_of_epoch = 100
validation_samples = 807

#---- LOADING DATASET--------
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.Resize((img_size,img_size))])#

train_set = torchvision.datasets.ImageFolder(train_data_dir,transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, num_workers=4,shuffle=True)

test_set = torchvision.datasets.ImageFolder(test_data_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=4,shuffle=False)

#---- DEFINING MODEL--------
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv = nn.Conv2d(3,64,3)
        self.conv2 = nn.Conv2d(64,32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*6*6, 8)
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool(F.relu(self.conv2(x)))
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
        ini.xavier_normal_(m.weight.data)  
             
model.apply(weights_init)    
    
print(model)


loss_function= nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)


model.cuda()
#writer = SummaryWriter()


train_acc_hist = []
test_acc_hist = []
train_loss_hist = []
test_loss_hist = []
best_test_accuracy = 0.0

for epoch in range(number_of_epoch+1) :
#----TRAIN-------------- 
    print('-'*10)
    print('Epoch {}/{}'.format(epoch, number_of_epoch))
    print('-'*10)

    running_loss = 0
    true_positives = 0
    
    model.train()

    for x, (train_samples, train_labels) in enumerate(train_loader,0):            

        train_inputs = train_samples.cuda()
        labels = train_labels.cuda()
        
        #zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True) : 
            train_outputs = model(train_inputs)
            
            loss = loss_function(train_outputs, labels)
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * train_inputs.size(0) 
        values, indices = torch.max(train_outputs, 1)                
        true_positive = (indices == labels).float().sum()
        true_positives += true_positive            
        
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = true_positives/len(train_loader.dataset)
    print('Train - Loss : {:.4f} Accuracy: {}'.format(train_loss,train_acc)) 
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)   
    
#----TEST-------------- 

    running_loss_test = 0
    true_positives_test = 0
    
    model.eval()
    
    for x, (test_samples, test_labels) in enumerate(test_loader, 0):
    
        test_inputs = test_samples.cuda()
        labels = test_labels.cuda()
    
        with torch.set_grad_enabled(False):
    
            test_outputs = model(test_inputs)
    
            loss_test = loss_function(test_outputs, labels)
        
        running_loss_test += loss_test.item() * test_inputs.size(0)
    
        values, indices = torch.max(test_outputs, 1)
        true_positive_test = (indices == labels).float().sum()
        true_positives_test += true_positive_test
        
    test_loss = running_loss_test / len(test_loader.dataset)
    
    test_acc = true_positives_test/len(test_loader.dataset)
    print('Val   - Loss : {:.4f} Accuracy: {}'.format(test_loss,test_acc))
    test_loss_hist.append(test_loss)
    test_acc_hist.append(test_acc)
    
    if test_acc > best_test_accuracy:
        best_test_accuracy = test_acc   

# Save model
torch.save(model.state_dict(),model_dir+'/baseline_ian_weights.pkl')

# Plot results
print('Best test accuracy: {:.3f}'.format(best_test_accuracy))
plot_accuracy(train_acc_hist, test_acc_hist, results_dir,baseline=0.78,xmax=number_of_epoch)
plot_loss(train_loss_hist, test_loss_hist, results_dir,baseline=1.2,xmax=number_of_epoch)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters for this model: {}'.format(total_params))
