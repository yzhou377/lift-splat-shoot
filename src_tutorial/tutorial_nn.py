import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

class Net(nn.Module):
    
    def __init__(self):
        # Refering to the parent class, initialize that as well
        # In python3, could also write super().__init__()
        super(Net, self).__init__()

        # The convolutional kernal
        self.conv1= nn.Conv2d(1,6,3)
        self.conv2= nn.Conv2d(6,16,3)

        # The affine layers: y=Wx+b
        # At this stage, the image size is 6x6
        #(with maxPooling in mind for forward pass)
        self.fc1= nn.Linear(16*6*6, 120)
        self.fc2= nn.Linear(120,84)
        self.fc3= nn.Linear(84,10) # ended up with 10 classes

    def forward(self,x):
        # Max pooling over a (2,2) window
        x= F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x= F.max_pool2d(F.relu(self.conv2(x)),2)
        x= x.view(-1,self.num_flat_features(x)) #vectorize
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x= self.fc3(x)
        return x
    
    def num_flat_features(self,x):
        size= x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

net=Net()
print(net)

input = torch.randn(2, 1, 32, 32)
# First dimension is the batch size
# Second dimension is the channel number
# Third and forth dimension are the image size 

out = net(input)
print(out)