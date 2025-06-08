import torch
import torch.nn as nn   
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToTensor



def get_default_device():
    """Pick GPU if available, else CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

#2. to_device --> For moving tensors to the chosen device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        batch = to_device(batch, device)
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        batch = to_device(batch, device)
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc = accuracy(out, labels) # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1],result['train_loss'],result['val_loss'],result['val_acc']
        ))



# Defining the Helper function for our ResNet9 Model
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(3)) # 3 Pooling layers
    return nn.Sequential(*layers)


# Defining the ResNet9 Model through Custom class
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # 3 x 130 x 130 
        self.conv1 = conv_block(in_channels=3, out_channels=64)  # 64 x 130 x 130
        self.conv2 = conv_block(in_channels=64, out_channels=128, pool=True) # 128 x 43 x 43   
        self.res1 = nn.Sequential(conv_block(in_channels=128, out_channels=128),    # 1st Residual Block
                                  conv_block(in_channels=128, out_channels=128)) # 128 x 65 x 65  
        
        self.conv3 = conv_block(in_channels=128, out_channels=256, pool=True) # 256 x 14 x 14 
        self.conv4 = conv_block(in_channels=256, out_channels=512, pool=True) # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(in_channels=512, out_channels=512),  # 2nd Residual Block
                                  conv_block(in_channels=512, out_channels=512)) # 512 x 4 x 4
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),  # 512 x 4 x 4 --> # 512 x 1 x 1
                                        nn.Flatten(),      # 512
                                        nn.Dropout(0.2), # p=0.2 About 20% of the random neurons will be set to zero
                                        nn.Linear(512, num_classes)) # 6
  
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out 