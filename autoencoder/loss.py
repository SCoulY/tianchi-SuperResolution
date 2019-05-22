import torch.nn as nn

# modified at 14/01/2019, pytorch released cross_entropy_func
class cross_entropy_loss_2d(nn.Module):
    def __init__(self, weight=None):
        super(cross_entropy_loss_2d, self).__init__()        
        self.loss = nn.CrossEntropyLoss(weight, reduction='mean')

    def forward(self, inputs, targets):
        return self.loss(inputs, targets) 