import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import cat


__all__ = ['OneDCNN', 'cnn']


##----------------------------------Model Published in-------------------------#
# Chowdhury, Anurag, and Arun Ross. "Fusing MFCC and LPC Features using 1D Triplet CNN 
# for Speaker Recognition in Severely Degraded Audio Signals."
# IEEE Transactions on Information Forensics and Security (2019).
##-----------------------------------------------------------------------------#

class OneD_DCNN_triplet(nn.Module):
    ##  Try SELU activation
    def __init__(self, num_classes=168):
        super(OneD_DCNN_triplet, self).__init__()

        self.features = nn.Sequential(
            nn.ZeroPad2d((0, 0, 2, 2)),
            nn.Conv2d(2, 16, kernel_size=(3,1), stride=1, padding=0 , dilation = (2,1)),
            nn.SELU(),

            nn.Conv2d(16, 32, kernel_size=(3,1), stride=1, padding=0, dilation = (2,1)),
            nn.SELU(),

            nn.Conv2d(32, 64, kernel_size=(7,1), padding=0, dilation = (2,1)),
            nn.SELU(),

            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(64, 128, kernel_size=(9,1), stride=1, dilation = (3,1)),
            nn.SELU()
        )


        self.regularization = nn.Sequential(
            nn.AlphaDropout(p=0.25)
        )

        self.fc = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1,1,1), padding=0)
        )



    def evaluate(self, x):
        o = self.features(x)
        pool_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1,o.size()[3]))
        )
        o = self.regularization(o)
        o = pool_layer(o)
        o = o.view(o.size()[0], -1)
        return o

    def forward(self, x1):
        x1 = self.evaluate(x1)
        return x1

def cnn(pretrained=False, **kwargs):
    """OneDCNN model architecture (with dilations) for speaker identification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OneD_DCNN_triplet(**kwargs)
    return model
