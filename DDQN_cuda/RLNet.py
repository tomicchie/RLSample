import torch
import torch.nn as nn
import torch.nn.functional as F

import RLUtil

class Net(nn.Module):
    '''
        Create Deep-Neural-Network
    '''
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()

        self.util = RLUtil.RLUtil()

        # デバイスの設定
        d = self.util.getDEVICE()
        self.device = torch.device(d)

        if self.device.type == "cuda":
            self.fc1 = nn.Linear(n_in, n_mid).cuda()
            self.fc2 = nn.Linear(n_mid, n_mid).cuda()
            self.fc3 = nn.Linear(n_mid, n_out).cuda()
        else:
            self.fc1 = nn.Linear(n_in, n_mid)
            self.fc2 = nn.Linear(n_mid, n_mid)
            self.fc3 = nn.Linear(n_mid, n_out)
    
    def forward(self, x):
        if self.device.type == "cuda":
            h1 = F.relu(self.fc1(x)).cuda()
            h2 = F.relu(self.fc2(h1)).cuda()
        else:
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output
