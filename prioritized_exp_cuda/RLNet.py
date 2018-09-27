import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    '''
        Create Deep-Neural-Network
    '''
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        # Dueling Network
        # fc3_adv = Advantage, fc3_v = Value
        self.fc3_adv = nn.Linear(n_mid, n_out)
        self.fc3_v = nn.Linear(n_mid, 1)
    
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        # この出力はReLUしない
        adv = self.fc3_adv(h2)

        # この出力はReLUしない
        # valはadvと足し算するために、サイズを[minibatch*1]から[minibatch*2]にexpandする
        # adv.size(1)は出力する行動の種類数の2
        val = self.fc3_v(h2).expand(-1, adv.size(1))

        # val*advからadvの平均値を引き算する
        # adv.mean(1, keepdim=True)で列方向（行動の種類方向）に平均し、サイズが[minibatch*1]
        # expandで展開して、サイズ[minibatch*2]
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return output
