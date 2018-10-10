# -*- coding: utf-8 -*-
'''
強化学習のサンプルプログラム
'''

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
            # Actor-Critic
            # 行動を決めるのではなく出力は行動の種類数
            self.actor = nn.Linear(n_mid, n_out).cuda()
            # 状態価値なので出力は1つ
            self.critic = nn.Linear(n_mid, 1).cuda()
        else:
            self.fc1 = nn.Linear(n_in, n_mid)
            self.fc2 = nn.Linear(n_mid, n_mid)
            # Actor-Critic
            # 行動を決めるのではなく出力は行動の種類数
            self.actor = nn.Linear(n_mid, n_out)
            # 状態価値なので出力は1つ
            self.critic = nn.Linear(n_mid, 1)

    def forward(self, x):
        '''
        ネットワークのフォワード計算を定義
        '''
        
        if self.device.type == "cuda":
            h1 = F.relu(self.fc1(x)).cuda()
            h2 = F.relu(self.fc2(h1)).cuda()
        else:
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(h1))


        # 状態価値の計算
        critic_output = self.critic(h2)
        actor_output = self.actor(h2)

        return critic_output, actor_output

    def act(self, x):
        '''
        状態xから行動を確率的に求める
        '''
        _, actor_output = self(x)
        # dim=1で行動の種類方向にsoftmaxを計算
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)
        # dim=1で行動の種類報告に圧縮機確率
        # dim=1で構造の種類方法に確立計算
        return action

    def get_Value(self, x):
        '''
        状態xから状態価値を求める
        '''
        value, _ = self(x)
        return value

    def evaluate_actions(self, x, actions):
        '''
        状態xから状態価値、実際の行動actionsのlog確率とエントロピーを求める
        '''
        value, actor_output = self(x)

        # dim=1で行動の種類方向に計算
        log_probs = F.log_softmax(actor_output, dim=1)

        # 実際の行動のlog_probsを求める
        action_log_probs = log_probs.gather(1, actions)

        # dim = 1で行動の種類方向に計算
        probs = F.softmax(actor_output, dim=1)
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy
