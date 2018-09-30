# -*- coding: utf-8 -*-
'''
エージェントが持つ脳となるクラス。PrioritizedExperienceReplayを実行する
'''

import torch
from torch import optim
import torch.nn as nn

import RLUtil

class RLBrain:
    '''
    学習用のクラス。A2Cで実装した
    '''

    def __init__(self, actor_critic):
        self.util = RLUtil.RLUtil()
        # actor_criticがクラスNetのDNNとなる
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)

        # デバイスの設定
        d = self.util.getDEVICE()
        self.device = torch.device(d)

    def update(self, rollouts):
        '''
        Advantageで計算した5つのstepのすべてを使って更新する
        '''
        # torch.Size([4,84,84])
        # obs_shape = rollouts.observations.size()[2:]
        num_steps = self.util.getNUM_ADVANCED_STEP()
        num_process = self.util.getNUM_PROCESSES()

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(rollouts.observations[:-1].view(-1, 4), rollouts.actions.view(-1, 1))

        # 注意：各変数のサイズ
        # rollouts.observations[:-1].view(-1, 4) torch.Size([80,4])
        # rollouts.actions.view(-1, 1) torch.Size([80,1])
        # values torch.Size([80,1])
        # action_log_probs torch.Size([80,1])
        # entropy torch.Size([])

        # torch.Size([5,16,1])
        values = values.view(num_steps, num_process, 1)

        action_log_probs = action_log_probs.view(num_steps, num_process, 1)

        # Advantage（行動価値-状態価値）の計算
        # torch.Size([5,16,1])
        advantages = rollouts.returns[:-1] - values

        # Criticのlossを計算
        value_loss = advantages.pow(2).mean()

        # Actorのgainを計算、あとの数式でマイナスをかけてlossにする
        # advantagesをdetach()して定数として扱う
        action_gain = (action_log_probs * advantages.detach()).mean()

        # 誤差関数の総和を計算
        total_loss = (value_loss * self.util.getVALUE_LOSS_COEF() - action_gain - entropy * self.util.getENTROPY_COEF())

        # -------------------
        # 結合パラメタを更新
        # -------------------

        # 訓練モードに変更
        self.actor_critic.train()
        # 勾配をリセット
        self.optimizer.zero_grad()
        # バックプロパゲーションを計算
        total_loss.backward()
        # 一気に結合パラメタが変化しすぎないように、勾配の大きさはMAX_GRAD_NORMまでに抑える
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.util.getMAX_GRAD_NORM())
        # 結合パラメタを更新
        self.optimizer.step()



