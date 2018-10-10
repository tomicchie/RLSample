# -*- coding: utf-8 -*-
'''
強化学習のサンプルプログラム
'''
import torch
import RLUtil

class RolloutStorage(object):
    '''
    メモリクラスの定義
    '''

    def __init__(self, num_steps, num_processes, obs_shape):
        self.util = RLUtil.RLUtil()

        # デバイス情報の設定
        d = self.util.getDEVICE()
        self.device = torch.device(d)

        self.observations = torch.zeros(num_steps + 1, num_processes, 4, device=self.device)
        self.masks = torch.ones(num_steps + 1, num_processes, 1, device=self.device)
        self.rewards = torch.zeros(num_steps, num_processes, 1, device=self.device)

        if self.device.type == "cuda":
            self.actions = torch.zeros(num_steps, num_processes, 1).type(torch.cuda.LongTensor)
        else:
            self.actions = torch.zeros(num_steps, num_processes, 1).type(torch.LongTensor)

        # 割引報酬和を格納
        self.returns = torch.zeros(num_steps + 1, num_processes, 1, device=self.device)
        # insertするindex値
        self.index = 0
    
    def insert(self, current_obs, action, reward, mask):
        '''
        次のindexにtransitionを格納する
        '''
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        # indexの更新
        self.index = (self.index + 1) % self.util.getNUM_ADVANCED_STEP()
    
    def after_update(self):
        '''
        Advantageするstep数が完了したら、最新のものをindex0に格納
        '''
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])
    
    def compute_returns(self, next_value):
        '''
        Advantageするステップ中の各ステップの割引報酬和を計算する
        '''

        # 注意：5step目（NUM_ADVANCED_STEP）から逆向きに計算している
        # 注意：5step目はAdvantage1となる、4step目はAdvantage2となるため、
        # 一般化するとMstep中のNstep目はAdvantage(M-N+1)となる
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * self.util.getGAMMA() * self.masks[ad_step + 1] + self.rewards[ad_step]
    
