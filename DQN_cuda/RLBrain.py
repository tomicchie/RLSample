# -*- coding: utf-8 -*-
r'''
強化学習のサンプルプログラム
'''
import random
from collections import namedtuple
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import RLUtil
import ReplayMemory


class RLBrain:
    '''
    学習用のクラス。DQNで実装した
    '''

    def __init__(self, num_states, num_actions):
        self.util = RLUtil.RLUtil()
        # Cartpoleの行動（左右どちらかに押す）の2を取得する
        self.num_actions = num_actions
        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory.ReplayMemory(self.util.getCAPACITY())

        # デバイス情報の設定
        d = self.util.getDEVICE()
        self.device = torch.device(d)

        # NNの構築
        self.model = nn.Sequential()
        if self.device.type == "cuda":
            self.model.add_module('fc1', nn.Linear(num_states, 32).cuda())
            self.model.add_module('relu1', nn.ReLU().cuda())
            self.model.add_module('fc2', nn.Linear(32, 32).cuda())
            self.model.add_module('relu2', nn.ReLU().cuda())
            self.model.add_module('fc3', nn.Linear(32, num_actions).cuda())
        else:
            self.model.add_module('fc1', nn.Linear(num_states, 32))
            self.model.add_module('relu1', nn.ReLU())
            self.model.add_module('fc2', nn.Linear(32, 32))
            self.model.add_module('relu2', nn.ReLU())
            self.model.add_module('fc3', nn.Linear(32, num_actions))

        # NWの形を出力
        print(self.model)

        # 最適化手法の設定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        '''
        Experience Replay でネットワークの結合パラメタを学習する
        '''
        batch_size = self.util.getBATCH_SIZE()
        gamma = self.util.getGAMMA()


        # ---------------------------------------------------------
        # 1.メモリサイズの確認
        # ---------------------------------------------------------

        # 1.1. メモリサイズがミニバッチより小さい間は何もしない
        if len(self.memory) < batch_size:
            return

        # ---------------------------------------------------------
        # 2.ミニバッチの作成
        # ---------------------------------------------------------

        # 2.1. メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(batch_size)

        # 2.2. 各変数をミニバッチに対応する形に変形
        # transitionsは1stepごとのstate, action, state_next, rewardが、BATCH_SIZE分格納されている。
        # つまり、(state, action, state_next, reward) * BATCH_SIZE
        # をミニバッチにしたい。つまり、transitionの各メンバに対してBATCH_SIZEをかけることをする。
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        batch = Transition(*zip(*transitions))

        # 2.3. 各変数の要素をミニバッチに対応する形に変形する
        # 例えばstateの場合、[torch.FloatTensor of size 1*4]がBATCH_SIZE分並んでいるが、
        # それをtorch.FloatTensor of sieze BATCH_SIZE * 4に変換する。
        # cat = 結合
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # ---------------------------------------------------------
        # 3. 教師信号となるQ(s_t, a_t)値を求める
        # ---------------------------------------------------------

        # 3.1. ネットワークを推論モードに切り替える
        self.model.eval()

        # 3.2. ネットワークが出力したQ（s_t, a_t）を求める
        # self.model(state_batch)は、左右の両方のQ値を出力しており
        # [torch.FloatTensor of size BATCH_SIZE * 2]になっている
        # ここから実行したアクションa_tに対応するQ値を求めるため、action_batchで行った行動a_tが右か左かのindexを求め、
        # それに対応するQ値をgatherで引っ張り出す。
        state_action_values = torch.tensor(self.model(state_batch), device=self.device).gather(1, action_batch)

        # 3.3. max{Q(s_t + 1, a)}値を求める。ただし次の状態があるかどうかに注意する

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
        no_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8, device=self.device)
        # まずは全部0にしておく
        next_state_values = torch.zeros(batch_size, dtype=torch.float, device=self.device)

        # 次の状態があるindexの最大Q値を求める
        # 出力にアクセスし、max(1)で列方向の最大値の[値, index]を求める
        # そして、そのQ値（index = 0）を出力する
        # detachでその値を取り出す
        next_state_values[no_final_mask] = torch.tensor(self.model(non_final_next_states), device=self.device).max(1)[0].detach()

        # 3.4. 教師となるQ（s_t, a_t）値を、Q学習の式から求める
        expected_state_action_values = reward_batch + gamma * next_state_values

        # ---------------------------------------------------------
        # 4. 結合パラメタの更新
        # ---------------------------------------------------------

        # 4.1. ネットワークを訓練モードに切り替える
        self.model.train()

        # 4.2. 損失関数を計算する（smooth_l1_lossはHuberloss）
        # expected_state_action_valuesはsizeが[minbatch]になっているので、unsqueezeで[minbatch * 1]へ
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # 4.3. 結合パラメタを更新する
        # 勾配をリセット
        self.optimizer.zero_grad()
        # バックプロパゲーションを計算
        loss.backward()
        # 結合パラメタを更新
        self.optimizer.step()

    def decide_action(self, state, episode):
        '''
        現在の状態に応じて、行動を決定する
        '''

        # ε-greedy法で徐々に最適行動のみを採用する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            # ネットワークを推論モードに切り替える
            self.model.eval()
            with torch.no_grad():
                action = torch.tensor(self.model(state).max(1)[1].view(1, 1), dtype=torch.long, device=self.device)
            # このとき、max(1)[1]でネットワークの出力の最大値のindexを取り出す
            # view(1, 1)は[torch.LongTensor of size 1]をsize 1*1に変換する

        else:
            # 0-1の行動をランダムに返す
            action = torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long, device=self.device)
            # このとき、actionは[torch.LongTensor of size 1*1]の形になる

        return action