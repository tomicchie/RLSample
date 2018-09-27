# -*- coding: utf-8 -*-
'''
エージェントが持つ脳となるクラス。PrioritizedExperienceReplayを実行する
'''
import random
from collections import namedtuple
import numpy as np

import torch
from torch import optim
import torch.nn.functional as F

import RLUtil
import ReplayMemory
import RLNet
import TDErrorMemory


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

        # NNの構築
        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = RLNet.Net(n_in, n_mid, n_out)
        self.target_q_network = RLNet.Net(n_in, n_mid, n_out)
        # NNの形を出力
        print(self.main_q_network)
        # 最適化手法の設定
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

        # TD誤差のメモリオブジェクトを生成
        self.td_error_memory = TDErrorMemory.TDErrorMemory(self.util.getCAPACITY())

        # デバイスの設定
        d = self.util.getDEVICE()
        self.device = torch.device(d)

    def replay(self,episode):
        '''
        Experience Replay でネットワークの結合パラメタを学習する
        '''
        batch_size = self.util.getBATCH_SIZE()


        # ---------------------------------------------------------
        # 1.メモリサイズの確認
        # ---------------------------------------------------------

        # 1.1. メモリサイズがミニバッチより小さい間は何もしない
        if len(self.memory) < batch_size:
            return

        # ---------------------------------------------------------
        # 2.ミニバッチの作成
        # ---------------------------------------------------------

        # 処理を別メソッドに分離
        batch, state_batch, action_batch, reward_batch, non_final_next_states = self.make_minivatch(episode, batch_size)

        # ---------------------------------------------------------
        # 3. 教師信号となるQ(s_t, a_t)値を求める
        # ---------------------------------------------------------

        # 処理を別メソッドに分離
        expected_state_action_value, state_action_values = self.get_expected_state_action_values(batch_size, batch, state_batch, action_batch, reward_batch, non_final_next_states)

        # ---------------------------------------------------------
        # 4. 結合パラメタの更新
        # ---------------------------------------------------------

        # 処理を別メソッドに分離
        self.update_main_q_network(expected_state_action_value, state_action_values)

    def decide_action(self, state, episode):
        '''
        現在の状態に応じて、行動を決定する
        '''

        # ε-greedy法で徐々に最適行動のみを採用する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            # ネットワークを推論モードに切り替える
            self.main_q_network.eval()
            with torch.no_grad():
                action = torch.tensor(self.main_q_network(state).max(1)[1].view(1, 1), dtype=torch.long, device=self.device)
            # このとき、max(1)[1]でネットワークの出力の最大値のindexを取り出す
            # view(1, 1)は[torch.LongTensor of size 1]をsize 1*1に変換する

        else:
            # 0-1の行動をランダムに返す
            action = torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long, device=self.device)
            # このとき、actionは[torch.LongTensor of size 1*1]の形になる

        return action

    def make_minivatch(self, episode, batch_size):
        '''
            2.ミニバッチの作成
        '''
        # 2.1. メモリからミニバッチ分のデータを取り出す
        if episode < 30:
            transitions = self.memory.sample(batch_size)
        else:
            # TD誤差に応じてミニバッチを取り出すに変更
            indexes = self.td_error_memory.get_prioritized_indexes(batch_size)
            transitions = [self.memory.memory[n] for n in indexes]

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

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self, batch_size, batch, state_batch, action_batch, reward_batch, non_final_next_states):
        '''
            3. 教師信号となるQ(s_t, a_t)値を求める
        '''
        gamma = self.util.getGAMMA()
        
        # 3.1. ネットワークを推論モードに切り替える
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2. ネットワークが出力したQ（s_t, a_t）を求める
        # self.main_q_network(state_batch)は、左右の両方のQ値を出力しており
        # [torch.FloatTensor of size BATCH_SIZE * 2]になっている
        # ここから実行したアクションa_tに対応するQ値を求めるため、action_batchで行った行動a_tが右か左かのindexを求め、
        # それに対応するQ値をgatherで引っ張り出す。
        state_action_values = torch.tensor(self.main_q_network(state_batch), device=self.device).gather(1, action_batch)

        # 3.3. max{Q(s_t + 1, a)}値を求める。ただし次の状態があるかどうかに注意する

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8, device=self.device)
        # まずは全部0にしておく
        next_state_values = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        a_m = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # 次の状態での最大Q値の行動a_mをMain Q-Networkから求める
        # 最後の[1]で行動に対応したindexが返る
        a_m[non_final_mask] = self.main_q_network(non_final_next_states).detach().max(1)[1]

        # 次の状態があるものだけにフィルタし、size 32 を32 * 1へ
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 次の状態があるindexの行動a_mのQ値をtarget Q-Networkから求める
        # detach()で取り出す。
        # squeeze()でsize[minibatch * 1]を[minibatch]にする。
        next_state_values[non_final_mask] = torch.tensor(self.target_q_network(non_final_next_states), device=self.device).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4. 教師となるQ（s_t, a_t）値を、Q学習の式から求める
        expected_state_action_values = reward_batch + gamma * next_state_values

        return expected_state_action_values, state_action_values

    def update_main_q_network(self, expected_state_action_values, state_action_values):
        '''
            4. 結合パラメタの更新
        '''
         # 4.1. ネットワークを訓練モードに切り替える
        self.main_q_network.train()

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
    
    def update_target_q_network(self):
        '''
            Target Q-NetworkをMainと同じにする
        '''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())
    
    def update_td_error_memory(self):
        '''
        TD誤差メモリに格納されているTD誤差を更新する
        '''
        gamma = self.util.getGAMMA()

        # ネットワークを推論モードに切り替える
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 全メモリでミニバッチを作成
        transitions = self.memory.memory
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # ネットワークが出力したQ(s_t, a_t)を求める
        state_action_values = self.main_q_network(state_batch).gather(1, action_batch)

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8, device=self.device)

        # まずは全部0にしておく。サイズはメモリの長さ
        next_state_values = torch.zeros(len(self.memory))
        a_m = torch.zeros(len(self.memory), dtype=torch.long, device=self.device)

        # 次の状態での最大Q値の行動a_mをMain Q-Networkから求める
        # 最後の[1]で行動に対応したindexが返る
        a_m[non_final_mask] = self.main_q_network(non_final_next_states).detach().max(1)[1]

        # 次の状態があるものだけにフィルタし、size 32を32*1へ
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 次の状態があるindexの、行動a_mのQ値をtarget Q-Networkから求める
        # detachで取り出す
        # squeezeでsize[minibatch*1]を[minibatch]に変換。
        next_state_values[non_final_mask] = self.target_q_network(non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # TD誤差を求める
        # state_action_valuesはsize[minibatch*1]なので、squeezeしてsize[minibatch]へ
        td_errors = (reward_batch + gamma * next_state_values) - state_action_values.squeeze()

        # TD誤差メモリを更新、Tensorをdetachで取り出し、NumPyにしてから、Pythonのリストまで変換
        self.td_error_memory.memory = td_errors.detach().numpy().tolist()