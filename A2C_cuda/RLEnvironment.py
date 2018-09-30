# -*- coding: utf-8 -*-
'''
強化学習のサンプルプログラム
'''
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib import animation
import gym

import copy

#from JSAnimation.IPython_display import display_animation
#from IPython.display import display

import torch

import RLUtil
import RLNet
import RLBrain
import RolloutStorage


class RLEnvironment:
    '''
    cartpoleを実行する環境クラス
    '''

    def __init__(self):
        # Utilクラスインスタンス生成
        self.util = RLUtil.RLUtil()

        # 同時実行する環境数分、envを生成
        self.envs = [gym.make(self.util.getENV()) for i in range(self.util.getNUM_PROCESSES())]

        # 全エージェントが共有して持つBrainを生成
        # 状態は4
        n_in = self.envs[0].observation_space.shape[0]
        # 行動は2
        n_out = self.envs[0].action_space.n
        n_mid = 32
        # DNNの生成
        self.actor_critic = RLNet.Net(n_in, n_mid, n_out)

        self.global_brain = RLBrain.RLBrain(self.actor_critic)
        self.obs_shape = n_in

        # デバイス情報の設定
        d = self.util.getDEVICE()
        self.device = torch.device(d)

    def run(self):
        '''
        実行用のメソッド
        '''
        # 定数情報の取得
        num_processes = self.util.getNUM_PROCESSES()
        num_advanced_step = self.util.getNUM_ADVANCED_STEP()
        num_episodes = self.util.getNUM_EPISODES()
        max_steps = self.util.getMAX_STEPS()

        # 格納用変数の設定
        current_obs = torch.zeros(num_processes, self.obs_shape)
        # RolloutStorageオブジェクトを生成
        rollouts = RolloutStorage.RolloutStorage(num_advanced_step, num_processes, self.obs_shape)
        # 現在の試行の報酬を保持
        episode_rewards = torch.zeros([num_processes, 1])
        # 最後の試行の報酬を保持
        final_rewards = torch.zeros([num_processes, 1])
        # Numpy配列
        obs_np = np.zeros([num_processes, self.obs_shape])
        reward_np = np.zeros([num_processes, 1])
        done_np = np.zeros([num_processes, 1])
        # 各環境のstep数を記録
        each_step = np.zeros([num_processes])
        # 環境の試行数
        episode = 0

        # 初期状態の開始
        obs = [self.envs[i].reset() for i in range(num_processes)]
        obs = np.array(obs)
        # デバイスによりデータ型が変わるので分岐。torch.Size([16,4])
        if self.device == "cuda":
            obs = torch.from_numpy(obs).type(torch.cuda.FloatTensor)
        else:
            obs = torch.from_numpy(obs).type(torch.FloatTensor)
        # 最新のobsを格納
        current_obs = obs
        
        # Advanced学習用のオブジェクトrolloutsの状態の1つめに、現在の状態を保存
        rollouts.observations[0].copy_(current_obs)

        # 実行ループ

        # 全体のループ
        for j in range(num_episodes * num_processes):
            # Advanced学習するstep数ごとに計算
            for step in range(num_advanced_step):
                # 行動を求める
                with torch.no_grad():
                    action = self.actor_critic.act(rollouts.observations[step])
                # Tensorをnumpyに変換
                actions = action.squeeze(1).numpy()

                # 1stepの実行
                for i in range(num_processes):
                    obs_np[i], reward_np[i], done_np[i], _ = self.envs[i].step(actions[i])
                
                    # episodesの終了評価と、state_nextを設定
                    # step数がmax_stepsを超えるか、一定角度以上傾くとtrueになる
                    if done_np[i]:

                        # 環境0のみ出力
                        if i == 0:
                            print("%d Episode: Finished after %d steps" % (episode, each_step[i]+1))
                            episode += 1
                        
                        # 報酬の設定
                        if each_step[i] < max_steps - 5:
                            # 途中でこけたら罰則として報酬-1を与える
                            reward_np[i] = -1.0
                        else:
                            # 立ったまま終了したら報酬1を与える
                            reward_np[i] = 1.0
                        
                        # step数のリセット
                        each_step[i] = 0
                        # 実行環境のリセット
                        obs_np[i] = self.envs[i].reset()

                    else:
                        # 普段は報酬0
                        reward_np[i] = 0.0
                        each_step[i] += 1
                
                # 報酬をTensorに変換し、試行の総報酬に足す
                # デバイスによりデータ型が変わるので分岐。
                if self.device == "cuda":
                    reward = torch.from_numpy(reward_np).type(torch.cuda.FloatTensor)
                else:
                    reward = torch.from_numpy(reward_np).type(torch.FloatTensor)

                episode_rewards += reward

                # 実行環境それぞれについて、doneならmask=0, 継続中ならmask=1にする
                masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done_np], dtype=torch.float, device=self.device)
                # 最終試行の総報酬を更新する
                # 継続中の場合は1を掛け算してそのまま、done時には0をかけてリセット
                final_rewards *= masks
                # 継続中は0を足す。done時にはepisode_rewardsを足す
                final_rewards += (1 - masks) * episode_rewards

                # 試行の総報酬を更新する
                # 継続中のmasksは1なのでそのまま、doneの場合は0に
                episode_rewards *= masks

                # 現在の状態をdone時には全部0にする
                current_obs *= masks

                # current_obsを更新
                # 最新のobsを格納
                if self.device == "cuda":
                    obs = torch.from_numpy(obs_np).type(torch.cuda.FloatTensor)
                else:
                    obs = torch.from_numpy(obs_np).type(torch.FloatTensor)
                
                current_obs = obs
                # メモリオブジェクトに今stepsのtransitionを挿入
                rollouts.insert(current_obs, action.data, reward, masks)
            
            # advancedのfor loop終了

            # advanceした最終stepの状態から予想する状態価値を計算

            with torch.no_grad():
                # rollouts.observationsのサイズはtorch.Size([6,16,4])
                next_value = self.actor_critic.get_Value(rollouts.observations[-1].detach())
            
            # 全stepの割引報酬和を計算して、rolloutsの変数returnsを更新
            rollouts.compute_returns(next_value)

            # ネットワークとrolloutの更新
            self.global_brain.update(rollouts)
            rollouts.after_update()

            # 全部のfinal_rewardsがnum_processesを超えたら成功
            if final_rewards.sum().numpy() >= num_processes:
                print("連続成功")
                break


if __name__ == '__main__':
    rle = RLEnvironment()
    rle.run()
