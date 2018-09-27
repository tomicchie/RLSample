# -*- coding: utf-8 -*-
r'''
強化学習のサンプルプログラム
'''
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib import animation
import gym

from JSAnimation.IPython_display import display_animation

from IPython.display import display

import torch

import RLUtil
import RLAgent

class RLEnvironment:
    '''
    cartpoleを実行する環境クラス
    '''

    def __init__(self):
        # Utilクラスインスタンス生成
        self.util = RLUtil.RLUtil()
        # 実行する課題を設定
        self.env = gym.make(self.util.getENV())

        # 課題の状態と行動の数を設定
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        # このとき、cartpoleの行動（left/right）の2を取得

        # Agentクラスのインスタンスを生成
        self.agent = RLAgent.RLAgent(self.num_states, self.num_actions)

        # デバイス情報の設定
        d = self.util.getDEVICE()
        self.device = torch.device(d)

    def display_frames_as_gif(self, frames):
        """
        Displays a alist of frames as a gif, with controls
        """

        plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

        # 動画のファイル名を指定して保存
        anim.save('movie_cartpole_DQN.mp4')

        # display(display_animation(anim, default_mode='loop'))

    def run(self):
        '''
        実行用のメソッド
        '''
        # 10試行分の立ち続けたstep数を格納し、平均step数を出力に利用
        episode_10_list = np.zeros(10)

        # 195step以上連続で立ち続けた試行数
        complete_episodes = 0

        # 最後の試行フラグ
        episode_final = False

        # 最後の試行を動画にするために画像を格納する変数
        frames = []

        # 試行数分繰り返す
        for episode in range(self.util.getNUM_EPISODES()):

            # 環境の初期化
            observation = self.env.reset()

            # 観測値をそのまま状態sとして使用
            state = observation
            # numpy変数をPyTorchのテンソルに変換
            if self.device == "cuda":
                state = torch.from_numpy(state).type(torch.cuda.FloatTensor)
            else:
                state = torch.from_numpy(state).type(torch.FloatTensor)
            # FloatTensor of size 4 をsize 1*4に変換
            state = torch.unsqueeze(state, 0)
    
            # 最大step数の呼び出し
            max_steps = self.util.getMAX_STEPS()

            # エピソード単位のループ
            for step in range(max_steps):

                # 最終試行の場合はframesに各時刻の画像を追加していく
                #if episode_final is True:
                #    frames.append(self.env.render(mode='rgb_array'))

                # 行動を求める
                action = self.agent.get_action(state, episode)

                # 行動a_tの実行により、s_{t+1}とdoneフラグを求める
                # actionから.item()を指定して中身を取り出す
                observation_next, _, done, _ = self.env.step(action.item())
                # このとき、rewardとinfoは使わないので_にする

                # 報酬を与える。さらにepisodeの終了評価と、state_nextを設定する
                # step数がMAX_STEP分経過するか、一定角度以上傾くとdoneはtrueになる
                if done:
                    # 次の状態はないので、Noneを格納
                    state_next = None

                    # 直近10episodeの立てたstep数リストに追加
                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))

                    if step < max_steps - 5:
                        # 途中でこけた場合、ペナルティとして報酬-1を与える
                        reward = torch.tensor([-1.0], dtype=torch.float, device=self.device)
                        # 連続成功記録をリセット
                        complete_episodes = 0
                    else:
                        # 立ったまま終了時は報酬1を与える
                        reward = torch.tensor([1.0], dtype=torch.float, device=self.device)
                        # 連続成功記録を更新
                        complete_episodes = complete_episodes + 1

                else:
                    # 普段は報酬0
                    reward = torch.tensor([0.0], dtype=torch.float, device=self.device)
                    # 観測をそのまま状態とする
                    state_next = observation_next
                    # numpy変数をPytorchのテンソルに変換
                    if self.device == "cuda":
                        state_next = torch.from_numpy(state_next).type(torch.cuda.FloatTensor)
                    else:
                        state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    # FloatTensor of size 4をsize 1*4に変換
                    state_next = torch.unsqueeze(state_next, 0)

                # メモリに経験を追加
                self.agent.memorize(state, action, state_next, reward)
                # Experience ReplayでQ関数を更新する
                self.agent.update_q_function()
                # 観測の更新
                state = state_next

                # 終了時の処理
                if done:
                    print('%d Episode: Finished after %d steps : 10Average = %.1lf' \
                    % (episode, step + 1, episode_10_list.mean()))

                    # ★★★DDQNで追加★★★
                    # ２試行に一度、Target Q-NetworkをMainと同じにコピーする
                    if(episode % 2 == 0):
                        self.agent.update_target_q_function()
                    break

            if episode_final:
                # 動画を保存し描画
                # self.display_frames_as_gif(frames)
                break

            if complete_episodes >= 10:
                print('10回連続成功！')
                # 次の試行を描画を行う最終試行とする
                episode_final = True

if __name__ == '__main__':
    rle = RLEnvironment()
    rle.run()
