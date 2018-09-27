'''
TD誤差を格納するメモリクラス
'''
import numpy as np
import RLUtil

class TDErrorMemory:

    def __init__(self, capacity):
        # 定数用クラス
        self.util = RLUtil.RLUtil()
        # メモリの最大長さ
        self.capacity = capacity
        # 経験を保存する変数
        self.memory = []
        # 保存するindexを示す変数
        self.index = 0
    
    def push(self, td_error):
        '''
        TD誤差をメモリに保存する
        '''

        # メモリが満タンでない場合は足す
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = td_error
        # 保存するインデックスを１つずらす
        self.index = (self.index + 1) % self.capacity
    
    def __len__(self):
        '''
        関数lenに対して、現在の変数memoryの長さを返すようにオーバーライド
        '''
        return len(self.memory)
    
    def get_prioritized_indexes(self, batch_size):
        '''
        TD誤差に応じた確率でindexを取得
        '''

        # TD誤差の和を計算
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        # 微小値を足す
        sum_absolute_td_error += self.util.getTD_ERROR_EPSILON() * len(self.memory)

        # batch_size分の乱数を生成して、昇順に並べる
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        # 作成した乱数で串刺しにして、インデックスを求める
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (abs(self.memory[idx]) + self.util.getTD_ERROR_EPSILON())
                idx += 1
            # 微小値を計算に使用した関係でindexがメモリの長さを超えた場合の補正
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)
        
        return indexes
    
    def update_td_error(self, updated_td_errors):
        '''
        TD誤差の更新
        '''
        self.memory = updated_td_errors