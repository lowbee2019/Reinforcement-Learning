import numpy as np
import pandas as pd
import copy

class QLearning:
    def __init__(self,actions_num,learning_rate=0.9,reward_decay=0.9,e_greedy=0.1):
        # self.actions =[x for x in range(actions_num)]
        self.actions =actions_num
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions,dtype=np.float64)
        self.name = "QLearning"

    def __set__(self, instance, value):
        instance.epsilon = value
    def SetTarget(self,target):
        self.DESTINATION = target

    def choose_action(self,observation):
        self.check_state_exist(observation)
        state_action_values = self.q_table.loc[observation, :]
        if (np.random.uniform() < self.epsilon) :#or ((state_action_values==0).all()):
            action = np.random.choice(self.actions)
            # state_action_values = self.q_table.loc[observation,:]
            # action = np.random.choice(state_action_values[state_action_values==np.min(state_action_values)].index)
        else:
            action = np.random.choice(state_action_values[state_action_values==np.min(state_action_values)].index)
            # action = np.random.choice(self.actions)

        return action

    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)
        if s_ != self.DESTINATION:
            q_target = r + self.gamma*self.q_table.loc[s_,:].min()
        else:
            q_target = r
        loss = self.lr*(q_target - self.q_table.loc[s,a])
        self.q_table.loc[s,a] += loss

        return s_,loss

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            # print(state,"is not in table!")
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )

##### Sarsa lambda
class Sarsa_lambda:
    def __init__(self,actions_num,learning_rate=0.9,reward_decay=0.9,e_greedy=0.1,lambda_decay=0.3):
        # self.actions =[x for x in range(actions_num)]
        self.actions =actions_num
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.lambda_decay = lambda_decay
        self.q_table = pd.DataFrame(columns=self.actions,dtype=np.float64)
        self.e_table = pd.DataFrame(columns=self.actions,dtype=np.float64)
        self.name = "Sarsa_lambda"

    def __set__(self, instance, value):
        instance.epsilon = value

    def SetTarget(self,target):
        self.DESTINATION = target

    # def choose_action(self,observation):
    #     self.check_state_exist(observation)
    #
    #     if np.random.uniform() >= self.epsilon:
    #         state_action_values = self.q_table.loc[observation,:]
    #         action = np.random.choice(state_action_values[state_action_values==np.min(state_action_values)].index)
    #     else:
    #         action = np.random.choice(self.actions)
    #
    #     return action
    def choose_action(self, observation):
        self.check_state_exist(observation)
        state_action_values = self.q_table.loc[observation, :]
        if (np.random.uniform() < self.epsilon) or ((state_action_values == 0).all()):
            action = np.random.choice(self.actions)
            # state_action_values = self.q_table.loc[observation,:]
            # action = np.random.choice(state_action_values[state_action_values == np.min(state_action_values)].index)
        else:
            action = np.random.choice(state_action_values[state_action_values == np.min(state_action_values)].index)
            # action = np.random.choice(self.actions)

        return action

    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)
        a_ = self.choose_action(s_)
        if s_ != self.DESTINATION:

            q_target = r + self.gamma*self.q_table.loc[s_,a_]
        else:
            q_target = r

        error = q_target - self.q_table.loc[s,a]
        # self.e_table.loc[s,a] +=1
        self.e_table.loc[s,:]=0
        self.e_table.loc[s,a]=1
        self.q_table += self.lr*error*self.e_table

        if s_ !=self.DESTINATION:
            self.e_table *=self.gamma*self.lambda_decay
        else:
            self.e_table = pd.DataFrame(
                np.zeros(self.q_table.shape),
                index= self.q_table.index,
                columns=self.actions,
                dtype=np.float64)
        return s_,a_

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )
        if state not in self.e_table.index:
            self.e_table = self.e_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.e_table.columns,
                    name= state,
                )
            )


##### Sarsa

class Sarsa:
    def __init__(self,actions_num,learning_rate=0.9,reward_decay=0.9,e_greedy=0.1):
        # self.actions =[x for x in range(actions_num)]
        self.actions =actions_num
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions,dtype=np.float64)
        self.name = "Sarsa"

    def __set__(self, instance, value):
        instance.epsilon = value

    def SetTarget(self, target):
        self.DESTINATION = target

    # def choose_action(self,observation):
    #     self.check_state_exist(observation)
    #
    #     if np.random.uniform() >= self.epsilon:
    #
    #         state_action_values = self.q_table.loc[observation,:]
    #         action = np.random.choice(state_action_values[state_action_values==np.min(state_action_values)].index)
    #     else:
    #         action = np.random.choice(self.actions)
    #
    #     return action
    def choose_action(self, observation):
        self.check_state_exist(observation)
        state_action_values = self.q_table.loc[observation, :]
        if (np.random.uniform() < self.epsilon) :#or ((state_action_values == 0).all()):
            action = np.random.choice(self.actions)
            # state_action_values = self.q_table.loc[observation,:]
            # action = np.random.choice(state_action_values[state_action_values == np.min(state_action_values)].index)
        else:
            action = np.random.choice(state_action_values[state_action_values == np.min(state_action_values)].index)
            # action = np.random.choice(self.actions)

        return action

    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)
        a_ = self.choose_action(s_)
        if s_ != self.DESTINATION:
            q_target = r + self.gamma*self.q_table.loc[s_,a_] # different with Q learning
        else:
            q_target = r

        self.q_table.loc[s,a] += self.lr*(q_target - self.q_table.loc[s,a])

        return s_,a_

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            # print(state,"is not in table!")
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )

def Dijkstra(net, s, d):  # ????????????????????????s-d??????????????????????????????????????????
    network = copy.deepcopy(net)
    network[network==0] = 999
    row, col = np.diag_indices_from(network)
    network[row,col] = 0
    # np.diag_indices(network) = 0
    print("Start Dijstra Path??????")
    path = []  # ????????????s-d???????????????
    n = len(network)  # ????????????????????????????????????
    fmax = float('inf')
    w = [[0 for _ in range(n)] for j in range(n)]  # ???????????????????????????????????????0???max

    book = [0 for _ in range(n)]  # ????????????????????????????????????
    dis = [fmax for i in range(n)]  # s??????????????????????????????
    # book[s - 1] = 1  # ???????????????1????????????????????????0??????
    book[s] = 1
    midpath = [-1 for i in range(n)]  # ???????????????
    for i in range(n):
      for j in range(n):
        if network[i][j] != 0:
          w[i][j] = network[i][j]  # 0???max
        else:
          w[i][j] = fmax
        if i == s  and network[i][j] != 0:  # ?????????????????????????????????network[i][j]
          dis[j] = network[i][j]
    for i in range(n - 1):  # n-1??????????????????s??????
      min = fmax
      for j in range(n):
        if book[j] == 0 and dis[j] < min:  # ??????????????????????????????
          min = dis[j]
          u = j
      book[u] = 1
      for v in range(n):  # u???????????????????????????
        if dis[v] > dis[u] + w[u][v]:
          dis[v] = dis[u] + w[u][v]
          midpath[v] = u + 1  # ???????????????
    j = d - 1  # j?????????
    path.append(d)  # ?????????????????????????????????????????????????????????d???????????????
    while (midpath[j] != -1):
      path.append(midpath[j])
      j = midpath[j] - 1
    path.append(s)
    path.reverse()  # ????????????
    # print("path:",path)
    # print(midpath)
    # print("dis:",dis)
    # return path
    return dis

