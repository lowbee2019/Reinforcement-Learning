import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from policy import *

# t = [[0, 2, 20, 0, 0, 0, 0],
#     [2, 0, 0, 15, 10, 0, 0],
#     [20, 0, 0, 10, 0, 9, 0],
#     [0, 15, 10, 0, 0, 0, 1],
#     [0, 10, 0, 0, 0, 0, 16],
#     [0, 0, 9, 0, 0, 0, 3],
#     [0, 0, 0, 1, 16, 3, 0]]


class Node:
    def __init__(self,index,GT):
        self.index = index

        self.neighbor = []
        for i in range(len(GT)):
            if(GT[self.index,i]):
                self.neighbor.append(i)

        self.Sarsa_Agent = Sarsa(actions_num=self.neighbor)
        self.Q_Agent = QLearning(actions_num=self.neighbor)
        self.Slambda_Agent = Sarsa_lambda(actions_num=self.neighbor)
        self.agent=[self.Q_Agent,self.Sarsa_Agent,self.Slambda_Agent]

    def routing(self,idx,observation,GT,target,is_trainable = False):
        a = self.agent[idx].choose_action(target)
        r = GT[observation,a]
        if is_trainable:
            self.agent[idx].learn(target,a,r,a)
        return a,r


def S_T(num):
    S = np.random.randint(0,num-1)
    T = np.random.randint(0,num-1)
    while(S==T):
        T = np.random.randint(0,num-1)
    return S,T


def TrainOnce(NodeNum):
    S,T = S_T(num=NodeNum)
    for i in range(3): #the num of agents
        Tmp = S
        while(Tmp!=T):
            N[Tmp].agent[i].SetTarget(T)
            Tmp,_ = N[Tmp].routing(i,Tmp,t,T,is_trainable=True)

    print("{0}->{1} is trained!".format(S,T))


def TestOnce(NodeNum,MinDis,TestRecord):
    S,T = S_T(num=NodeNum)
    RL_length = []
    for i in range(3): #the num of agents
        Tmp = S
        length = 0
        while(Tmp!=T):
            N[Tmp].agent[i].SetTarget(T)
            Tmp,l = N[Tmp].routing(i,Tmp,t,T)
            length +=l
        RL_length.append(length)
    print("{0}->{1} is tested!".format(S,T))
    RL_length.append(MinDis[S, T])
    TestRecord[(S,T)] = RL_length


def Train(SampleNum,NodeNum):
    while(SampleNum):
        TrainOnce(NodeNum)
        SampleNum-=1


def Test(SampleNum,NodeNum,MinDis,TestRecord):
    while(SampleNum):
        TestOnce(NodeNum,MinDis,TestRecord)
        SampleNum-=1


def GetMinDist(MinDis,Map):
    dis = []
    for i in range(0, len(Map)):
        temp = Dijkstra(Map, i, len(Map))
        dis.append(temp)
        MinDis[i,] = temp
    r,c = np.diag_indices_from(MinDis)
    MinDis[r,c] = 0
    return MinDis


def GetDataFrame(Dict):
    d = pd.DataFrame(columns=('Q-learning', 'Sarsa', 'Sarsa_lambda','dijkstra'))
    for k,v in Dict.items():
        d.loc[str(k)] = v
    return d


def DrawBar(Dict):
    l = len(Dict)
    Q=[]
    S=[]
    SL=[]
    D=[]
    labels = []
    for k,v in Dict.items():
        labels.append(k)
        Q.append(v[0])
        S.append(v[1])
        SL.append(v[2])
        D.append(v[3])

    bar_width = 0.2
    width = 0.2
    x = np.arange(l)
    plt.bar(x,Q,bar_width,width,label='Q_learning')
    plt.bar(x+bar_width,S,width,label='Sarsa')
    plt.bar(x+2*bar_width,SL,width,label='Sarsa lambda')
    plt.bar(x+3*bar_width,D,width,label='Dijkstra')

    plt.legend()
    plt.tick_params(axis='x', labelsize=5)
    plt.xticks(x+bar_width,labels)
    plt.show()


if __name__ == '__main__':


    t = [[0, 2, 20, 0, 0, 0, 0],
         [2, 0, 0, 15, 10, 0, 0],
         [20, 0, 0, 10, 0, 9, 0],
         [0, 15, 10, 0, 0, 0, 1],
         [0, 10, 0, 0, 0, 0, 16],
         [0, 0, 9, 0, 0, 0, 3],
         [0, 0, 0, 1, 16, 3, 0]]
    t = np.array(t)
    NodeNum= len(t)
    MinDis = np.zeros((len(t),len(t)))
    MinDis = GetMinDist(MinDis,t)
    N = [Node(i,t) for i in range(len(t))]

    Train(SampleNum=20,NodeNum=NodeNum)

    TestRecord = dict()
    Test(SampleNum=50,NodeNum=NodeNum,MinDis=MinDis,TestRecord=TestRecord)

    dataframe = GetDataFrame(TestRecord)
    print(dataframe)
    DrawBar(TestRecord)
