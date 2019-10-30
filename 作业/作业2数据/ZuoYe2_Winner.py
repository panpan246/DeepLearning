import numpy as np
import pandas as pd

def train(x_train,y_train,epoch):
    num=x_train.shape[0]
    '''y.shape 返回的一个元组，代表 y 数据集的信息如（行，列）
    y.shape[0], 意思是：返回 y 中行的总数。这个值在 y 是单特征的情况下 和 len(y) 是等价的，
    即数据集中数据点的总数。'''
    dim=x_train.shape[1]
    bias=0#偏置初始化
    weights=np.ones(dim)#权重初始化
    Learning_rate=1#学习率和正则项系数初始化
    Regular_coefficient=0.001
    #用于存放偏置值的梯度平方和,adagrad用到
    bg2_sum=0
    wg2_sum=np.zeros(dim)

    #迭代求w,b
    for i in range(epoch):
        b_g=0#初始化
        w_g=np.zeros(dim)
       # 计算梯度，梯度计算时针对损失函数求导,在所有数据上
        for j in range(num):
            z=weights.dot(x_train[j,:])+bias#Z函数表达式
            sigmoid=1/(1+np.exp(-z))#sigmoid function
            #损失函数对b求导
            b_g+=((-1)*(y_train[j]-sigmoid))
            # 损失函数对w求导,并且有正则化（防overfitting)

            for k in range(dim):
                w_g[k]+=(-1)*(y_train[j]-sigmoid)*x_train[j,k]+2*Regular_coefficient*weights[k]
        #平均数
        b_g/=num
        w_g/=num

        #adagrad
        bg2_sum+=b_g**2
        wg2_sum+=w_g**2
        #更新w和b
        weights-=Learning_rate/wg2_sum**0.5*w_g
        bias-=Learning_rate/bg2_sum**0.5*b_g

    # 每训练3轮，输出一次在训练集上的正确率
    # 在计算loss时，由于涉及g()运到lo算，因此可能出现无穷大，计算并打印出来的loss为nan
    # 有兴趣的同学可以把下面涉及到loss运算的注释去掉，观察一波打印出的loss
        if i%3==0:
            Correct_quantity=0
            result=np.zeros(num)
            #loss=0
            for j in range(num):
                z = weights.dot(x_train[j, :]) + bias  # Z函数表达式
                sigmoid = 1 / (1 + np.exp(-z))  # sigmoid function
                if sigmoid>=0.5:
                    result[j]=1
                else:
                    result[j]=0
                if result[j]==y_train[j]:
                    Correct_quantity+=1.0
                #loss += (-1) * (y_train[j] * np.ln(sigmoid) + (1 - y_train[j]) * np.ln(1 - sigmoid))
            #print(f"epoch{0},the loss on train data is::{1}", i, loss / num)
            print(f"epoch{0},the Correct rate on train data is:{1}",i,Correct_quantity/num)
    return weights,bias

#对求出来的W和b验证一下效果
def validate(x_val,y_val,weights,bias):
    num=x_val.shape[0]
    Correct_quantity = 0
    result = np.zeros(num)
    loss=0
    for j in range(num):
        z = weights.dot(x_val[j, :]) + bias  # Z函数表达式
        sigmoid = 1 / (1 + np.exp(-z))  # sigmoid function
        if sigmoid >= 0.5:
            result[j] = 1
        if sigmoid < 0.5:
            result[j] = 0
        if result[j] == y_val[j]:
            Correct_quantity += 1.0
        #验证集上的损失函数
        loss += (-1) * (y_val[j] * np.ln(sigmoid) + (1 - y_val[j]) * np.ln(1 - sigmoid))

    return Correct_quantity/num


def main():
    #数据的预处理
    df=pd.read_csv('spam_train.csv')#读文件
    df=df.fillna(0)#空值用0填充
    array=np.array(df)#转化为对象（4000，49）
    x=array[:,1:-1]#抛弃第一列和最后一列shape(4000,47)
    y=array[:,-1]#最后一列label
    #将倒数第二列和第三列除以平均值
    x[:,-1]=x[:,-1]/np.mean(x[:,-1])
    x[:, -2] = x[:, -2] / np.mean(x[:, -2])

    #划分测试集和验证集
    x_train=x[0:3500,:]
    y_train = y[0:3500]
    x_val=x[3500:4001,:]
    y_val=y[3500:4001]

    #迭代次数为30次
    epoch=30
    w,b=train(x_train,y_train,epoch)
    #验证集上的结果
    Correct_rate=validate(x_val,y_val,w,b)
    print(f"The Correct rate on val data is:{0}",Correct_rate)

if __name__ == '__main__':
    main()