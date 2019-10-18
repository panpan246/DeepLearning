import numpy as np
import matplotlib.pyplot as plt  # plt

#相当于x^n和y^n所对应的cp值
x_data = [338, 333, 328, 207, 226, 25, 179, 60, 208, 606]
y_data = [640, 633, 619, 393, 428, 27, 193, 66, 226, 1591]

#准备三维函数及待投影平面的网格坐标
x = np.arange(-200, -100, 1)  # bias
y = np.arange(-5, 5, 0.1)  # weight
Z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)
#[X,Y] = meshgrid(x,y) 将向量x和y定义的区域转换成矩阵X和Y,
# 其中矩阵X的行向量是向量x的简单复制，而矩阵Y的列向量是向量y的简单复制
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n]) ** 2
        Z[j][i] = Z[j][i] / len(x_data)

# yadata = b + w*xdata
b = -120  # intial b
w = -4  # intial w
lr = 0.0000001  # 学习率
iteration = 100000#迭代次数

#存储绘图的初始值
b_history = [b]
w_history = [w]

# iterations
for i in range(iteration):

    b_grad = 0.0
    w_grad = 0.0
    #L(w,b)对b,w分别求导
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
        w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]

    #更新参数相当于第一代版本的W^(t+1)=W^t-(lr)^t*g^t
    b = b - lr * b_grad
    w = w - lr * w_grad

    # 存储绘图参数
    b_history.append(b)
    w_history.append(w)

# plot the figure
plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
''' contourf() 函数为等高线图填充颜色.
x：指定 X 轴数据。
y：指定 Y 轴数据。
z：指定 X、Y 坐标对应点的高度数据。
colors：指定不同高度的等高线的颜色。
alpha：指定等高线的透明度。
cmap：指定等高线的颜色映射，即自动使用不同的颜色来区分不同的高度区域。
linewidths：指定等高线的宽度。
linestyles：指定等高线的样式'''
plt.plot([-188.4], [2.67], 'x', ms=6, markeredgewidth=3, color='orange')
# ms和marker分别代表指定点的长度和宽度。
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()