import numpy as np
import matplotlib.pyplot as plt
x_data = [338,333,328,207,226,25,179,60,208,606]
y_data = [640,633,619,393,428,27,193,66,226,1591]
x = np.arange(-200,100,1)#bias
y = np.arange(-5,5,0.1)#weight
z = np.zeros((len(x),len(y)))
x,y = np.meshgrid(x,y)

# for i in range(len(x)-1):
#     for j in range(len(y)-1):
#         b = x[i]
#         w = y[j]
#         for n in range(len(x_data)):
#             z[i][j] = z[i][j] + (y_data[n] - b - w * x_data[n])**2
#
#         z[j][i] = z[j][i]/len(x_data)

b = 120
w = -4
lr = 1
iteration = 100000

b_history = [b]
w_history = [w]

lr_b=0
lr_w=0

for i in range(iteration):

    b_gard = 0.0
    w_gard = 0.0
    for n in range(len(x_data)):
        b_gard = b_gard - 2.0 * (y_data[n] - b - w*x_data[n])*1.0
        w_gard = w_gard - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]


    lr_b = lr_b + b_gard**2
    lr_w = lr_w + w_gard ** 2
    b = b - lr/np.sqrt(lr_b) * b_gard
    w = w - lr/np.sqrt(lr_w) * w_gard
    b_history.append(b)
    w_history.append(w)

#plt.contourf(x,y,z,50,alpha)
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,1100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()