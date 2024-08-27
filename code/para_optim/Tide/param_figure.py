import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sharpe = np.array([2.631, 2.671, 2.615, 2.593, 2.693, 2.532, #12
                   2.627, 2.637, 2.643, 2.581, 2.669, 2.516, #13
                   2.721, 2.754, 2.732, 2.665, 2.790, 2.649, #14
                   2.658, 2.686, 2.642, 2.571, 2.718, 2.555, #15
                   2.651, 2.653, 2.646, 2.563, 2.704, 2.567, #16
                   2.582, 2.610, 2.580, 2.510, 2.642, 2.483, #17
                   2.438, 2.433, 2.403, 2.363, 2.488, 2.346, #18
                   2.399, 2.395, 2.405, 2.342, 2.473, 2.314, #19
                   2.421, 2.437, 2.470, 2.388, 2.538, 2.369, #20
                   2.468, 2.473, 2.484, 2.418, 2.618, 2.448, #21
                   2.370, 2.363, 2.383, 2.306, 2.498, 2.324, #22
                   2.386, 2.373, 2.375, 2.302, 2.506, 2.331, #23
                   ]) 

dates = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
minutes = [5, 7, 9, 11, 13, 15]

df = pd.DataFrame(sharpe.reshape((len(dates), len(minutes))), columns=minutes, index=dates).T

# 打印DataFrame以查看
print(df)

# 创建网格
X, Y = np.meshgrid(df.columns, df.index)
Z = df.values

# 绘制曲面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Days')
ax.set_ylabel('Minutes')
ax.set_zlabel('Sharpe Ratio')

# 保存图像
plt.savefig('/home/zhanxy/result/figures/para_optim.png', dpi=1000)

plt.show()
