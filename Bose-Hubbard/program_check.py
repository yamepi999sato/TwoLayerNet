import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

time_start = time.time()
J_list = np.linspace(0, 0.2, 20)
mu_list = np.linspace(0, 1, 10)
beta_list = np.empty([10, 20])
print(J_list)
print(mu_list)
for yi in range(10):
    for xi in range(20):
        beta_list[yi, xi] = xi + yi 

print(beta_list)



fig = plt.figure()
fig.suptitle(
    f"ElapsedTime:{time.time()-time_start:.2f}s, "
    f"date:{datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")
X, Y = np.meshgrid(J_list, mu_list)
print(X.shape)
print(Y.shape)
print(beta_list.shape)
plt.pcolormesh(X, Y, beta_list, shading='auto')
plt.colorbar()
plt.xlabel('J/U')
plt.ylabel('mu/U')
plt.show()