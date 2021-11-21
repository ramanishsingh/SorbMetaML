import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



data = np.genfromtxt('best-fit.csv',delimiter=",")
print(data)
data = np.delete(data, np.where(data<=0)[0])

logMSE = np.log10(data)

#ax = sns.violinplot(y="logMSE", data=logMSE)
plt.figure()
plt.violinplot([logMSE])

plt.savefig("logMSE-fit.png")

