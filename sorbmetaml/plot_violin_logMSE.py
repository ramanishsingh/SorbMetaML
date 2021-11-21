import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



data = np.genfromtxt('nn-latents.csv',delimiter=",")
print(data)
logMSE = np.log10(data[:, -1])

#ax = sns.violinplot(y="logMSE", data=logMSE)
plt.figure()
plt.violinplot([logMSE])

plt.savefig("logMSE")

