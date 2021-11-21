import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
plt.style.use('ggplot')
import os


# ggplot style C1=#348ABD
c1 = np.array([52,138,189]) / 255

rgb = lambda x: np.array(matplotlib.colors.to_rgb(x))

import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset



models=['fd8a93-248.pt', '1c1e4e-33.pt', '5cab11-53.pt', '6d87a6-235.pt']
model_names=[ 'All hydrogen', 'completely random 8 hydrogen','All methane' ,'completely random 8 methane']
columns = ['Sampling', 'Model', 'Log MSE']

data_h2=[]
data_ch4=[]

for i, model in enumerate(models):
    print("evalulating model {} {}".format(i, model))
    os.system("python evaluate.py ../data/iza/hydrogen/iza_hydrogen.npy output/{}".format(model))
    data = np.genfromtxt('nn-latents.csv',delimiter=",")
    #print(data)
    logMSE = np.log10(data[:, -1])
    #logMSE=logMSE.reshape((logMSE.shape[0],1))
    print(logMSE.shape)
    #data_h2.append(logMSE)
    for x in logMSE:
        data_h2.append([model_names[i], "Meta-learning", x])
    os.system("python evaluate.py ../data/iza/methane/iza_methane.npy output/{}".format(model))
    data = np.genfromtxt('nn-latents.csv',delimiter=",")
    #print(data)

    logMSE = np.log10(data[:, -1])
    print(logMSE.shape)
    #logMSE=logMSE.reshape((logMSE.shape[0],1))
    for x in logMSE:
        data_ch4.append([model_names[i], "Meta-learning", x])

    if model_names[i] == 'All hydrogen':
        os.system('python fit_isotherm.py ../data/iza/hydrogen/iza_hydrogen.npy best --test ../data/iza/hydrogen/iza_hydrogen.npy')
        data = np.genfromtxt('best-fit.csv',delimiter=",")
        data = np.delete(data, np.where(data<=0)[0])

        logMSE = np.log10(data)
        for x in logMSE:
            data_h2.append([model_names[i], "Best AIF", x])



    if model_names[i] == 'completely random 8 hydrogen':
        os.system('python fit_isotherm.py ../data/iza/hydrogen/iza_hydrogen_completely_random8.npy best --test ../data/iza/hydrogen/iza_hydrogen.npy')
        data = np.genfromtxt('best-fit.csv',delimiter=",")
        data = np.delete(data, np.where(data<=0)[0])

        logMSE = np.log10(data)
        #logMSE=logMSE.reshape((logMSE.shape[0],1))
        for x in logMSE:
            data_h2.append([model_names[i], "Best AIF", x])
        #fit_h2.append(logMSE)


    if model_names[i] == 'All methane':
        data = np.genfromtxt('best-fit.csv',delimiter=",")
        data = np.delete(data, np.where(data<=0)[0])

        logMSE = np.log10(data)
        print(logMSE.shape)
        for x in logMSE:
            data_ch4.append([model_names[i], "Best AIF", x])



    if model_names[i] == 'completely random 8 methane':

        os.system('python fit_isotherm.py ../data/iza/methane/iza_methane_completely_random8.npy best --test ../data/iza/methane/iza_methane.npy')
        data = np.genfromtxt('best-fit.csv',delimiter=",")
        data = np.delete(data, np.where(data<=0)[0])

        logMSE = np.log10(data)
        for x in logMSE:
            data_ch4.append([model_names[i], "Best AIF", x])



print(data_h2)
print(data_ch4)
#df1 = pd.DataFrame(data_h2)
#df1=pd.DataFrame(np.concatenate(data_h2,axis=1))

#df3=pd.DataFrame(np.concatenate(fit_h2,axis=1))
#print(df1)
df1 = pd.DataFrame(data_h2, columns=columns)

df2 = pd.DataFrame(data_ch4, columns=columns)
print(df1)
print(df2)

fig = plt.figure(figsize=(8, 2), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
plot = sns.violinplot(ax=ax, data=df1,
                   x="Sampling", y="Log MSE", hue="Model", split=False,
                   palette="Set2", linewidth=1, scale='count')
plot.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=6, fontsize=10, frameon=False)

#plt.savefig('performance_h2.png')
plt.savefig('performace_h2_singly_trained.pdf', format='pdf', bbox_inches='tight')

fig = plt.figure(figsize=(8, 2), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
plot = sns.violinplot(ax=ax, data=df2,x="Sampling", y="Log MSE", hue="Model", split=False,palette="Set2", linewidth=1, scale='count')
ax.set_xlabel("")
plot.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=6, fontsize=10, frameon=False)

#plt.savefig('performance_ch4.png')
plt.savefig('performace_ch4_singly_trained.pdf', format='pdf', bbox_inches='tight')
