import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import scipy.stats as ss
import scipy
from collections import defaultdict
import time
from scipy.optimize import minimize
import scipy.integrate as si
import copy
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import scipy.integrate as integrate
from flai.util.Time import Time

flupredict2color={ "1": "#004E98" ,
         "1C.2A.3A.4B": "#872853",
         "1C.2B.3D": "#CF48B1",
         "1C.2B.3G": "#F0E68C",
         "1C.2B.3J": "#7400B8",
         "1C.2B.3J.4E": "#1E90FF",
         "1C.2B.3J.4E.5B": "#B08968",
         "1C.2B.3J.4E.5N": "#B1A7A6",
         "1C.2B.3J.4E.5N.6J": "#D3D3D3",
         "1C.2B.3J.4E.5C": "#BA181B",
         "1C.2B.3J.4E.5C.6A": "#1F618D",
         "1C.2B.3J.4E.5C.6I.7C": "#C08552",
         "1C.2B.3J.4E.5C.6F": "#D39DC0",
         "1C.2B.3J.4D": "#4CC9F0",
         "1C.2B.3J.4D.5A": "#FF69B4",
         "1C.2B.3J.4F": "#6FBA78",
         "1C.2B.3J.4F.5D": "#344E41",
         "1C.2B.3J.4G": "#6FBA78",
         "1C.2B.3J.4G.5E": "#00AFB9",
         "1C.2B.3J.4G.5F": "#AFFC41",
         "1C.2B.3J.4G.5F.6B": "#ECF39E",
         "1C.2B.3J.4H": "#5E2F0B",
         "1C.2D.3F": "#FF0040"}
WHOlabels = {
'1C.2A.3A.4B':'BETA',
'1C.2A.3A.4A':'EPSILON',
'1C.2A.3A.4C':'IOTA',
'1C.2A.3I':'MU',
'1C.2B.3D':'ALPHA',
'1C.2B.3J':'OMICRON',
'1C.2B.3J.4D':'BA.1',
# '1C.2B.3J.4D.5A':'BA.1.1',
'1C.2B.3J.4E':'BA.2',
'1C.2B.3J.4E.5B':'BA.2.12.1',
'1C.2B.3J.4E.5L':'BJ.1',
'1C.2B.3J.4E.5C':'BA.2.75',
'1C.2B.3J.4E.5C.6A':'BA.2.75.2',
'1C.2B.3J.4E.5C.6E':'BM.1.1',
'1C.2B.3J.4E.5C.6F':'BN.1',
'1C.2B.3J.4F':'BA.4',
'1C.2B.3J.4F.5D':'BA.4.6',
'1C.2B.3J.4G':'BA.5',
'1C.2B.3J.4G.5K':'BA.5.9',
'1C.2B.3J.4G.5E':'BF.7',
'1C.2B.3J.4G.5F':'BQ.1',
'1C.2B.3J.4G.5F.6B':'BQ.1.1',
'1C.2D.3F':'DELTA',
'1C.2B.3G':'GAMMA',
'1C.2B.3J.4E.5N':'XBB',
'1C.2B.3J.4E.5N.6J':'XBB.1.5',
'1C.2B.3J.4E.5C.6I':'CH.1',
'1C.2B.3J.4E.5C.6I.7C':'CH.1.1'}

pango2flupredict = {a:b for b,a in WHOlabels.items()}

df = pd.read_csv("../output/data_immune_trajectories.txt",'\t',index_col=False)
df_R = pd.read_csv("../output/R_average.txt",'\t',index_col=False)
countries = sorted(list(set(df.country)))

df_R.loc[:,'x_BA.1'] = df_R['x_BA.1'] + df_R['x_BA.1.1']
df_R.loc[:,'x_BA.2'] = df_R['x_BA.2'] + df_R['x_BA.2.12.1']
df_R.loc[:,'x_BA.4/5'] = df_R['x_BA.4']+ df_R['x_BA.5'] + df_R['x_BA.5.9']
df_R.loc[:,'x_BQ.1'] = df_R['x_BQ.1']+ df_R['x_BQ.1.1'] 
df_R.loc[:,'x_XBB'] = df_R['x_XBB']+ df_R['x_XBB.1.5'] 
df_R.loc[:,'x_CH.1'] = df_R['x_CH.1'] + df_R['x_CH.1.1']
df_R.loc[:,'x_wt'] = 1 - df_R['x_ALPHA'] - df_R['x_DELTA'] - df_R['x_BETA'] - df_R['x_EPSILON'] - df_R['x_IOTA'] - df_R['x_MU'] - df_R['x_OMICRON'] - df_R['x_GAMMA'] - df_R['x_BA.1'] - df_R['x_BA.2'] - df_R['x_BA.4/5'] - df_R['x_BQ.1'] - df_R['x_XBB']
df_R.loc[:,'x_OMICRON'] = df_R['x_BA.1'] + df_R['x_BA.2']+ df_R['x_BA.2.75'] + df_R['x_BA.4/5']
df_R.pop("x_BA.1.1")
df_R.pop("x_BA.4")
df_R.pop("x_BA.5")
df_R.pop("x_BQ.1.1")
df_R.pop("x_XBB.1.5")
df_R.pop("x_CH.1.1")
df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01")),'x_wt'] = np.ones(len(df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01"))]))
df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01")),'x_ALPHA'] = np.zeros(len(df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01"))]))
df_R.loc[(df_R.time < Time.dateToCoordinate("2021-01-01")),'x_DELTA'] = np.zeros(len(df_R.loc[(df_R.time < Time.dateToCoordinate("2021-01-01"))]))
df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01")),'x_BQ.1'] = np.zeros(len(df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01"))]))
df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01")),'x_BA.4/5'] = np.zeros(len(df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01"))]))
df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01")),'x_BA.1'] = np.zeros(len(df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01"))]))
df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01")),'x_BA.2'] = np.zeros(len(df_R.loc[(df_R.time < Time.dateToCoordinate("2020-11-01"))]))

df['x_BA.1'] = df['x_BA.1'] + df['x_BA.1.1']
df['x_BA.2'] = df['x_BA.2'] + df['x_BA.2.12.1']
df['x_BA.4/5'] = df['x_BA.4']+ df['x_BA.5'] + df_R['x_BA.5.9']
df['x_BQ.1'] = df['x_BQ.1']+ df['x_BQ.1.1'] 
df['x_XBB'] = df['x_XBB']+ df['x_XBB.1.5'] 
df['x_CH.1'] = df['x_CH.1'] + df['x_CH.1.1']
df.pop("x_BA.1.1")
df.pop("x_BA.4")
df.pop("x_BA.5")
df.pop("x_BQ.1.1")
df.pop("x_XBB.1.5")
df.pop("x_CH.1.1")
df.loc[(df.time < Time.dateToCoordinate("2020-11-01")),'x_wt'] = np.ones(len(df.loc[(df.time < Time.dateToCoordinate("2020-11-01"))]))
df.loc[(df.time < Time.dateToCoordinate("2020-11-01")),'x_ALPHA'] = np.zeros(len(df.loc[(df.time < Time.dateToCoordinate("2020-11-01"))]))
df.loc[(df.time < Time.dateToCoordinate("2021-01-01")),'x_DELTA'] = np.zeros(len(df.loc[(df.time < Time.dateToCoordinate("2021-01-01"))]))
df.loc[(df.time < Time.dateToCoordinate("2020-11-01")),'x_BQ.1'] = np.zeros(len(df.loc[(df.time < Time.dateToCoordinate("2020-11-01"))]))
df.loc[(df.time < Time.dateToCoordinate("2020-11-01")),'x_BA.4/5'] = np.zeros(len(df.loc[(df.time < Time.dateToCoordinate("2020-11-01"))]))
df.loc[(df.time < Time.dateToCoordinate("2020-11-01")),'x_BA.1'] = np.zeros(len(df.loc[(df.time < Time.dateToCoordinate("2020-11-01"))]))
df.loc[(df.time < Time.dateToCoordinate("2020-11-01")),'x_BA.2'] = np.zeros(len(df.loc[(df.time < Time.dateToCoordinate("2020-11-01"))]))



vocs = ['wt','ALPHA','DELTA','BA.1','BA.2','BA.4/5','BQ.1','XBB']
voc2era = defaultdict(lambda: [])
for voc in vocs:
    for line in df_R.iterrows():
        line = line[1]
        x = line[f'x_{voc}']
        era = True
        for voc2 in vocs:
            if voc2 == voc:
                continue
            if x < line[f'x_{voc2}']:
                era = False
        if era:
            voc2era[voc].append(line.time)

freqs = [c for c in df_R.columns if c[:2] == 'x_']

fs = 12
ls=10
lw=2
alpha=0.15

plt.figure(figsize=(14,8))
plt.subplot(211)

era2color = {'wt':flupredict2color['1'],'ALPHA':flupredict2color[pango2flupredict['ALPHA']],'DELTA':flupredict2color[pango2flupredict['DELTA']],'BA.1':flupredict2color[pango2flupredict['BA.1']], 'BA.2':flupredict2color[pango2flupredict['BA.2']],
'BA.4/5':flupredict2color[pango2flupredict['BA.5']],'BQ.1':flupredict2color[pango2flupredict['BQ.1']],'XBB':flupredict2color[pango2flupredict['XBB']]}
for voc in ['ALPHA','DELTA','BA.1','BA.2','BA.4/5','BQ.1']:
    plt.barh(1.07,width=max(voc2era[voc]) - min(voc2era[voc]),height=0.05, left= min(voc2era[voc]), color = era2color[voc])

for voc_freq in freqs:
    voc = voc_freq[2:]
    if voc in ['wt', 'ALPHA','DELTA','BA.1','BA.2','BA.4/5','BA.4.6','BQ.1','XBB','BF.7','BN.1','CH.1']:
        if voc == 'CH.1':
            color = flupredict2color[pango2flupredict['CH.1.1']]
        elif voc == 'BA.4/5':
            color = flupredict2color[pango2flupredict['BA.5']]
        elif voc == 'wt':
            color=flupredict2color['1']
        else:
            color = flupredict2color[pango2flupredict[voc]]

        for country in countries:
            plt.plot(df.loc[df.country == country,'time'], df.loc[df.country == country,f'x_{voc}'], color=color,alpha=alpha,linewidth=1.0)

        plt.plot(df_R.loc[df_R[voc_freq] > 0.0,'time'], df_R.loc[df_R[voc_freq] > 0.0,voc_freq],'-',color =color, label=voc,linewidth=lw)

plt.ylabel("Variant Frequency",fontsize=fs)
plt.ylim([0,1.15])

xtick_labels = ['2020-01-01','2020-05-01','2020-09-01','2021-01-01','2021-05-01','2021-09-01','2022-01-01','2022-05-01','2022-09-01','2023-01-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Jan. $\'$20','May $\'$ 20','Sep. $\'$20', 'Jan. $\'$21','May $\'$ 21','Sep. $\'$21','Jan. $\'$22','May $\'$22','Sep. $\'$22','Jan. $\'$23']
plt.xticks(xtick_pos,xtick_labels,rotation=0,ha='center',fontsize=fs)
plt.xlim([Time.dateToCoordinate("2020-01-01"), Time.dateToCoordinate("2023-03-01")-1])
plt.legend(loc='center left',bbox_to_anchor=(1.02,0.5),prop={'size':ls})


plt.subplot(212)
for voc in ['ALPHA','DELTA','BA.1','BA.2','BA.4/5','BQ.1']:
    plt.barh(1.0,width=max(voc2era[voc]) - min(voc2era[voc]),height=0.05, left= min(voc2era[voc]), color = era2color[voc])
for country in countries:
    plt.plot(df.loc[df.country == country,'time'],df.loc[df.country == country,'tot_cases'],color='k',linewidth=1.0,alpha=alpha)
    plt.plot(df.loc[df.country == country,'time'],df.loc[df.country == country,'vac'],color='darkgrey',linewidth=1.0,alpha=alpha)
    plt.plot(df.loc[df.country == country,'time'],df.loc[df.country == country,'boost'],color='dimgrey',linewidth=1.0,alpha=alpha)
    plt.plot(df.loc[df.country == country,'time'],df.loc[df.country == country,'bivalent'],color='rosybrown',linewidth=1.0,alpha=alpha)

plt.plot(df_R.loc[df_R['tot_cases'] > 0.0,'time'], df_R.loc[df_R['tot_cases'] > 0.0,'tot_cases'],'-',color='k',label='tot_cases',linewidth=lw)
plt.plot(df_R.loc[df_R['vac'] > 0.0,'time'], df_R.loc[df_R['vac'] > 0.0,'vac'],'-',color='darkgrey',label='vac',linewidth=lw)
plt.plot(df_R.loc[df_R['boost'] > 0.0,'time'], df_R.loc[df_R['boost'] > 0.0,'boost'],'-',color='dimgrey',label='boost',linewidth=lw)
plt.plot(df_R.loc[df_R['bivalent'] > 0.0,'time'], df_R.loc[df_R['bivalent'] > 0.0,'bivalent'],'-',color='rosybrown',label='bivalent',linewidth=lw)
plt.legend(loc='center left',bbox_to_anchor=(1.02,0.5),prop={'size':ls})
xtick_labels = ['2020-01-01','2020-05-01','2020-09-01','2021-01-01','2021-05-01','2021-09-01','2022-01-01','2022-05-01','2022-09-01','2023-01-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Jan. $\'$20','May $\'$ 20','Sep. $\'$20', 'Jan. $\'$21','May $\'$ 21','Sep. $\'$21','Jan. $\'$22','May $\'$22','Sep. $\'$22','Jan. $\'$23']
plt.xticks(xtick_pos,xtick_labels,rotation=0,ha='center',fontsize=fs)
plt.xlim([Time.dateToCoordinate("2020-01-01"), Time.dateToCoordinate("2023-03-01")-1])
plt.ylim([0,1.08])
plt.ylabel("Population fraction",fontsize=fs)
plt.subplots_adjust(right=0.8)
plt.savefig("Fig1.pdf")
plt.close()

