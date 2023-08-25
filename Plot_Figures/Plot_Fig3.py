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
import sys
sys.path.insert(0,"..")
from util.time import Time

pd.options.mode.chained_assignment = None  # default='warn'

flupredict2color={ "1": "#004E98" ,
         "1C.2A.3A.4B": "#872853",
         "1C.2B.3D": "#CF48B1",
         "1C.2B.3G": "#F0E68C",
         "1C.2B.3J": "#7400B8",
         "1C.2B.3J.4E": "#BC6C25",
         "1C.2B.3J.4E.5B": "#B08968",
         "1C.2B.3J.4E.5N": "#FF9F1C",
         "1C.2B.3J.4E.5N.6J": "#D3D3D3",
         "1C.2B.3J.4E.5C": "#BA181B",
         "1C.2B.3J.4E.5C.6A": "#1F618D",
         "1C.2B.3J.4E.5C.6I.7C": "#C08552",
         "1C.2B.3J.4E.5C.6F": "#D39DC0",
         '1C.2B.3J.4E.5C.6E':"#FF006E",
         "1C.2B.3J.4D": "#DDA15E",
         "1C.2B.3J.4D.5A": "#FF69B4",
         "1C.2B.3J.4F": "#6FBA78",
         "1C.2B.3J.4F.5D": "#335C67",
         "1C.2B.3J.4G": "#6FBA78",
         "1C.2B.3J.4G.5E": "#FF74FD",
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
countries = ['BELGIUM','CA','CANADA','FINLAND','FRANCE','GERMANY','ITALY','NETHERLANDS','NORWAY','NY','SPAIN','SWITZERLAND','USA'] 

#================================================================================
#Compute average fitness, f-fbar from frequency trajectories
#================================================================================
# vocs = ['wt','ALPHA','DELTA','OMICRON','BA.1','BA.2','BA.4/5','BQ.1','BF.7','XBB','CH.1']
vocs = ['wt','ALPHA','DELTA','OMICRON','BA.1','BA.2','BA.4/5','BQ.1','XBB']
shat_lines = []
dt = 30

for country in countries:
    df_country = df.loc[df.country == country]

    with open("../DATA/2023_04_01/freq_traj_" + country.upper() + ".json",'r') as f:
        freq_traj = json.load(f)
    with open("../DATA/2023_04_01/multiplicities_" + country.upper() + ".json",'r') as f:
        counts = json.load(f)
    with open("../DATA/2023_04_01/multiplicities_Z_" + country.upper() + ".json",'r') as f:
        Zdata = json.load(f)

    df_country['x_BA.1'] = df_country['x_BA.1'] + df_country['x_BA.1.1']
    df_country['x_BA.2'] = df_country['x_BA.2'] + df_country['x_BA.2.12.1']
    df_country['x_BA.4/5'] = df_country['x_BA.4']+ df_country['x_BA.5'] + df_country['x_BA.5.9']
    df_country['x_BQ.1'] = df_country['x_BQ.1']+ df_country['x_BQ.1.1'] 
    df_country['x_XBB'] = df_country['x_XBB']+ df_country['x_XBB.1.5'] 
    df_country['x_CH.1'] = df_country['x_CH.1'] + df_country['x_CH.1.1']
    df_country['x_wt'] = 1 - df_country['x_ALPHA'] - df_country['x_DELTA'] - df_country['x_BETA'] - df_country['x_EPSILON'] - df_country['x_IOTA'] - df_country['x_MU'] - df_country['x_OMICRON'] - df_country['x_GAMMA'] - df_country['x_BA.1'] - df_country['x_BA.2']


    for voc in vocs:
        if voc == 'OMICRON':
            df_here = df_country.loc[(df_country.time > Time.dateToCoordinate("2021-01-01"))]
            df_here['x_OMICRON'] = df_here['x_BA.1'] + df_here['x_BA.2']+ df_here['x_BA.2.75'] + df_here['x_BA.4/5']
            df_here = df_here.loc[(df_here[f'x_{voc}'] > 0.01) & (df_here.time<Time.dateToCoordinate("2022-02-10"))& (df_here.time>Time.dateToCoordinate("2021-09-01"))]
        elif voc == 'BA.1':
            df_here = df_country.loc[(df_country.time > Time.dateToCoordinate("2021-01-01"))]
            df_here = df_here.loc[(df_here[f'x_{voc}'] > 0.01) & (df_here.time>Time.dateToCoordinate("2022-01-10"))]
        elif voc == 'BA.2':
            df_here = df_country.loc[(df_country.time > Time.dateToCoordinate("2021-01-01"))]
            df_here = df_here.loc[(df_here[f'x_{voc}'] > 0.01) & (df_here.time<Time.dateToCoordinate("2022-07-01"))]
        elif voc == 'BA.4/5':
            df_here = df_country.loc[(df_country.time > Time.dateToCoordinate("2021-01-01"))]
            df_here = df_here.loc[df_here[f'x_{voc}'] > 0.01]
        elif voc == 'BQ.1':
            df_here = df_country.loc[(df_country.time > Time.dateToCoordinate("2021-01-01"))]
            df_here = df_here.loc[df_here[f'x_{voc}'] > 0.01]
        elif voc == 'XBB':
            df_here = df_country.loc[(df_country.time > Time.dateToCoordinate("2021-01-01"))]
            df_here = df_here.loc[df_here[f'x_{voc}'] > 0.1]
        elif voc == 'wt':
            df_here = df_country.loc[(df_country.time > Time.dateToCoordinate("2021-01-01"))]
            df_here = df_here.loc[(df_here[f'x_{voc}'] > 0.01) & (df_here.time < Time.dateToCoordinate("2021-04-01"))]
        elif voc == 'DELTA':
            df_here = df_country.loc[(df_country.time > Time.dateToCoordinate("2021-01-01"))]
            df_here['x_OMICRON'] = df_here['x_BA.1'] + df_here['x_BA.2']+ df_here['x_BA.2.75'] + df_here['x_BA.4/5']
            df_here = df_here.loc[(df_here[f'x_{voc}'] > 0.01) & (df_country.time > Time.dateToCoordinate("2021-01-01"))]
        else:
            df_here = df_country.loc[(df_country.time > Time.dateToCoordinate("2021-01-01")) & (df_country[f'x_{voc}']>0.01)]
        
        df_here.index = df_here.time

        #renomarlize on vocs:
        if voc in ['wt','ALPHA','DELTA','OMICRON']:
            Z = np.sum(df_here[['x_wt','x_ALPHA','x_DELTA','x_OMICRON']],axis=1)
            df_here['x_wt']    /= Z
            df_here['x_ALPHA'] /= Z
            df_here['x_DELTA'] /= Z
            df_here['x_OMICRON'] /= Z
        else:
            Z = np.sum(df_here[['x_BA.1','x_BA.2','x_BA.4/5','x_BQ.1','x_XBB']],axis=1)
            df_here['x_BA.1'] /= Z
            df_here['x_BA.2'] /= Z
            df_here['x_BA.4/5'] /= Z
            df_here['x_BQ.1'] /= Z
            df_here['x_XBB'] /= Z
            
        t0 = int(df_here.iloc[0].time)
        tf = t0 + dt
        while tf in list(df_here.time):
            time = int((t0+tf)/2)

            Z0 = np.exp(Zdata[str(t0)])
            Zt = np.exp(Zdata[str(tf)])

            x0 = df_here.loc[t0,f'x_{voc}']
            xt = df_here.loc[tf,f'x_{voc}']
            if x0 > 0.9999999:
                x0 = 0.9999999
            if xt > 0.9999999:
                xt = 0.9999999
            N0_sample = np.random.binomial(Z0, x0, 100) / Z0
            Nt_sample = np.random.binomial(Zt, xt, 100) / Zt
            N0_sample = np.where(N0_sample == 0, np.ones(len(N0_sample)) * 1/Z0, N0_sample)
            Nt_sample = np.where(Nt_sample == 0, np.ones(len(Nt_sample)) * 1/Zt, Nt_sample)

            shat = np.mean((np.log(Nt_sample) - np.log(N0_sample)) / dt)
            shat_var = np.var((np.log(Nt_sample) - np.log(N0_sample)) / dt)
            if shat_var == 0.0:
                shat_var = 1e-7

            shat_lines.append([country, voc,time,str(Time.coordinateToDate(int(time))), t0,tf,shat,shat_var,x0,xt])
            t0 += 7
            tf += 7

df_shat = pd.DataFrame(shat_lines, columns=['country','voc','time','date','t0','tf','shat','shat_var','x0','xt'])

for voc in vocs:
    country_variance = []

    for c in list(set(df_shat.country)):
        if len(df_shat.loc[(df_shat.country == c) & (df_shat.voc == voc)]) < 1:
            continue
        meansvar = np.mean(df_shat.loc[(df_shat.country == c) & (df_shat.voc == voc),'shat_var'])
        country_variance.append(meansvar)
    df_shat.loc[(df_shat.voc == voc),'shat_var'] += np.median(country_variance)

#================================================================================
#Compute fitness model for the average countries
#===============================================================================
gamma_vac_ad = 1.22049
gamma_inf_ad = 1.22049*2
gamma_vac_od = 0.27859
gamma_inf_od = 0.65718
gamma_inf_omi1 = 1.5924
gamma_inf_omi2 = 1.7450
F0 = [0.05212319, 0.02772249, 0.05731528, 0.07480703,0.0,0.0]


df_R = pd.read_csv("../output/R_average.txt",'\t',index_col=False)
df_R.loc[:,'x_BA.1'] = df_R['x_BA.1'] + df_R['x_BA.1.1']
df_R.loc[:,'x_BA.2'] = df_R['x_BA.2'] + df_R['x_BA.2.12.1']
df_R.loc[:,'x_BA.4/5'] = df_R['x_BA.4']+ df_R['x_BA.5'] + df_R['x_BA.5.9']
df_R.loc[:,'x_BQ.1'] = df_R['x_BQ.1']+ df_R['x_BQ.1.1'] 
df_R.loc[:,'x_XBB'] = df_R['x_XBB']+ df_R['x_XBB.1.5'] 
df_R.loc[:,'x_CH.1'] = df_R['x_CH.1'] + df_R['x_CH.1.1']
df_R.loc[:,'x_wt'] = 1 - df_R['x_ALPHA'] - df_R['x_DELTA'] - df_R['x_BETA'] - df_R['x_EPSILON'] - df_R['x_IOTA'] - df_R['x_MU'] - df_R['x_OMICRON'] - df_R['x_GAMMA'] - df_R['x_BA.1'] - df_R['x_BA.2'] - df_R['x_BA.4/5'] - df_R['x_BQ.1'] - df_R['x_XBB']
df_R.loc[:,'x_OMICRON'] = df_R['x_BA.1'] + df_R['x_BA.2']+ df_R['x_BA.2.75'] + df_R['x_BA.4/5']
df_here = df_R.loc[df_R.time > Time.dateToCoordinate("2021-01-01")]
vocs = ['wt','ALPHA','DELTA','OMICRON','BA.1','BA.2','BA.4/5','BQ.1','XBB']

lines = []
for line in df_here.iterrows():
    line = line[1]
    t = line.time
    if (t >= Time.dateToCoordinate("2022-04-21")) and (t <= Time.dateToCoordinate("2022-09-01")):
        gamma_vac = 0.27859
        gamma_inf =  1.5924
    if (t >= Time.dateToCoordinate("2021-09-01")) and (t < Time.dateToCoordinate("2022-04-21")):
        gamma_vac = 0.27859
        gamma_inf = 0.65718
    if t < Time.dateToCoordinate("2021-09-01"):
        gamma_vac = 1.22049
        gamma_inf = 1.22049 * 2
    if t > Time.dateToCoordinate("2022-09-01"):
        gamma_inf = 1.7450

    F_wt = -gamma_vac * (line['C_VAC_WT'] + line['C_BOOST_WT']) - gamma_inf * (line['C_RECOV_ALPHA_WT'] + line['C_RECOV_DELTA_WT'])
    F_alpha = -gamma_vac * (line['C_VAC_ALPHA'] + line['C_BOOST_ALPHA']) - gamma_inf * (line['C_RECOV_ALPHA_ALPHA'] + line['C_RECOV_DELTA_ALPHA'])
    F_delta = -gamma_vac * (line['C_VAC_DELTA'] + line['C_BOOST_DELTA']) - gamma_inf * (line['C_RECOV_ALPHA_DELTA'] + line['C_RECOV_DELTA_DELTA'] + line['C_RECOV_OMI_DELTA'])
    F_omi = -gamma_vac * (line['C_VAC_OMICRON'] + line['C_BOOST_OMICRON']) - gamma_inf * (line['C_RECOV_ALPHA_OMICRON'] + line['C_RECOV_DELTA_OMICRON'] + line['C_RECOV_OMI_BA.1'])

    F_ba1 = -gamma_vac * (line['C_BOOST_OMICRON'] + line['C_BIVALENT_BA.2']) - gamma_inf * (line['C_RECOV_BA1_BA.1'] + line['C_RECOV_BA2_BA.1'] + line['C_RECOV_BA45_BA.1'])
    F_ba2 = -gamma_vac * (line['C_BOOST_BA.2'] + line['C_BIVALENT_BA.2']) - gamma_inf * (line['C_RECOV_BA1_BA.2'] + line['C_RECOV_BA2_BA.2'] + line['C_RECOV_BA45_BA.2'])
    F_ba5 = -gamma_vac * (line['C_BOOST_BA.4/5'] + line['C_BIVALENT_BA.4/5']) - gamma_inf * (line['C_RECOV_BA1_BA.4/5'] + line['C_RECOV_BA2_BA.4/5'] + line['C_RECOV_BA45_BA.4/5'] + line['C_RECOV_BA45_BQ.1'])
    F_bq1 = -gamma_vac * (line['C_BOOST_BQ.1'] + line['C_BIVALENT_BQ.1']) - gamma_inf * (line['C_RECOV_BA1_BQ.1'] + line['C_RECOV_BA2_BQ.1'] + line['C_RECOV_BA45_BQ.1'] + line['C_RECOV_BQ1_BQ.1'])
    F_xbb = -gamma_vac * (line['C_BOOST_XBB'] + line['C_BIVALENT_XBB']) - gamma_inf * (line['C_RECOV_BA1_XBB'] + line['C_RECOV_BA2_XBB'] + line['C_RECOV_BA45_XBB'] + line['C_RECOV_BQ1_XBB'])

    Fitness = [F_wt,F_alpha,F_delta,F_omi,F_ba1,F_ba2,F_ba5,F_bq1,F_xbb]
    X = [line[f'x_{voc}'] for voc in vocs]
    lines.append(['Average', int(t), str(Time.coordinateToDate(int(t)))] + Fitness + X)

df_antigenic_fitness = pd.DataFrame(lines, columns=['country','time','date','F_wt','F_ALPHA','F_DELTA','F_OMICRON','F_BA.1','F_BA.2','F_BA.4/5','F_BQ.1','F_XBB'] + [f'x_{voc}' for voc in vocs])


df_pre = df_antigenic_fitness.loc[df_antigenic_fitness.time < Time.dateToCoordinate("2022-03-01")]
df_post = df_antigenic_fitness.loc[df_antigenic_fitness.time > Time.dateToCoordinate("2022-01-00")]

for dfline in df_pre.iterrows():
    dfline = dfline[1]
    F = np.array([dfline['F_wt'],dfline['F_ALPHA'],dfline['F_DELTA'],dfline['F_OMICRON']])
    F += np.array([0,F0[0],F0[0] + F0[1],F0[0] + F0[1] + F0[2]])
    X = np.array([dfline['x_wt'],dfline['x_ALPHA'],dfline['x_DELTA'],dfline['x_OMICRON']])
    X = X / np.sum(X)
    F_av = np.dot(F,X)
    F = F - F_av
    df_pre.loc[(df_pre.time == dfline.time), ['F_wt','F_ALPHA','F_DELTA','F_OMICRON']] = F
for dfline in df_post.iterrows():
    dfline = dfline[1]
    F = np.array([dfline['F_BA.1'],dfline['F_BA.2'],dfline['F_BA.4/5'],dfline['F_BQ.1'],dfline['F_XBB']])
    F += np.array(  [0, F0[3], F0[3] + F0[4], F0[3]+ F0[4]+ F0[5], F0[3]+ F0[4] + F0[5]]) 
    X = np.array([dfline['x_BA.1'],dfline['x_BA.2'],dfline['x_BA.4/5'],dfline['x_BQ.1'],dfline['x_XBB']])
    X = X / np.sum(X)
    F_av = np.dot(F,X)
    F = F - F_av
    df_post.loc[(df_post.time == dfline.time), ['F_BA.1','F_BA.2','F_BA.4/5','F_BQ.1','F_XBB']] = F

#average shat measurements:
av_shat = []
for voc in vocs:
    shat_here = df_shat.loc[df_shat.voc == voc]
    tmin = min(shat_here.time)
    tmax = max(shat_here.time)
    t = tmin
    while t + 7 < tmax:
        countries = sorted(list(set(shat_here.loc[(shat_here.time >= t) & (shat_here.time < t+7), 'country'])))
        if len(countries) < 3:
            t += 7
            continue
        shat = np.mean(shat_here.loc[(shat_here.time >= t) & (shat_here.time < t+7),'shat'])
        shat_var = np.sum(shat_here.loc[(shat_here.time >= t) & (shat_here.time < t+7),'shat_var']) / len(shat_here.loc[(shat_here.time >= t) & (shat_here.time < t+7),'shat_var'])**2 
        av_shat.append([voc, t+3, str(Time.coordinateToDate(t+3)), shat, shat_var,"_".join(countries)])
        t += 7
av_shat = pd.DataFrame(av_shat, columns=['voc','time','date','shat','shat_var','countries'])

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

ls = 10
fs= 12
ratio = 0.3
plt.figure(figsize=(14,12))
ax = plt.subplot(211)

era2color = {'wt':flupredict2color['1'],'ALPHA':flupredict2color[pango2flupredict['ALPHA']],'DELTA':flupredict2color[pango2flupredict['DELTA']],'BA.1':flupredict2color[pango2flupredict['BA.1']], 'BA.2':flupredict2color[pango2flupredict['BA.2']],
'BA.4/5':flupredict2color[pango2flupredict['BA.5']],'BQ.1':flupredict2color[pango2flupredict['BQ.1']],'XBB':flupredict2color[pango2flupredict['XBB']]}
for voc in ['ALPHA','DELTA','BA.1','BA.2','BA.4/5','BQ.1']:
    plt.barh(0.14,width=max(voc2era[voc]) - min(voc2era[voc]),height=0.01, left= min(voc2era[voc]), color = era2color[voc])
plt.axhline(0,color = 'k',alpha=0.5)

vocs = ['wt','ALPHA','DELTA','OMICRON','BA.1','BA.2','BA.4/5','BQ.1']
for voc in vocs:
    if voc == 'wt':
        color = flupredict2color['1']
    elif voc == 'BA.4/5':
        color =flupredict2color[pango2flupredict['BA.5']]
    elif voc == 'CH.1':
        color =flupredict2color[pango2flupredict['CH.1.1']]
    elif voc == 'OMICRON':
        color = flupredict2color[pango2flupredict['BA.1']]
    else:
        color =flupredict2color[pango2flupredict[voc]]
    plt.errorbar(av_shat.loc[(av_shat.voc == voc),'time'],av_shat.loc[(av_shat.voc == voc),'shat'], yerr = np.sqrt(av_shat.loc[(av_shat.voc == voc),'shat_var']),fmt='o', color=color,label=voc)
    
    t0 = av_shat.loc[av_shat.voc == voc].iloc[0].time
    tf = av_shat.loc[av_shat.voc == voc].iloc[-1].time
    if voc in ['wt','ALPHA','DELTA','OMICRON']:
        plt.plot(df_pre.loc[(df_pre.time > t0) & (df_pre.time < tf),'time'],df_pre.loc[(df_pre.time > t0) & (df_pre.time < tf),f'F_{voc}'], '-', color=color)
    if voc in ['BA.1','BA.2','BA.4/5','BQ.1','XBB','CH.1']:
        plt.plot(df_post.loc[(df_post.time > t0) & (df_post.time < tf),'time'],df_post.loc[(df_post.time > t0) & (df_post.time < tf),f'F_{voc}'], '-', color=color)

# plt.legend(loc='center left',bbox_to_anchor=(1.02,0.5),prop={'size':ls})
xtick_labels = ['2021-01-01','2021-05-01','2021-09-01','2022-01-01','2022-05-01','2022-09-01','2023-01-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Jan. $\'$21','May $\'$ 21','Sep. $\'$21','Jan. $\'$22','May $\'$22','Sep. $\'$22','Jan. $\'$23']
plt.xticks(xtick_pos,xtick_labels,rotation=0,ha='center',fontsize=fs)
plt.xlim([Time.dateToCoordinate("2021-01-01"), Time.dateToCoordinate("2023-03-01")-1])
plt.ylim([-0.12,0.15])
plt.ylabel("Fitness, $f_i(t)$",fontsize=fs)
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
plt.tick_params(direction='in',labelsize=ls)

ax = plt.subplot(212)
vocs = ['ALPHA','DELTA','OMICRON','BA.2','BA.4/5','BQ.1']

for voc in ['ALPHA','DELTA','BA.1','BA.2','BA.4/5','BQ.1']:
    plt.barh(0.09,width=max(voc2era[voc]) - min(voc2era[voc]),height=0.01, left= min(voc2era[voc]), color = era2color[voc])

plt.axhline(0,color = 'k',alpha=0.5)
plt.ylabel("Selection breakdown, $s_k$",fontsize=fs)
#alpha-wt
t = int(df_pre.loc[df_pre['x_ALPHA'] > df_pre['x_wt']].iloc[0].time)
line = df_R.loc[df_R.time == t].iloc[0]
s0 = F0[0]
s_vac = gamma_vac_ad * (line['C_VAC_WT'] - line['C_VAC_ALPHA'])
s_alpha = gamma_inf_ad * (line['C_RECOV_ALPHA_WT'] - line['C_RECOV_ALPHA_ALPHA'])

times = np.arange(Time.dateToCoordinate("2021-01-01"),Time.dateToCoordinate("2021-04-01"))
plt.bar([t-28,t-14,t],[s0,s_vac,s_alpha],color=['k','#00B4D8',flupredict2color[pango2flupredict['ALPHA']]], width=14)

#delta-alpha
t = int(df_pre.loc[(df_pre.time > Time.dateToCoordinate("2021-05-01")) & (df_pre['x_DELTA'] > df_pre['x_ALPHA'])].iloc[0].time)
line = df_R.loc[df_R.time == t].iloc[0]
s0 = F0[1]
s_vac = gamma_vac_ad * (line['C_VAC_ALPHA'] - line['C_VAC_DELTA'])
s_alpha = gamma_inf_ad * (line['C_RECOV_ALPHA_ALPHA'] - line['C_RECOV_ALPHA_DELTA'])
s_delta = gamma_inf_ad * (line['C_RECOV_DELTA_ALPHA'] - line['C_RECOV_DELTA_DELTA'])
times = np.arange(Time.dateToCoordinate("2021-05-01"),Time.dateToCoordinate("2021-09-01"))
plt.bar([t-28,t-14,t,t+14],[s0,s_vac,s_alpha,s_delta],color=['k','#00B4D8',flupredict2color[pango2flupredict['ALPHA']],flupredict2color[pango2flupredict['DELTA']]], width=14)
gamma_vac = 0.29
gamma_inf = 0.58

#omi-delta
t = int(df_pre.loc[(df_pre.time > Time.dateToCoordinate("2021-10-01")) & (df_pre['x_OMICRON'] > df_pre['x_DELTA'])].iloc[0].time)
line = df_R.loc[df_R.time == t].iloc[0]
s0 = F0[2]
s_vac = gamma_vac_od * (line['C_VAC_DELTA'] - line['C_VAC_OMICRON'])
s_vac0 = gamma_vac_od * (line['C_VAC0_DELTA'] - line['C_VAC0_OMICRON'])
s_boost = gamma_vac_od * (line['C_BOOST_DELTA'] - line['C_BOOST_OMICRON'])
s_boost = s_vac - s_vac0 + s_boost
s_alpha = gamma_inf_od * (line['C_RECOV_ALPHA_DELTA'] - line['C_RECOV_ALPHA_OMICRON'])
s_delta = gamma_inf_od * (line['C_RECOV_DELTA_DELTA'] - line['C_RECOV_DELTA_OMICRON'])
s_omi = gamma_inf_od * (line['C_RECOV_OMI_DELTA'] - line['C_RECOV_OMI_BA.1'])
times = np.arange(Time.dateToCoordinate("2021-10-15"),Time.dateToCoordinate("2022-01-25"))
t -= 20
plt.bar([t-28,t-14,t,t+14,t+28,t+42],[s0,s_vac0,s_boost,s_alpha,s_delta,s_omi],color=['k','#00B4D8','#0077B6',flupredict2color[pango2flupredict['ALPHA']],flupredict2color[pango2flupredict['DELTA']],flupredict2color[pango2flupredict['OMICRON']]], width=14)

#ba.2-ba.1
df_post.loc[(df_post['x_BA.2'] > df_post['x_BA.1'])]
t = int(df_post.loc[(df_post.time > Time.dateToCoordinate("2021-10-01")) & (df_post['x_BA.2'] > df_post['x_BA.1'])].iloc[0].time)
line = df_R.loc[df_R.time == t].iloc[0]
s0 = F0[3]
s_boost = gamma_vac_od * (line['C_BOOST_OMICRON'] - line['C_BOOST_BA.2'])
s_ba1 = gamma_inf_od * (line['C_RECOV_BA1_BA.1'] - line['C_RECOV_BA1_BA.2'])
s_ba2 = gamma_inf_od * (line['C_RECOV_BA2_BA.1'] - line['C_RECOV_BA2_BA.2'])
times = np.arange(Time.dateToCoordinate("2022-01-25"),Time.dateToCoordinate("2022-04-15"))
t+=7
plt.bar([t-28,t-14,t,t+14],[s0,s_boost,s_ba1,s_ba2],color=['k','#0077B6',flupredict2color[pango2flupredict['BA.1']],flupredict2color[pango2flupredict['BA.2']]], width=14)

#ba.45-ba.2
t = int(df_post.loc[(df_post.time > Time.dateToCoordinate("2022-05-01")) & (df_post['x_BA.4/5'] > df_post['x_BA.2'])].iloc[0].time)
line = df_R.loc[df_R.time == t].iloc[0]
s0 = F0[4]
s0 = 0.0
s_boost =  gamma_vac_od * (line['C_BOOST_BA.2'] - line['C_BOOST_BA.4/5'])
s_ba1 = gamma_inf_omi1 * (line['C_RECOV_BA1_BA.2'] - line['C_RECOV_BA1_BA.4/5'])
s_ba2 = gamma_inf_omi1 * (line['C_RECOV_BA2_BA.2'] - line['C_RECOV_BA2_BA.4/5'])
s_ba45 = gamma_inf_omi1 * (line['C_RECOV_BA45_BA.2'] - line['C_RECOV_BA45_BA.4/5'])
times = np.arange(Time.dateToCoordinate("2022-05-01"),Time.dateToCoordinate("2022-07-20"))
plt.bar([t-28,t-14,t,t+14,t+28],[s0,s_boost,s_ba1,s_ba2,s_ba45],color=['k','#0077B6',flupredict2color[pango2flupredict['BA.1']],flupredict2color[pango2flupredict['BA.2']],flupredict2color[pango2flupredict['BA.5']]], width=14)

#bq1-ba45
t = int(df_post.loc[(df_post.time > Time.dateToCoordinate("2022-09-01")) & (df_post['x_BQ.1'] > df_post['x_BA.4/5'])].iloc[0].time)
line = df_R.loc[df_R.time == t].iloc[0]
s0 = 0.0
s_boost =  gamma_vac_od * (line['C_BOOST_BA.4/5'] - line['C_BOOST_BQ.1'])
s_biv =  gamma_vac_od * (line['C_BIVALENT_BA.4/5'] - line['C_BIVALENT_BQ.1'])
s_ba1 = gamma_inf_omi2 * (line['C_RECOV_BA1_BA.4/5'] - line['C_RECOV_BA1_BQ.1'])
s_ba2 = gamma_inf_omi2 * (line['C_RECOV_BA2_BA.4/5'] - line['C_RECOV_BA2_BQ.1'])
s_ba45 = gamma_inf_omi2 * (line['C_RECOV_BA45_BA.4/5'] - line['C_RECOV_BA45_BQ.1'])
s_bq1 = gamma_inf_omi2 * (line['C_RECOV_BQ1_BA.4/5'] - line['C_RECOV_BQ1_BQ.1'])
times = np.arange(Time.dateToCoordinate("2022-09-10"),Time.dateToCoordinate("2022-12-30"))
plt.bar([t-42, t-28,t-14,t,t+14,t+28,t+42],[s0,s_boost,s_biv,s_ba1,s_ba2,s_ba45, s_bq1],color=['k','#0077B6','#03045E',flupredict2color[pango2flupredict['BA.1']],flupredict2color[pango2flupredict['BA.2']],flupredict2color[pango2flupredict['BA.5']],flupredict2color[pango2flupredict['BQ.1']]], width=14)

xtick_labels = ['2021-01-01','2021-05-01','2021-09-01','2022-01-01','2022-05-01','2022-09-01','2023-01-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Jan. $\'$21','May $\'$ 21','Sep. $\'$21','Jan. $\'$22','May $\'$22','Sep. $\'$22','Jan. $\'$23']
xtick_labels = ['', '', '', '', '', '', '']
plt.xticks(xtick_pos,xtick_labels,rotation=0,ha='center',fontsize=fs)
plt.xticks([],[],rotation=0,ha='center',fontsize=fs)
plt.xlim([Time.dateToCoordinate("2021-01-01"), Time.dateToCoordinate("2023-03-01")-1])
plt.ylim([-0.01,0.1])
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
plt.tick_params(direction='in',labelsize=ls)
legend_elements = []
ms=4
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color['1'],linestyle='-',label='wt', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['ALPHA']],linestyle='-',label='Alpha', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['DELTA']],linestyle='-',label='Delta', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['OMICRON']],linestyle='',label='Omicron', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['BA.1']],linestyle='-',label='Omicron o(BA.1)', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['BA.2']],linestyle='-',label='Omicron BA.2', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['BA.5']],linestyle='-',label='Omicron BA.4/5', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['BQ.1']],linestyle='-',label='Omicron BQ.1', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['XBB']],linestyle='-',label='Omicron XBB', linewidth=2.0))

plt.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1.05,0.5),prop={'size':ls})
plt.savefig('Fig3.pdf')
plt.close()
