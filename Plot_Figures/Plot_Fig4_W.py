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
from scipy import stats as ss
from sklearn.linear_model import LinearRegression as LR

pd.options.mode.chained_assignment = None  # default='warn'

WHOlabels = {
'1C.2A.3A.4B':'BETA',
'1C.2A.3A.4A':'EPSILON',
'1C.2A.3A.4C':'IOTA',
'1C.2A.3I':'MU',
'1C.2B.3D':'ALPHA',
'1C.2B.3J':'OMICRON',
'1C.2B.3J.4D':'BA.1',
'1C.2B.3J.4D.5A':'BA.1.1',
'1C.2B.3J.4E':'BA.2',
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
df_R.loc[:,'x_BA.1'] = df_R['x_BA.1'] + df_R['x_BA.1.1']
df_R.loc[:,'x_BA.2'] = df_R['x_BA.2'] + df_R['x_BA.2.12.1']
df_R.loc[:,'x_BA.4/5'] = df_R['x_BA.4']+ df_R['x_BA.5'] + df_R['x_BA.5.9']
df_R.loc[:,'x_BQ.1'] = df_R['x_BQ.1']+ df_R['x_BQ.1.1'] 
df_R.loc[:,'x_XBB'] = df_R['x_XBB']+ df_R['x_XBB.1.5'] 
df_R.loc[:,'x_CH.1'] = df_R['x_CH.1'] + df_R['x_CH.1.1']
df_R.loc[:,'x_wt'] = 1 - df_R['x_ALPHA'] - df_R['x_DELTA'] - df_R['x_BETA'] - df_R['x_EPSILON'] - df_R['x_IOTA'] - df_R['x_MU'] - df_R['x_OMICRON'] - df_R['x_GAMMA'] - df_R['x_BA.1'] - df_R['x_BA.2']
df_R.loc[:,'x_OMICRON'] = df_R['x_BA.1'] + df_R['x_BA.2']+ df_R['x_BA.2.75'] + df_R['x_BA.4/5']
df_R.index = df_R.time
countries = sorted(list(set(df.country)))

gamma_inf_update = pd.read_csv("../output/Update_gamma_inf.txt",'\t',index_col=False)

countries = ['BELGIUM','CA','CANADA','FINLAND','FRANCE','GERMANY','ITALY','NETHERLANDS','NORWAY','NY','SPAIN','SWITZERLAND','USA'] #Belgium, Finland not included for bq.1

gamma_vac = 0.27859
#===========================================================================
#W-What, excluding new variants Make data and figure
#===========================================================================
vocs = ['BA.2','BA.5','BQ.1','BA.4.6','BF.7','XBB','CH.1','BN.1']
lines = []
for country in countries:
    with open("../DATA/2023_04_01/freq_traj_" + country.upper() + ".json",'r') as f:
        freq_traj = json.load(f)
    with open("../DATA/2023_04_01/multiplicities_" + country.upper() + ".json",'r') as f:
        counts = json.load(f)
    with open("../DATA/2023_04_01/multiplicities_Z_" + country.upper() + ".json",'r') as f:
        Z = json.load(f)

    df_country = df.loc[df.country == country]
    df_country['x_BA.1'] = df_country['x_BA.1'] + df_country['x_BA.1.1']
    df_country['x_BA.2'] = df_country['x_BA.2'] + df_country['x_BA.2.12.1']
    df_country['x_BA.4/5'] = df_country['x_BA.4']+ df_country['x_BA.5'] + df_country['x_BA.5.9']
    df_country['x_BQ.1'] = df_country['x_BQ.1']+ df_country['x_BQ.1.1'] 
    df_country['x_XBB'] = df_country['x_XBB']+ df_country['x_XBB.1.5'] 
    df_country['x_CH.1'] = df_country['x_CH.1'] + df_country['x_CH.1.1']
    df_country['x_wt'] = 1 - df_country['x_ALPHA'] - df_country['x_DELTA'] - df_country['x_BETA'] - df_country['x_EPSILON'] - df_country['x_IOTA'] - df_country['x_MU'] - df_country['x_OMICRON'] - df_country['x_GAMMA'] - df_country['x_BA.1'] - df_country['x_BA.2']

    df_here = df_country.loc[df_country.time > Time.dateToCoordinate("2022-04-01")]
    df_here.index = df_here.time

    dt = 75
    for focal_voc in ['BA.2','BA.5','BQ.1','BA.4.6','BF.7','XBB','CH.1','BN.1']:
        gamma_vac = 0.27859
        if len(df_here.loc[df_here[f'x_{focal_voc}']>0.01]) < 1:
            continue
        t0_1 = df_here.loc[df_here[f'x_{focal_voc}']>0.01].iloc[0].time
        t0_2 = 100000
        t0_3 = 100000
        if len(df_here.loc[df_here[f'x_{focal_voc}']>0.2]) > 1:
            t0_2 = df_here.loc[df_here[f'x_{focal_voc}']>0.2].iloc[0].time
        if len(df_here.loc[df_here[f'x_{focal_voc}']>0.4]) > 1:
            t0_3 = df_here.loc[df_here[f'x_{focal_voc}']>0.4].iloc[0].time
        max_x = max(df_here[f'x_{focal_voc}'])
        max_t = df_here.loc[df_here[f'x_{focal_voc}'] == max_x,'time'].iloc[0]
        df_here.loc[df_here.time > max_t]
        t0_4 = 100000
        t0_5 = 100000
        if len(df_here.loc[(df_here.time > max_t) & (df_here[f'x_{focal_voc}'] < max_x * 0.5)]) > 1:
            t0_4 = df_here.loc[(df_here.time > max_t) & (df_here[f'x_{focal_voc}'] < max_x * 0.5)].iloc[0].time
        if len(df_here.loc[(df_here.time > max_t) & (df_here[f'x_{focal_voc}'] < max_x * 0.25)]) > 1:
            t0_5 = df_here.loc[(df_here.time > max_t) & (df_here[f'x_{focal_voc}'] < max_x * 0.25)].iloc[0].time

        t0_list = [t0_1, t0_2, t0_3, max_t,t0_4,t0_5]
        if focal_voc == 'BM.1.1':
            t0_list = [t0_1]
        for t0_index, t0 in enumerate(t0_list):
            tf = t0+dt
            if tf > df_here.iloc[-1].name:
                continue
            if focal_voc == 'BA.2' and t0_index < 2:
                continue
            #Get gamma_vac, gamma_inf 
            if t0 < 44689 + 15: #First update of gamma_inf at t = 44704 (24-05-2022)
                gamma_inf = 0.65718
            else:
                gamma_inf = gamma_inf_update.loc[gamma_inf_update.time_cut_off == t0 - 15].iloc[0].gamma_inf_new

            x0 = df_here.loc[t0,f'x_{focal_voc}']
            xt = df_here.loc[tf,f'x_{focal_voc}']

            #present vocs at t0
            voc_here = df_here.loc[t0,[f'x_{voc}' for voc in vocs]] 
            voc_here = list(voc_here[voc_here>0.001].index)
            voc_here = [a[2:] for a in voc_here]

            #Estimate error bar on W:
            clade_flupredict = [pango2flupredict[focal_voc]]
            if focal_voc == 'BQ.1':
                clade_flupredict.append(pango2flupredict['BQ.1.1'])
            elif focal_voc == 'XBB':
                clade_flupredict.append(pango2flupredict['XBB.1.5'])
            elif focal_voc == 'CH.1':
                clade_flupredict.append(pango2flupredict['CH.1.1'])
            elif focal_voc == 'BA.5':
                clade_flupredict.append(pango2flupredict['BA.5.9'])

            flupredict_clades_here = [pango2flupredict[voc] for voc in voc_here]
            if 'BQ.1' in voc_here:
                flupredict_clades_here.append(pango2flupredict['BQ.1.1'])
            elif 'XBB' in voc_here:
                flupredict_clades_here.append(pango2flupredict['XBB.1.5'])
            elif 'CH.1' in voc_here:
                flupredict_clades_here.append(pango2flupredict['CH.1.1'])
            elif 'BA.5' in voc_here:
                flupredict_clades_here.append(pango2flupredict['BA.5.9'])

            def sum_counts(clades, t):
                N = 0.0
                for c in clades:
                    if c not in counts.keys():
                        continue
                    if t not in counts[c].keys():
                        continue
                    N += np.exp(counts[c][t])
                return N

            N0 = sum_counts(clade_flupredict,str(t0))
            Nt = sum_counts(clade_flupredict,str(tf))
            Z0 = sum_counts(flupredict_clades_here,str(t0))
            Zt = sum_counts(flupredict_clades_here,str(tf))
            N0_sample = np.random.binomial(Z0, x0, 1000) / Z0
            Nt_sample = np.random.binomial(Zt, xt, 1000) / Zt
            N0_sample = np.where(N0_sample == 0, np.ones(len(N0_sample)) * 1/Z0, N0_sample)
            Nt_sample = np.where(Nt_sample == 0, np.ones(len(Nt_sample)) * 1/Zt, Nt_sample)

            if Z0 < 250 or Zt < 250:
                print(country, focal_voc, t0)

            W_hat = np.mean(Nt_sample / N0_sample)
            W_hat_var = np.var(Nt_sample / N0_sample)

            df_line = df_here.loc[t0]
            df_lineR = df_R.loc[t0]

            F_ba2 = -gamma_vac * (df_line['C_BOOST_BA.2'] + df_line['C_BIVALENT_BA.2']) - gamma_inf * (df_lineR['C_RECOV_BA1_BA.2'] + df_lineR['C_RECOV_BA2_BA.2'] + df_lineR['C_RECOV_BA45_BA.2'])
            F_ba5 = -gamma_vac * (df_line['C_BOOST_BA.4/5'] + df_line['C_BIVALENT_BA.4/5']) - gamma_inf * (df_lineR['C_RECOV_BA1_BA.4/5'] + df_lineR['C_RECOV_BA2_BA.4/5'] + df_lineR['C_RECOV_BA45_BA.4/5'] + df_lineR['C_RECOV_BQ1_BA.4/5'])
            F_bq1 = -gamma_vac * (df_line['C_BOOST_BQ.1'] + df_line['C_BIVALENT_BQ.1']) - gamma_inf * (df_lineR['C_RECOV_BA1_BQ.1'] + df_lineR['C_RECOV_BA2_BQ.1'] + df_lineR['C_RECOV_BA45_BQ.1']+ df_lineR['C_RECOV_BQ1_BQ.1'])
            F_ba46 = -gamma_vac * (df_line['C_BOOST_BA.4.6'] + df_line['C_BIVALENT_BA.4.6']) - gamma_inf * (df_lineR['C_RECOV_BA1_BA.4.6'] + df_lineR['C_RECOV_BA2_BA.4.6'] + df_lineR['C_RECOV_BA45_BA.4.6'])
            F_bf7 = -gamma_vac * (df_line['C_BOOST_BF.7'] + df_line['C_BIVALENT_BF.7']) - gamma_inf * (df_lineR['C_RECOV_BA1_BF.7'] + df_lineR['C_RECOV_BA2_BF.7'] + df_lineR['C_RECOV_BA45_BF.7'])
            F_xbb = -gamma_vac * (df_line['C_BOOST_XBB'] + df_line['C_BIVALENT_XBB']) - gamma_inf * (df_lineR['C_RECOV_BA1_XBB'] + df_lineR['C_RECOV_BA2_XBB'] + df_lineR['C_RECOV_BA45_XBB']+ df_lineR['C_RECOV_BQ1_XBB'])
            F_ch1 = -gamma_vac * (df_line['C_BOOST_CH.1'] + df_line['C_BIVALENT_CH.1']) - gamma_inf * (df_lineR['C_RECOV_BA1_CH.1'] + df_lineR['C_RECOV_BA2_CH.1'] + df_lineR['C_RECOV_BA45_CH.1'] + df_lineR['C_RECOV_BQ1_CH.1'])
            F_bn1 = -gamma_vac * (df_line['C_BOOST_BN.1'] + df_line['C_BIVALENT_BA.2.75.2']) - gamma_inf * (df_lineR['C_RECOV_BA1_BN.1'] + df_lineR['C_RECOV_BA2_BN.1'] + df_lineR['C_RECOV_BA45_BN.1'])
            # F_bm11 = -gamma_vac * (df_line['C_BOOST_BM.1.1'] + df_line['C_BIVALENT_BA.2.75.2']) - gamma_inf * (df_lineR['C_RECOV_BA1_BM.1.1'] + df_lineR['C_RECOV_BA2_BM.1.1'] + df_lineR['C_RECOV_BA45_BM.1.1'])

            #sample for initial conditions
            vocs = ['BA.2','BA.5','BQ.1','BA.4.6','BF.7','XBB','CH.1','BN.1']
            N0 = []
            for voc in vocs:
                flupredict_clade = [pango2flupredict[voc]]
                if voc == 'BQ.1':
                    flupredict_clade.append(pango2flupredict['BQ.1.1'])
                elif voc == 'XBB' :
                    flupredict_clade.append(pango2flupredict['XBB.1.5'])
                elif voc == 'CH.1':
                    flupredict_clade.append(pango2flupredict['CH.1.1'])
                elif voc == 'BA.5':
                    flupredict_clade.append(pango2flupredict['BA.5.9'])
                    flupredict_clade.append(pango2flupredict['BA.4'])
                N = int(sum_counts(flupredict_clade, str(t0)))
                N0.append(N)

            X_samples = np.random.multinomial(np.sum(N0), N0 / np.sum(N0), 1000) / np.sum(N0)
            W_list = []
            for X in X_samples:
                F = np.array([F_ba2, F_ba5,F_bq1,F_ba46,F_bf7,F_xbb,F_ch1,F_bn1])
                F_av = np.dot(X,np.array([F_ba2, F_ba5,F_bq1,F_ba46,F_bf7,F_xbb,F_ch1,F_bn1]))
                Fitness = list(np.array([F_ba2, F_ba5,F_bq1,F_ba46,F_bf7,F_xbb,F_ch1,F_bn1]) - F_av)
                F_dict = {}
                for i, voc in enumerate(vocs):
                    F_dict[voc] = Fitness[i]
                F = np.array([F_dict[voc] for voc in voc_here])
                X_voc_here = []
                for voc in voc_here:
                    voc_index = vocs.index(voc)
                    X_voc_here.append(X[voc_index])
                X_voc_here = X_voc_here / np.sum(X_voc_here)
                Xt = X_voc_here * np.exp(F * dt)
                Xt = Xt / np.sum(Xt)
                xt_pred = Xt[voc_here.index(focal_voc)]
                W_list.append(np.mean(xt_pred / N0_sample))

            W = np.mean(W_list)
            W_var = np.var(W_list)

            lines.append([country, dt, t0, tf,t0_index+1, focal_voc, x0, xt, xt_pred, W,W_var, W_hat,W_hat_var])

df_W = pd.DataFrame(lines, columns=['country','time_step','t0','tf','t0_index','voc','x0','xt','xt_pred','W','W_var','W_hat','W_hat_var'])
#==============================================================================
#Plot W-What figure
#==============================================================================
voc2color={ "1": "#004E98" ,
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

R1 = ss.linregress(df_W.W, df_W.W_hat)
ratio=1
ls=10
fs=10
ms=5
plt.figure(figsize=(14,5))
ax = plt.subplot(111)
dt = 60
for voc in vocs:
    if voc == 'CH.1':
        color = voc2color[pango2flupredict['CH.1.1']]
    else:
        color = voc2color[pango2flupredict[voc]]
    plt.errorbar(df_W.loc[(df_W.voc == voc), 'W'], df_W.loc[(df_W.voc == voc), 'W_hat'],xerr=np.sqrt(df_W.loc[(df_W.voc == voc),'W_var']),yerr=np.sqrt(df_W.loc[(df_W.voc == voc),'W_hat_var']),fmt='o',color=color,alpha=0.8, markersize=5)
plt.xlabel("Frequency change, $W$")
plt.ylabel("Predicted frequency change, $\\hat{W}$")
plt.xscale('log')
plt.yscale('log')
plt.plot(np.linspace(0.001,150),np.linspace(0.001,150),'k:',linewidth=0.8)
plt.xlim([0.005,150])
plt.ylim([0.005,150])
plt.axhline(1,color = 'k',alpha=0.5)
plt.axvline(1,color = 'k',alpha=0.5)
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
legend_elements = []
for voc in vocs:
    if voc == 'CH.1':
        color = voc2color[pango2flupredict['CH.1.1']]
    else:
        color = voc2color[pango2flupredict[voc]]
    legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=color,linestyle='',label=voc, linewidth=2.0))
plt.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1.05,0.5),prop={'size':ls})
plt.subplots_adjust(right= 0.85)
plt.savefig("Fig4_frequency_change.pdf")
plt.close()

print(f"{len(df_W.loc[(df_W.W > 1.0) & (df_W.W_hat > 1.0)])} out of {len(df_W.loc[(df_W.W > 1.0)])}  that show growth")
print(f"{len(df_W.loc[(df_W.W < 1.0) & (df_W.W_hat < 1.0)])} out of {len(df_W.loc[(df_W.W < 1.0)])}  that show decline")
