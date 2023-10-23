import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import time
from matplotlib.lines import Line2D
import sys
from util.time import Time

sys.path.insert(0, "..")

pd.options.mode.chained_assignment = None  # default='warn'

WHOlabels = {'1C.2A.3A.4B': 'BETA',
             '1C.2A.3A.4A': 'EPSILON',
             '1C.2A.3A.4C': 'IOTA',
             '1C.2A.3I': 'MU',
             '1C.2B.3D': 'ALPHA',
             '1C.2B.3J': 'OMICRON',
             '1C.2B.3J.4D': 'BA.1',
             '1C.2B.3J.4D.5A': 'BA.1.1',
             '1C.2B.3J.4E': 'BA.2',
             '1C.2B.3J.4E.5B': 'BA.2.12.1',
             '1C.2B.3J.4E.5L': 'BJ.1',
             '1C.2B.3J.4E.5C': 'BA.2.75',
             '1C.2B.3J.4E.5C.6A': 'BA.2.75.2',
             '1C.2B.3J.4E.5C.6E': 'BM.1.1',
             '1C.2B.3J.4E.5C.6F': 'BN.1',
             '1C.2B.3J.4F': 'BA.4',
             '1C.2B.3J.4F.5D': 'BA.4.6',
             '1C.2B.3J.4G': 'BA.5',
             '1C.2B.3J.4G.5K': 'BA.5.9',
             '1C.2B.3J.4G.5E': 'BF.7',
             '1C.2B.3J.4G.5F': 'BQ.1',
             '1C.2B.3J.4G.5F.6B': 'BQ.1.1',
             '1C.2D.3F': 'DELTA',
             '1C.2B.3G': 'GAMMA',
             '1C.2B.3J.4E.5N': 'XBB',
             '1C.2B.3J.4E.5N.6J': 'XBB.1.5',
             '1C.2B.3J.4E.5C.6I': 'CH.1',
             '1C.2B.3J.4E.5C.6I.7C': 'CH.1.1'}
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
df_R = df_R.loc[df_R.time > Time.dateToCoordinate("2022-01-01")]
countries = sorted(list(set(df.country)))

gamma_inf_update = pd.read_csv("../output/Update_gamma_inf.txt",'\t',index_col=False)


#================================================================================
#Compute time points for each variant
#================================================================================
emergence_lines = []
for focal_voc in ['BQ.1','BA.4.6','BF.7','XBB','CH.1','BN.1','BM.1.1','BA.2.75','BA.2.75.2','BJ.1']:
    line = []
    if len(df_R.loc[df_R[f'x_{focal_voc}'] > 0.001]) > 1:
        t_01 = int(df_R.loc[df_R[f'x_{focal_voc}'] > 0.001].iloc[0].time)
        x0 = df_R.loc[df_R[f'x_{focal_voc}'] > 0.001].iloc[0][f'x_{focal_voc}']
    else:
        t_01 = Time.dateToCoordinate("2025-01-01")

    if len(df_R.loc[df_R[f'x_{focal_voc}'] > 0.01]) > 1:
        t_1 = int(df_R.loc[df_R[f'x_{focal_voc}'] > 0.01].iloc[0].time)
        x0 = df_R.loc[df_R[f'x_{focal_voc}'] > 0.01].iloc[0][f'x_{focal_voc}']
    else:
        t_1 = Time.dateToCoordinate("2025-01-01")
    if len(df_R.loc[df_R[f'x_{focal_voc}'] > 0.05]) > 1:
        t_5 = int(df_R.loc[df_R[f'x_{focal_voc}'] > 0.05].iloc[0].time)
        x0 = df_R.loc[df_R[f'x_{focal_voc}'] > 0.05].iloc[0][f'x_{focal_voc}']
    else:
        t_5 = Time.dateToCoordinate("2025-01-01")
    line.append(t_01)
    line.append(str(Time.coordinateToDate(t_01)))
    line.append(t_1)
    line.append(str(Time.coordinateToDate(t_1)))
    line.append(t_5)
    line.append(str(Time.coordinateToDate(t_5)))
    emergence_lines.append(line)
emergence_lines = pd.DataFrame(emergence_lines,columns=['t_01','date_01','t_1','date_1','t_5','date_5'], index = ['BQ.1','BA.4.6','BF.7','XBB','CH.1','BN.1','BM.1.1','BA.2.75','BA.2.75.2','BJ.1'])

emergence_lines = emergence_lines.loc[emergence_lines['t_01'] < 45657]
emergence_lines = emergence_lines.loc[emergence_lines['t_1'] < 45657]
emergence_lines = emergence_lines.sort_values(by = 't_1',ascending=True)


#Emergence at >0.01% in averaged frequency. Calculate competing variants
x_min = 0.005
competing_vocs = []
for focal_voc in list(emergence_lines.index):
    time = emergence_lines.loc[focal_voc].t_1
    df_lineR = df_R.loc[time]
    vocs = ['BA.2','BA.5','BQ.1','BA.4.6','BF.7','XBB','CH.1','BN.1','BM.1.1']    
    voc_here = df_lineR[[f'x_{voc}' for voc in vocs]]
    voc_here = list(voc_here[voc_here>x_min].index)
    voc_here = [a[2:] for a in voc_here]
    competing_vocs.append("_".join(voc_here))
emergence_lines['competing_vocs'] = competing_vocs

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
#================================================================================
#Compute total number of counts
#================================================================================
vocs = ['BA.2','BA.5','BQ.1','BA.4.6','BF.7','XBB','CH.1','BN.1','BM.1.1']    
counts_average = defaultdict(lambda: defaultdict(lambda: 0))
Z_average = defaultdict(lambda: 0)
for country in countries:
    df_country = df.loc[df.country == country]
    df_country['x_BA.1'] = df_country['x_BA.1'] + df_country['x_BA.1.1']
    df_country['x_BA.2'] = df_country['x_BA.2'] + df_country['x_BA.2.12.1']
    df_country['x_BA.4/5'] = df_country['x_BA.4']+ df_country['x_BA.5'] + df_country['x_BA.5.9']
    df_country['x_BQ.1'] = df_country['x_BQ.1']+ df_country['x_BQ.1.1'] 
    df_country['x_XBB'] = df_country['x_XBB']+ df_country['x_XBB.1.5'] 
    df_country['x_CH.1'] = df_country['x_CH.1'] + df_country['x_CH.1.1']
    df_country['x_wt'] = 1 - df_country['x_ALPHA'] - df_country['x_DELTA'] - df_country['x_BETA'] - df_country['x_EPSILON'] - df_country['x_IOTA'] - df_country['x_MU'] - df_country['x_OMICRON'] - df_country['x_GAMMA'] - df_country['x_BA.1']  - df_country['x_BA.2']
    df_here = df_country.loc[df_country.time > Time.dateToCoordinate("2022-04-01")]
    df_here.index = df_here.time

    with open("../DATA/2023_04_01/multiplicities_" + country.upper() + ".json",'r') as f:
        counts = json.load(f)
    with open("../DATA/2023_04_01/multiplicities_Z_" + country.upper() + ".json",'r') as f:
        Zdata = json.load(f)

    def sum_counts(clades, t):
        N = 0.0
        for c in clades:
            if c not in counts.keys():
                continue
            if t not in counts[c].keys():
                continue
            N += np.exp(counts[c][t])
        return N

    for t in list(df_here.index):
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

            N = int(sum_counts(flupredict_clade, str(t)))
            counts_average[voc][str(t)] += N
            Z_average[str(t)] += N

emergence_lines['N0'] = [counts_average[voc][str(t)] for voc, t in zip(emergence_lines.index, emergence_lines.t_1)]
emergence_lines['Z0'] = [Z_average[str(t)] for t in emergence_lines.t_1]


df_av_freq = pd.read_csv("../output/Average_Frequencies.txt",'\t',index_col=0)
df_av_freq.columns = [f'x_{voc}' for voc in df_av_freq.columns]
df_av_freq['time'] = df_av_freq.index
df_av_freq = df_av_freq.rename(columns={'x_BA.4/5':'x_BA.5'})
gamma_vac = 0.27859
#================================================================================
#Plot reduced frequency trajectories on averaged trajectories
#================================================================================
ratio = 1/1.69
fs = 10
ls=10
t_future = 200
emergent_vocs = ['BF.7','BQ.1','XBB','CH.1']
vocs = ['BA.2','BA.5','BQ.1','BA.4.6','BF.7','XBB','CH.1','BN.1','BM.1.1'] 
plt.figure(figsize=(16,6))
for emergent_index, focal_voc in enumerate(emergent_vocs):
    print(f"Focal voc {focal_voc}")
    t0 = emergence_lines.loc[focal_voc].t_1

    #Get gamma_vac, gamma_inf 
    if t0 < 44689 + 15:#First update of gamma_inf at t = 44704 (24-05-2022)
        gamma_inf = 0.65718
    else:
        gamma_inf = gamma_inf_update.loc[gamma_inf_update.time_cut_off == t0 - 15].iloc[0].gamma_inf_new


    voc_here = emergence_lines.loc[focal_voc]['competing_vocs'].split("_")
    ax = plt.subplot(2,2,emergent_index+1)
    plt.title(focal_voc)
    df_lineR = df_R.loc[t0]
    F_ba2 = -gamma_vac * (df_lineR['C_BOOST_BA.2'] + df_lineR['C_BIVALENT_BA.2']) - gamma_inf * (df_lineR['C_RECOV_BA1_BA.2'] + df_lineR['C_RECOV_BA2_BA.2'] + df_lineR['C_RECOV_BA45_BA.2'])
    F_ba5 = -gamma_vac * (df_lineR['C_BOOST_BA.4/5'] + df_lineR['C_BIVALENT_BA.4/5']) - gamma_inf * (df_lineR['C_RECOV_BA1_BA.4/5'] + df_lineR['C_RECOV_BA2_BA.4/5'] + df_lineR['C_RECOV_BA45_BA.4/5'] + df_lineR['C_RECOV_BQ1_BA.4/5'])
    F_bq1 = -gamma_vac * (df_lineR['C_BOOST_BQ.1'] + df_lineR['C_BIVALENT_BQ.1']) - gamma_inf * (df_lineR['C_RECOV_BA1_BQ.1'] + df_lineR['C_RECOV_BA2_BQ.1'] + df_lineR['C_RECOV_BA45_BQ.1']+ df_lineR['C_RECOV_BQ1_BQ.1'])
    F_ba46 = -gamma_vac * (df_lineR['C_BOOST_BA.4.6'] + df_lineR['C_BIVALENT_BA.4.6']) - gamma_inf * (df_lineR['C_RECOV_BA1_BA.4.6'] + df_lineR['C_RECOV_BA2_BA.4.6'] + df_lineR['C_RECOV_BA45_BA.4.6'])
    F_bf7 = -gamma_vac * (df_lineR['C_BOOST_BF.7'] + df_lineR['C_BIVALENT_BF.7']) - gamma_inf * (df_lineR['C_RECOV_BA1_BF.7'] + df_lineR['C_RECOV_BA2_BF.7'] + df_lineR['C_RECOV_BA45_BF.7'])
    F_xbb = -gamma_vac * (df_lineR['C_BOOST_XBB'] + df_lineR['C_BIVALENT_XBB']) - gamma_inf * (df_lineR['C_RECOV_BA1_XBB'] + df_lineR['C_RECOV_BA2_XBB'] + df_lineR['C_RECOV_BA45_XBB']+ df_lineR['C_RECOV_BQ1_XBB'])
    F_ch1 = -gamma_vac * (df_lineR['C_BOOST_CH.1'] + df_lineR['C_BIVALENT_CH.1']) - gamma_inf * (df_lineR['C_RECOV_BA1_CH.1'] + df_lineR['C_RECOV_BA2_CH.1'] + df_lineR['C_RECOV_BA45_CH.1'] + df_lineR['C_RECOV_BQ1_CH.1'])
    F_bn1 = -gamma_vac * (df_lineR['C_BOOST_BN.1'] + df_lineR['C_BIVALENT_BA.2.75.2']) - gamma_inf * (df_lineR['C_RECOV_BA1_BN.1'] + df_lineR['C_RECOV_BA2_BN.1'] + df_lineR['C_RECOV_BA45_BN.1']+ df_lineR['C_RECOV_BQ1_CH.1'])
    F_bm11 = -gamma_vac * (df_lineR['C_BOOST_BM.1.1'] + df_lineR['C_BIVALENT_BA.2.75.2']) - gamma_inf * (df_lineR['C_RECOV_BA1_BM.1.1'] + df_lineR['C_RECOV_BA2_BM.1.1'] + df_lineR['C_RECOV_BA45_BM.1.1']+ df_lineR['C_RECOV_BQ1_CH.1'])

    X0 = [df_av_freq.loc[df_av_freq.time == t0, f'x_{voc}'].iloc[0] for voc in voc_here]
    X0 = np.array(X0 / np.sum(X0))
    Xt = df_av_freq.loc[df_av_freq.time > t0, [f'x_{voc}' for voc in voc_here]]
    for t in Xt.index:
        Xt.loc[t] = np.array(Xt.loc[t]) / np.sum(Xt.loc[t])
    t_range = np.arange(t0, min([t0+t_future]))

    #sample for initial conditions
    N0 = [counts_average[voc][str(t0)] for voc in voc_here]
    X_samples = np.random.multinomial(np.sum(N0), N0 / np.sum(N0), 1000) / np.sum(N0)
    Fitness = np.array([F_ba2, F_ba5,F_bq1,F_ba46,F_bf7,F_xbb,F_ch1,F_bn1, F_bm11])
    F_dict = {}
    for i, voc in enumerate(vocs):
        F_dict[voc] = Fitness[i]
    
    F = np.array([F_dict[voc] for voc in voc_here])
    F_av = np.dot(X0,F)
    F = F - F_av
    Xt_pred = []
    for t in t_range:
        Xt_pred.append((X0 * np.exp(F * (t-t0))))
    Xt_pred = np.array(Xt_pred)
    for i in range(len(Xt_pred)):
        Xt_pred[i] = Xt_pred[i] / Xt_pred[i].sum()
    Xt_total = Xt_pred
    Xt_total = pd.DataFrame(Xt_total, columns = voc_here, index = t_range)

    bold_voc = []
    for voc_index, voc in enumerate(voc_here):
        if X0[voc_index] < 0.5 and max(Xt_total[voc]) > 0.5:
            bold_voc.append(voc)

    voc_bold = []
    for voc in voc_here:
        if np.sum([voc_dummy == voc for voc_dummy in bold_voc]) > 0: #In 80% of the prediction, reaches >50% within 180 days.
            voc_bold.append(voc)
        print(f"{voc} reaches >50% in {np.sum([voc_dummy == voc for voc_dummy in bold_voc])} cases")

    for x_voc in Xt.columns:
        voc = x_voc.split("_")[1]
        if voc == 'CH.1':
            color = voc2color[pango2flupredict['CH.1.1']]
        else:
            color = voc2color[pango2flupredict[voc]]
        if voc in voc_bold:
            plt.plot(Xt.index, Xt[x_voc], '-', color=color,linewidth=3., alpha=1.)
        else:
            plt.plot(Xt.index, Xt[x_voc], '-', color=color,linewidth=1.5, alpha=1.)

    for voc in Xt_total.columns:
        if voc == 'CH.1':
            color = voc2color[pango2flupredict['CH.1.1']]
        else:
            color = voc2color[pango2flupredict[voc]]
        if voc in voc_bold:
            plt.plot(Xt_total.index, Xt_total[voc], '--', color=color,linewidth=3., alpha=1.)
        else:
            plt.plot(Xt_total.index, Xt_total[voc], '--', color=color,linewidth=1.5, alpha=1.)

    # plt.fill_between(np.linspace(t0,t0+200), np.ones(50) * -0.05, np.ones(50) * 0.5, color='grey',alpha=0.2, linewidth = 0.0)
    plt.fill_between(np.linspace(t0,t0+200), np.ones(50) * 0.005, np.ones(50) * 0.5, color='grey',alpha=0.2, linewidth = 0.0)
    plt.plot(np.linspace(t0, t0+200), np.ones(50) * 0.5, linewidth=2., color='grey')

    xtick_labels = ['2022-07-01','2022-09-01','2022-11-01','2023-01-01','2023-03-01','2023-05-01']
    xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
    xtick_labels = ['Jul. $\'$22','Sep. $\'$22','Nov. $\'$22','Jan. $\'$23','Mar. $\'$23','May. $\'$23']
    plt.xticks(xtick_pos,xtick_labels,rotation=0,ha='center',fontsize=fs)
    plt.xlim([t0, t0+200])
    plt.ylim([-0.05,1.05])
    plt.ylabel("Reduced frequency, $x_i(t)$",fontsize=fs)
    plt.tick_params(direction='in',labelsize=ls)

legend_elements = []
for voc in vocs:
    if voc == 'CH.1':
        color = voc2color[pango2flupredict['CH.1.1']]
    else:
        color = voc2color[pango2flupredict[voc]]
    legend_elements.append(Line2D([],[],color=color,linestyle='-',label=voc, linewidth=2.0))
plt.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1.05,0.5),prop={'size':ls})

plt.subplots_adjust(hspace=0.4,right=0.85)
plt.savefig("Fig4_trajectories_Y.pdf")
plt.close()
