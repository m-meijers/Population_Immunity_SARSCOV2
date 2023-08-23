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


dT = defaultdict(lambda:defaultdict(lambda:1.0)) #Titer drop matrix in fold dT[RHO][ALPHA]

dT['VAC']['ALPHA'] = 1.8
dT['VAC']['DELTA'] = 3.2
dT['VAC']['OMICRON'] = 47
dT['VAC']['WT'] = 1.
dT['BOOST']['WT']=1.
dT['BOOST']['ALPHA']=1.8
dT['BOOST']['DELTA']=2.8
dT['BOOST']['OMICRON']=6.7
dT['BOOST']['BA.1']=6.7
dT['BOOST']['BA.2']=5.2
dT['BOOST']['BA.4/5']=10.3
dT['BOOST']['BA.2.75']=7.2
dT['BOOST']['BA.2.75.2']=30.4
dT['BOOST']['BA.4.6']=12.3
dT['BOOST']['BA.5.9']=13.6
dT['BOOST']['BQ.1']=25.5
dT['BOOST']['BQ.1.1']=46
dT['BOOST']['BJ.1']=15.9
dT['BOOST']['XBB']=41
dT['BOOST']['XBB.1.5']=42
dT['BOOST']['BM.1.1']=29.6
dT['BOOST']['BF.7']=14.5
dT['BOOST']['BN.1']=24
dT['BOOST']['CH.1']=74
dT['BIVALENT']['BA.4/5']=1.
dT['BIVALENT']['BA.2']=0.6
dT['BIVALENT']['BA.2.75.2'] = 4
dT['BIVALENT']['BA.4.6'] = 1.3
dT['BIVALENT']['BQ.1'] = 3
dT['BIVALENT']['BQ.1.1'] = 5.7
dT['BIVALENT']['XBB'] = 9.4
dT['BIVALENT']['XBB.1.5'] = 8.5
dT['BIVALENT']['BF.7'] = 1.2
dT['BIVALENT']['CH.1'] = 16.6

dT['ALPHA']['WT'] = 1.8
dT['ALPHA']['ALPHA'] = 1
dT['ALPHA']['DELTA'] = 2.8
dT['ALPHA']['OMICRON']=33

dT['DELTA']['WT'] = 3.2
dT['DELTA']['ALPHA'] = 3.5
dT['DELTA']['DELTA'] = 1.
dT['DELTA']['OMICRON'] = 27

dT['OMICRON']['WT'] = 47
dT['OMICRON']['ALPHA'] = 33
dT['OMICRON']['DELTA'] = 27
dT['OMICRON']['BA.1'] = 1.

dT['BA.1']['ALPHA'] = 33
dT['BA.1']['DELTA'] = 27
dT['BA.1']['BA.1'] = 1.
dT['BA.1']['BA.2'] = 2.4
dT['BA.1']['BA.4/5'] = 5.9
dT['BA.1']['BA.4.6'] = 11.8 #ASSUME 2-fold further than BA.4/5
dT['BA.1']['BA.2.75'] = 4.3
dT['BA.1']['BA.2.75.2'] = 25
dT['BA.1']['BQ.1'] = 26.6
dT['BA.1']['BQ.1.1'] = 31.6
dT['BA.1']['BJ.1'] = 9.8
dT['BA.1']['XBB'] = 37
dT['BA.1']['XBB.1.5'] = 37
dT['BA.1']['BM.1.1'] = 25.8
dT['BA.1']['BF.7'] = 12.2
dT['BA.1']['BN.1'] = 10.1
dT['BA.1']['CH.1'] = 34

dT['BA.2']['BA.1'] = 3.6
dT['BA.2']['BA.2'] = 1.
dT['BA.2']['BA.4/5'] = 2.6
dT['BA.2']['BA.4.6'] = 5.2 #ASSUME 2-fold further than BA.4/5
dT['BA.2']['BA.2.75'] = 3.2
dT['BA.2']['BA.2.75.2'] = 19.9
dT['BA.2']['BQ.1'] = 9.6
dT['BA.2']['BQ.1.1'] = 12.4
dT['BA.2']['BJ.1'] = 6.1
dT['BA.2']['XBB'] = 30.6
dT['BA.2']['XBB.1.5'] = 35.6
dT['BA.2']['BM.1.1'] = 19.3
dT['BA.2']['BF.7'] = 7.2
dT['BA.2']['BN.1'] = 12
dT['BA.2']['CH.1'] = 29

dT['BA.4/5']['BA.1'] = 2.4
dT['BA.4/5']['BA.2'] = 0.6
dT['BA.4/5']['BA.4/5'] = 1.
dT['BA.4/5']['BA.2.75'] = 1.8
dT['BA.4/5']['BA.2.75.2'] = 3.4
dT['BA.4/5']['BA.4.6'] = 1.7
dT['BA.4/5']['BA.5.9'] = 2.45
dT['BA.4/5']['BQ.1'] = 3.6
dT['BA.4/5']['BQ.1.1'] = 4.9
dT['BA.4/5']['BJ.1'] = 4
dT['BA.4/5']['XBB'] = 11.6
dT['BA.4/5']['XBB.1.5'] = 11.8
dT['BA.4/5']['BM.1.1'] = 10.6
dT['BA.4/5']['BF.7'] = 2.4
dT['BA.4/5']['BN.1'] = 9.2
dT['BA.4/5']['CH.1'] = 14.9

dT['BQ.1']['BQ.1'] = 1.
dT['BQ.1']['XBB'] = 3.7 
dT['BQ.1']['CH.1'] = 4.7
dT['BQ.1']['BA.4/5'] = 0.3 #end of self-coupling?

def sigmoid_func(t,mean=np.log10(0.2 * 94),s=3.0):
	val = 1 / (1 + np.exp(s * (t - mean)))
	return val

def sigmoid_func_inv(c, mean= np.log10(0.2*94),s=3.0):
	val = 1/s * np.log(c/(1-c)) + mean
	return val

df_R = pd.read_csv("../output/R_average.txt",'\t',index_col=False)
df_R['x_BA.1'] = df_R['x_BA.1'] + df_R['x_BA.1.1']
df_R['x_BA.4/5'] = df_R['x_BA.4']+ df_R['x_BA.5'] + df_R['x_BA.5.9']
df_R['x_BQ.1'] = df_R['x_BQ.1']+ df_R['x_BQ.1.1'] 
df_R['x_XBB'] = df_R['x_XBB']+ df_R['x_XBB.1.5'] 
df_R['x_CH.1'] = df_R['x_CH.1'] + df_R['x_CH.1.1']

df_R.pop("x_BA.1.1")
df_R.pop("x_BA.4")
df_R.pop("x_BA.5")
df_R.pop("x_BQ.1.1")
df_R.pop("x_XBB.1.5")
df_R.pop("x_CH.1.1")

gamma_inf_update = pd.read_csv("../output/Update_gamma_inf.txt",'\t',index_col=False)

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



inch2cm = 2.54
ratio = 1/1.62
mpl.rcParams['axes.linewidth'] = 0.3 #set the value globally
ms2=20
lw=0.75
elw=0.75
mlw=1.5
lp=0.7
lp1=0.2
rot=0
fs = 7
ls = 7

ratio=1/1.618

integrate.quad(sigmoid_func,np.log10(223) - 2, np.log10(223))[0]/2
#==========================================Vaccination

def C_variant(dT, t, T0=223):
	return sigmoid_func(np.log10(T0) - np.log10(np.exp(t/tau_decay)) - np.log10(dT))

T_decay = np.log10(np.exp(1))
R = [0.1,0.2,0.3,0.4,0.5]
dT_list = np.arange(0,6.5,0.1)
f_vac = []
for T in dT_list:
	# f_vac.append(integrate.quad(sigmoid_func,np.log10(223) - np.log10(2**T) - T_decay, np.log10(223) - np.log10(2**T))[0]/T_decay)
	f_vac.append(sigmoid_func(np.log10(223) - np.log10(2**T)))
f_recov = []
for T in dT_list:
	# f_recov.append(integrate.quad(sigmoid_func,np.log10(94) - np.log10(2**T) - T_decay, np.log10(94) - np.log10(2**T))[0]/T_decay)
	f_recov.append(sigmoid_func(np.log10(94) - np.log10(2**T)))
f_bst = []
for T in dT_list:
	# f_bst.append(integrate.quad(sigmoid_func,np.log10(223 * 4) - np.log10(2**T) - T_decay, np.log10(223 * 4) - np.log10(2**T))[0]/T_decay)
	f_bst.append(sigmoid_func(np.log10(223*4) - np.log10(2**T)))
f_vac=  np.array(f_vac)
f_recov = np.array(f_recov)
f_bst = np.array(f_bst)

vocs = ['wt','ALPHA','DELTA','BA.2','BA.4/5','BQ.1','XBB',]
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

df_s_av = pd.read_csv("../output/selection_potentials_average.txt",'\t',index_col=False)
ratio = 0.3


fig = plt.figure(figsize=(18/inch2cm,8/inch2cm))
gs0 = gridspec.GridSpec(1,1,figure=fig,hspace=0.35)
gs01 = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=gs0[0],hspace=0.2,wspace=0.1)

#========================================================================================================================
#Antigenic selection window plots
#========================================================================================================================
ax = fig.add_subplot(gs01[0,0])
era2color = {'wt':flupredict2color['1'],'ALPHA':flupredict2color[pango2flupredict['ALPHA']],'DELTA':flupredict2color[pango2flupredict['DELTA']],'BA.1':flupredict2color[pango2flupredict['BA.1']], 'BA.2':flupredict2color[pango2flupredict['BA.2']],
'BA.4/5':flupredict2color[pango2flupredict['BA.5']],'BQ.1':flupredict2color[pango2flupredict['BQ.1']],'XBB':flupredict2color[pango2flupredict['XBB']]}
for voc in ['ALPHA','DELTA','BA.1','BA.2','BA.4/5','BQ.1','XBB']:
    plt.barh(0.11,width=max(voc2era[voc]) - min(voc2era[voc]),height=0.01, left= min(voc2era[voc]), color = era2color[voc])
plt.axhline(0,color = 'k',alpha=0.5)

# #========================================================================================================================
# #BA.2 - BA.5 shift
# #========================================================================================================================
gamma_vac = 0.27859
gamma_inf = 0.65718
t0 = min(voc2era['BA.1'])
tf = max(voc2era['BA.2'])
df_here = df_s_av.loc[(df_s_av.time > t0) & (df_s_av.time < tf)]
df_here['gamma_inf'] = np.ones(len(df_here)) * 0.65718
for t in np.arange(44689+15, tf+1):
    df_here.loc[df_here.time == t,'gamma_inf'] = gamma_inf_update.loc[gamma_inf_update.time_cut_off == t - 15].iloc[0].gamma_inf_new

s_vac = gamma_vac * np.array(df_here.sw_ba2_bst)
s_ba1 = df_here['gamma_inf'] * np.array(df_here.sw_ba2_ba1)
s_ba2 = df_here['gamma_inf'] * np.array(df_here.sw_ba2_ba2)
plt.fill_between(df_here.time, s_vac+s_ba1, s_vac + s_ba1 + s_ba2, color= flupredict2color[pango2flupredict['BA.2']], alpha=0.5,linewidth=0.0)
plt.fill_between(df_here.time, s_vac, s_vac + s_ba1, color= flupredict2color[pango2flupredict['BA.1']], alpha=0.5,linewidth=0.0)
plt.fill_between(df_here.time, s_vac, color= '#0077B6', alpha=0.5,linewidth=0.0)
plt.plot(df_here.time, s_vac + s_ba1 + s_ba2, color = flupredict2color[pango2flupredict['BA.2']])
plt.plot(df_here.time, s_vac + s_ba1, color = flupredict2color[pango2flupredict['BA.1']])
plt.plot(df_here.time, s_vac , color = '#0077B6')
plt.ylabel("Antigenic Selection, $s_{\\rm ag}$",fontsize=fs)

#selection for BA.5
width = 14
t_emerge = df_R.loc[df_R['x_BA.4/5']>0.05].iloc[0].time
line = df_R.loc[df_R.time == t_emerge]
gamma_inf = 0.65718
s_boost = gamma_vac * (line['C_BOOST_BA.2'] - line['C_BOOST_BA.4/5'])
s_ba1 = gamma_inf * (line['C_RECOV_BA1_BA.2'] - line['C_RECOV_BA1_BA.4/5'])
s_ba2 = gamma_inf * (line['C_RECOV_BA2_BA.2'] - line['C_RECOV_BA2_BA.4/5'])
s_ba45 = gamma_inf * (line['C_RECOV_BA45_BA.2'] - line['C_RECOV_BA45_BA.4/5'])
plt.bar([t_emerge], s_boost, width=width, bottom=0.0,color='#0077B6')
plt.bar([t_emerge], s_ba1, width=width, bottom=s_boost,color=flupredict2color[pango2flupredict['BA.1']])
plt.bar([t_emerge], s_ba2, width=width, bottom=s_boost + s_ba1,color=flupredict2color[pango2flupredict['BA.2']])

# #========================================================================================================================
# #BA.5 - BQ.1
# #========================================================================================================================
gamma_vac = 0.27859
gamma_inf = 1.5924
t0 = min(voc2era['BA.4/5'])
tf = max(voc2era['BA.4/5'])
df_here = df_s_av.loc[(df_s_av.time > t0) & (df_s_av.time < tf)]
df_here['gamma_inf'] = np.ones(len(df_here)) * 0.65718
for t in np.arange(t0, tf+1):
    df_here.loc[df_here.time == t,'gamma_inf'] = gamma_inf_update.loc[gamma_inf_update.time_cut_off == t - 15].iloc[0].gamma_inf_new
s_bst = gamma_vac * np.array(df_here.sw_ba5_bst)
s_biv = gamma_vac * np.array(df_here.sw_ba5_biv)
s_ba5 = df_here['gamma_inf'] * np.array(df_here.sw_ba5_ba5)
s_ba2 = df_here['gamma_inf'] * np.array(df_here.sw_ba5_ba2)
s_ba1 = df_here['gamma_inf'] * np.array(df_here.sw_ba5_ba1)

plt.fill_between(df_here.time, s_bst + s_biv + s_ba1 + s_ba2, s_bst + s_biv + s_ba5 + s_ba2 + s_ba1, color=flupredict2color[pango2flupredict['BA.5']],alpha=0.5, linewidth=0.0)
plt.fill_between(df_here.time, s_bst + s_biv + s_ba1, s_bst + s_biv + s_ba1 + s_ba2 , color=flupredict2color[pango2flupredict['BA.2']],alpha=0.5, linewidth=0.0)
plt.fill_between(df_here.time, s_bst + s_biv, s_bst + s_biv + s_ba1, color=flupredict2color[pango2flupredict['BA.1']],alpha=0.5, linewidth=0.0)
plt.fill_between(df_here.time, s_bst, s_bst + s_biv, color='#03045E',alpha=0.5, linewidth=0.0)
plt.fill_between(df_here.time, s_bst, color='#0077B6',alpha=0.5, linewidth=0.0)

plt.plot(df_here.time, s_bst + s_biv + s_ba5 + s_ba2 + s_ba1, color=flupredict2color[pango2flupredict['BA.5']])
plt.plot(df_here.time, s_bst + s_biv + s_ba1 + s_ba2 , color=flupredict2color[pango2flupredict['BA.2']])
plt.plot(df_here.time, s_bst + s_biv + s_ba1, color=flupredict2color[pango2flupredict['BA.1']])
plt.plot(df_here.time, s_bst + s_biv , color='#03045E')
plt.plot(df_here.time, s_bst , color='#0077B6')

#selection for BQ.1
width = 14
t_emerge = df_R.loc[df_R['x_BQ.1']>0.05].iloc[0].time
line = df_R.loc[df_R.time == t_emerge]
gamma_inf = gamma_inf_update.loc[gamma_inf_update.time_cut_off == line.time.iloc[0] - 15].iloc[0].gamma_inf_new
s_boost = gamma_vac * (line['C_BOOST_BA.4/5'] - line['C_BOOST_BQ.1'])
s_biv = gamma_vac * (line['C_BIVALENT_BA.4/5'] - line['C_BIVALENT_BQ.1'])
s_ba1 = gamma_inf * (line['C_RECOV_BA1_BA.4/5'] - line['C_RECOV_BA1_BQ.1'])
s_ba2 = gamma_inf * (line['C_RECOV_BA2_BA.4/5'] - line['C_RECOV_BA2_BQ.1'])
s_ba45 = gamma_inf * (line['C_RECOV_BA45_BA.4/5'] - line['C_RECOV_BA45_BQ.1'])
plt.bar([t_emerge], s_boost, width=width, bottom=0.0,color='#0077B6')
plt.bar([t_emerge], s_biv, width=width, bottom=s_boost + s_biv,color='#03045E')
plt.bar([t_emerge], s_ba1, width=width, bottom=s_boost + s_biv,color=flupredict2color[pango2flupredict['BA.1']])
plt.bar([t_emerge], s_ba2, width=width, bottom=s_boost + s_biv + s_ba1,color=flupredict2color[pango2flupredict['BA.2']])
plt.bar([t_emerge], s_ba45, width=width, bottom=s_boost + s_biv + s_ba1 + s_ba2,color=flupredict2color[pango2flupredict['BA.5']])
 
#selection for BF.7
t_emerge = df_R.loc[df_R['x_BF.7']>0.05].iloc[0].time
line = df_R.loc[df_R.time == t_emerge]
gamma_inf = gamma_inf_update.loc[gamma_inf_update.time_cut_off == line.time.iloc[0] - 15].iloc[0].gamma_inf_new
s_boost = gamma_vac * (line['C_BOOST_BA.4/5'] - line['C_BOOST_BF.7'])
s_biv = gamma_vac * (line['C_BIVALENT_BA.4/5'] - line['C_BIVALENT_BF.7'])
s_ba1 = gamma_inf * (line['C_RECOV_BA1_BA.4/5'] - line['C_RECOV_BA1_BF.7'])
s_ba2 = gamma_inf * (line['C_RECOV_BA2_BA.4/5'] - line['C_RECOV_BA2_BF.7'])
s_ba45 = gamma_inf * (line['C_RECOV_BA45_BA.4/5'] - line['C_RECOV_BA45_BF.7'])
plt.bar([t_emerge], s_boost, width=width, bottom=0.0,color='#0077B6')
plt.bar([t_emerge], s_biv, width=width, bottom=s_boost + s_biv,color='#03045E')
plt.bar([t_emerge], s_ba1, width=width, bottom=s_boost + s_biv,color=flupredict2color[pango2flupredict['BA.1']])
plt.bar([t_emerge], s_ba2, width=width, bottom=s_boost + s_biv + s_ba1,color=flupredict2color[pango2flupredict['BA.2']])
plt.bar([t_emerge], s_ba45, width=width, bottom=s_boost + s_biv + s_ba1 + s_ba2,color=flupredict2color[pango2flupredict['BA.5']])

#selection for BA.4.6
t_emerge = df_R.loc[df_R['x_BA.4.6']>0.01].iloc[0].time
line = df_R.loc[df_R.time == t_emerge]
gamma_inf = gamma_inf_update.loc[gamma_inf_update.time_cut_off == line.time.iloc[0] - 15].iloc[0].gamma_inf_new
s_boost = gamma_vac * (line['C_BOOST_BA.4/5'] - line['C_BOOST_BA.4.6'])
s_ba1 = gamma_inf * (line['C_RECOV_BA1_BA.4/5'] - line['C_RECOV_BA1_BA.4.6'])
s_ba2 = gamma_inf * (line['C_RECOV_BA2_BA.4/5'] - line['C_RECOV_BA2_BA.4.6'])
s_ba45 = gamma_inf * (line['C_RECOV_BA45_BA.4/5'] - line['C_RECOV_BA45_BA.4.6'])
plt.bar([t_emerge], s_boost, width=width, bottom=0.0,color='#0077B6')
plt.bar([t_emerge], s_ba1, width=width, bottom=s_boost ,color=flupredict2color[pango2flupredict['BA.1']])
plt.bar([t_emerge], s_ba2, width=width, bottom=s_boost  + s_ba1,color=flupredict2color[pango2flupredict['BA.2']])
plt.bar([t_emerge], s_ba45, width=width, bottom=s_boost  + s_ba1 + s_ba2,color=flupredict2color[pango2flupredict['BA.5']])


# #========================================================================================================================
# #BQ.1 - Emerging shift
# #========================================================================================================================
gamma_vac = 0.27859
t0 = min(voc2era['BQ.1'])
tf = max(voc2era['BQ.1'])
df_here = df_s_av.loc[(df_s_av.time > t0) & (df_s_av.time < tf)]
df_here['gamma_inf'] = np.ones(len(df_here)) * 0.65718
for t in np.arange(t0, tf+1):
    df_here.loc[df_here.time == t,'gamma_inf'] = gamma_inf_update.loc[gamma_inf_update.time_cut_off == t - 15].iloc[0].gamma_inf_new
s_bst = gamma_vac * np.array(df_here.sw_bq1_bst)
s_biv = gamma_vac * np.array(df_here.sw_bq1_biv)
s_ba5 = gamma_inf * np.array(df_here.sw_bq1_ba5)
s_bq1 = gamma_inf * np.array(df_here.sw_bq1_bq1)
s_ba2 = gamma_inf * np.array(df_here.sw_bq1_ba2)
plt.fill_between(df_here.time, s_bst + s_biv + s_ba5, s_bst + s_biv + s_ba5 + s_bq1, color=flupredict2color[pango2flupredict['BQ.1']],alpha=0.5, linewidth = 0.0)
plt.fill_between(df_here.time, s_bst + s_biv, s_bst + s_biv + s_ba5, color=flupredict2color[pango2flupredict['BA.5']],alpha=0.5, linewidth = 0.0)
plt.fill_between(df_here.time, s_bst, s_bst + s_biv, color='#03045E',alpha=0.5, linewidth = 0.0)
plt.fill_between(df_here.time, s_bst, color='#0077B6',alpha=0.5, linewidth = 0.0)
plt.plot(df_here.time, s_bst + s_biv + s_ba5 + s_bq1, color=flupredict2color[pango2flupredict['BQ.1']])
plt.plot(df_here.time, s_bst + s_biv + s_ba5, color=flupredict2color[pango2flupredict['BA.5']])
plt.plot(df_here.time, s_bst + s_biv , color='#03045E')
plt.plot(df_here.time, s_bst , color='#0077B6')

df_here.iloc[-1]['sw_bq1_ba5']
#selection for XBB
width = 14
t_emerge = df_R.loc[df_R['x_XBB']>0.05].iloc[0].time
line = df_R.loc[df_R.time == t_emerge]
gamma_inf = gamma_inf_update.loc[gamma_inf_update.time_cut_off == line.time.iloc[0] - 15].iloc[0].gamma_inf_new
s_boost = gamma_vac * (line['C_BOOST_BQ.1'] - line['C_BOOST_XBB'])
s_biv = gamma_vac * (line['C_BIVALENT_BQ.1'] - line['C_BIVALENT_XBB'])
s_ba1 = gamma_inf * (line['C_RECOV_BA1_BQ.1'] - line['C_RECOV_BA1_XBB'])
s_ba2 = gamma_inf * (line['C_RECOV_BA2_BQ.1'] - line['C_RECOV_BA2_XBB'])
s_ba45 = gamma_inf * (line['C_RECOV_BA45_BQ.1'] - line['C_RECOV_BA45_XBB'])
s_bq1 = gamma_inf * (line['C_RECOV_BQ1_BQ.1'] - line['C_RECOV_BQ1_XBB'])
plt.bar([t_emerge], s_boost, width=width, bottom=0.0,color='#0077B6')
plt.bar([t_emerge], s_biv, width=width, bottom=s_boost ,color='#03045E')
plt.bar([t_emerge], s_ba45, width=width, bottom=s_boost + s_biv,color=flupredict2color[pango2flupredict['BA.5']])
plt.bar([t_emerge], s_bq1, width=width, bottom=s_boost + s_biv + s_ba45,color=flupredict2color[pango2flupredict['BQ.1']])

#selection for CH.1
t_emerge = df_R.loc[df_R['x_CH.1']>0.05].iloc[0].time
line = df_R.loc[df_R.time == t_emerge]
gamma_inf = gamma_inf_update.loc[gamma_inf_update.time_cut_off == line.time.iloc[0] - 15].iloc[0].gamma_inf_new
s_boost = gamma_vac * (line['C_BOOST_BQ.1'] - line['C_BOOST_CH.1'])
s_biv = gamma_vac * (line['C_BIVALENT_BQ.1'] - line['C_BIVALENT_CH.1'])
s_ba1 = gamma_inf * (line['C_RECOV_BA1_BQ.1'] - line['C_RECOV_BA1_CH.1'])
s_ba2 = gamma_inf * (line['C_RECOV_BA2_BQ.1'] - line['C_RECOV_BA2_CH.1'])
s_ba45 = gamma_inf * (line['C_RECOV_BA45_BQ.1'] - line['C_RECOV_BA45_CH.1'])
s_bq1 = gamma_inf * (line['C_RECOV_BQ1_BQ.1'] - line['C_RECOV_BQ1_CH.1']) 
plt.bar([t_emerge], s_boost, width=width, bottom=0.0,color='#0077B6')
plt.bar([t_emerge], s_biv, width=width, bottom=s_boost ,color='#03045E')
plt.bar([t_emerge], s_ba45, width=width, bottom=s_boost + s_biv,color=flupredict2color[pango2flupredict['BA.5']])
plt.bar([t_emerge], s_bq1, width=width, bottom=s_boost + s_biv + s_ba45,color=flupredict2color[pango2flupredict['BQ.1']])

# #========================================================================================================================
# #XBB-emerging shift
# #========================================================================================================================
gamma_vac = 0.27859
gamma_inf = 1.7450
t0 = min(voc2era['XBB'])
tf = max(voc2era['XBB'])
df_here = df_s_av.loc[(df_s_av.time > t0) & (df_s_av.time < tf)]
df_here['gamma_inf'] = np.ones(len(df_here)) * 0.65718
for t in np.arange(t0, tf+1):
    if t - 15 < gamma_inf_update.iloc[-1].time_cut_off:
        df_here.loc[df_here.time == t,'gamma_inf'] = gamma_inf_update.loc[gamma_inf_update.time_cut_off == t - 15].iloc[0].gamma_inf_new
    else:
        df_here.loc[df_here.time == t,'gamma_inf'] = gamma_inf_update.iloc[-1].gamma_inf_new
s_bst = gamma_vac * np.array(df_here.sw_xbb_bst)
s_biv = gamma_vac * np.array(df_here.sw_xbb_biv)
s_ba5 = gamma_inf * np.array(df_here.sw_xbb_ba5)
s_bq1 = gamma_inf * np.array(df_here.sw_xbb_bq1)
s_xbb = gamma_inf * np.array(df_here.sw_xbb_xbb)
plt.fill_between(df_here.time,  s_biv  + s_bq1,  s_biv  + s_bq1 + s_xbb, color=flupredict2color[pango2flupredict['XBB']],alpha=0.5, linewidth = 0.0)
plt.fill_between(df_here.time,  s_biv ,  s_biv  + s_bq1, color=flupredict2color[pango2flupredict['BQ.1']],alpha=0.5, linewidth = 0.0)
plt.fill_between(df_here.time, s_biv, color='#03045E',alpha=0.5, linewidth = 0.0)
plt.plot(df_here.time, s_biv  + s_bq1 + s_xbb, color=flupredict2color[pango2flupredict['XBB']])
plt.plot(df_here.time, s_biv  + s_bq1, color=flupredict2color[pango2flupredict['BQ.1']])
plt.plot(df_here.time, s_biv , color='#03045E')


xtick_labels = ['2021-01-01','2021-05-01','2021-09-01','2022-01-01','2022-05-01','2022-09-01','2023-01-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Jan. $\'$21','May $\'$ 21','Sep. $\'$21','Jan. $\'$22','May $\'$22','Sep. $\'$22','Jan. $\'$23']
ax.set_xticks(xtick_pos,xtick_labels,rotation=rot,ha='center',fontsize=fs)
plt.xlim([min(voc2era['BA.1']), Time.dateToCoordinate("2023-03-30")-1])


plt.ylim([0,0.12])
ax.set_yticks([0.0,0.05,0.1],['0.0','0.05','0.10'],fontsize=fs)
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
plt.tick_params(direction='in',labelsize=ls)

ms=4
legend_elements = []
legend_elements.append(Line2D([],[],marker='',markersize=ms,color='#00B4D8',linestyle='-',label='Vaccination', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='',markersize=ms,color='#0077B6',linestyle='-',label='Booster', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='',markersize=ms,color='#03045E',linestyle='-',label='Bivalent Booster', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color['1'],linestyle='-',label='Wild type', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['ALPHA']],linestyle='-',label='Alpha', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['DELTA']],linestyle='-',label='Delta', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['OMICRON']],linestyle='-',label='Omicron', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['BA.1']],linestyle='-',label='Omicron BA.1', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['BA.2']],linestyle='-',label='Omicron BA.2', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['BA.5']],linestyle='-',label='Omicron BA.4/5', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['BA.4.6']],linestyle='-',label='Omicron BA.4.6', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['BF.7']],linestyle='-',label='Omicron BF.7', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['BQ.1']],linestyle='-',label='Omicron BQ.1', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['XBB']],linestyle='-',label='Omicron XBB', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['CH.1.1']],linestyle='-',label='Omicron CH.1', linewidth=2.0))
# legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=flupredict2color[pango2flupredict['BN.1']],linestyle='-',label='Omicron BN.1', linewidth=2.0))

plt.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1.05,0.5),prop={'size':ls})
plt.subplots_adjust(bottom=0.05,top=0.95,left=0.075,right=0.8)
	
plt.savefig("Fig5.pdf")
plt.close() 








