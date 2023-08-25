import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"..")
from util.time import Time
import json
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from copy import copy
import scipy.integrate as si
import scipy.stats as ss
import scipy.optimize as so
from collections import defaultdict
import copy
import glob
import sys
from time import time
import scipy.integrate as integrate

def sigmoid_func(t,mean,s):
    val = 1 / (1 + np.exp(-s * (t - mean)))
    return val

def clean_vac(times,vac_vec):
    vac_rep = []
    vac_times = []
    i = 0
    while np.isnan(vac_vec[i]):
        vac_rep.append(0.0)
        vac_times.append(times[i])
        i += 1 
    for i in range(len(vac_vec)):
        if np.isnan(vac_vec[i]):
            continue
        else:
            vac_rep.append(vac_vec[i])
            vac_times.append(times[i])
    v_func = interp1d(vac_times,vac_rep,fill_value = 'extrapolate')
    v_interp = v_func(times)
    return v_interp

def smooth(times,vec,dt):
    smooth = []
    times_smooth = []
    for i in range(dt,len(times)-dt):
        smooth.append(np.mean(vec[i-dt:i+dt]))
        times_smooth.append(times[i])
    return smooth, times_smooth


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


meta_df = pd.read_csv("DATA/clean_data.txt",sep='\t',index_col=False)   
countries = ['BELGIUM','CA','CANADA','FINLAND','FRANCE','GERMANY','ITALY','NETHERLANDS','NORWAY','NY','SPAIN','SWITZERLAND','USA','JAPAN']

country2immuno2time = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
country2immuno2time_bivalent = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
for country in countries:
    x_limit = 0.01
    with open("DATA/2023_04_01/freq_traj_" + country.upper() + ".json",'r') as f:
        freq_traj = json.load(f)
    with open("DATA/2023_04_01/multiplicities_" + country.upper() + ".json",'r') as f:
        counts = json.load(f)
    with open("DATA/2023_04_01/multiplicities_Z_" + country.upper() + ".json",'r') as f:
        Z = json.load(f)
    meta_country= meta_df.loc[list(meta_df['location'] == country[0] + country[1:].lower())]
    if len(country) <= 3:
        meta_country= meta_df.loc[list(meta_df['location'] == country)]
    if country == 'NORWAY':
        meta_country = meta_country.iloc[1:]
    meta_country.index = meta_country['FLAI_time']

    vac_full = np.array([meta_country.loc[t]['people_fully_vaccinated_per_hundred']/100. for t in meta_country.index])
    vac_full = np.where(vac_full == 0.0, np.ones(len(vac_full)) * float('NaN'), vac_full)
    vinterp = clean_vac(list(meta_country.index),vac_full)
    country2immuno2time[country]['VAC'] = defaultdict(lambda: 0.0, {list(meta_country.index)[i] : vinterp[i] for i in range(len(list(meta_country.index)))})
    booster = np.array([meta_country.loc[t]['total_boosters_per_hundred']/100. for t in meta_country.index])
    booster = np.where(booster == 0.0, np.ones(len(booster)) * float('NaN'), booster)
    if np.sum(np.isnan(booster)) == len(booster):
        boosterp = np.zeros(len(booster))
    else:
        boosterp = clean_vac(list(meta_country.index), booster)
    country2immuno2time[country]['BOOST'] = defaultdict(lambda: 0.0, {list(meta_country.index)[i]:boosterp[i] for i in range(len(list(meta_country.index)))})
    bivalent = np.array([meta_country.loc[t]['total_bivalent_boosters_per_hundred']/100. for t in meta_country.index])
    bivalent = np.where(bivalent == 0.0, np.ones(len(bivalent)) * float('NaN'), bivalent)
    if np.sum(np.isnan(bivalent)) == len(bivalent):
        bivalentp = np.zeros(len(bivalent))
    else:
        bivalentp = clean_vac(list(meta_country.index), bivalent)
    country2immuno2time[country]['BIVALENT'] = defaultdict(lambda: 0.0, {list(meta_country.index)[i]:bivalentp[i] for i in range(len(list(meta_country.index)))})

    #correct the cumulative vaccinated: NB: looks different for each time point
    for time_meas in  list(meta_country.index):
        # for t in list(country2immuno2time[country]['VAC'].keys()):
        for t in np.arange(min(list(country2immuno2time[country]['VAC'].keys())), time_meas+1):

            if country2immuno2time[country]['VAC'][t] < country2immuno2time[country]['BOOST'][time_meas]:
                country2immuno2time[country][time_meas][t] = 0.0
            else:
                country2immuno2time[country][time_meas][t] = country2immuno2time[country]['VAC'][t] - country2immuno2time[country]['BOOST'][time_meas]
    #correct the cumulative boosted: NB: looks different for each time point
    for time_meas in  list(meta_country.index):
        for t in np.arange(min(list(country2immuno2time[country]['BOOST'].keys())), time_meas+1):
            if country2immuno2time[country]['BOOST'][t] < country2immuno2time[country]['BIVALENT'][time_meas]:
                country2immuno2time_bivalent[country][time_meas][t] = 0.0
            else:
                country2immuno2time_bivalent[country][time_meas][t] = country2immuno2time[country]['BOOST'][t] - country2immuno2time[country]['BIVALENT'][time_meas]


    x_delta = []
    x_omi = []
    x_alpha = []
    x_ba1 = []
    x_ba2 = []
    x_ba45 = []
    x_bq1 = []
    x_xbb = []
    x_wt = []
    all_vocs = sorted(list(WHOlabels.values()))
    for t in list(meta_country.index):
        x_vocs = []
        for voc in all_vocs:
            if pango2flupredict[voc] not in freq_traj.keys():
                x_vocs.append(0.0)
                continue
            if str(t) in freq_traj[pango2flupredict[voc]].keys():
                x_vocs.append(freq_traj[pango2flupredict[voc]][str(t)])
            else:
                x_vocs.append(0.0)

        x_delta.append(x_vocs[all_vocs.index('DELTA')])
        x_omi.append(x_vocs[all_vocs.index('OMICRON')])
        x_alpha.append(x_vocs[all_vocs.index('ALPHA')])
        x_ba1.append(x_vocs[all_vocs.index('BA.1')] + x_vocs[all_vocs.index('BA.1.1')])
        x_ba2.append(x_vocs[all_vocs.index('BA.2')] + x_vocs[all_vocs.index('BA.2.12.1')])
        x_ba45.append(x_vocs[all_vocs.index('BA.4')] + x_vocs[all_vocs.index('BA.5')] + x_vocs[all_vocs.index('BA.5.9')])
        x_bq1.append(x_vocs[all_vocs.index('BQ.1')] + x_vocs[all_vocs.index('BQ.1.1')])
        x_xbb.append(x_vocs[all_vocs.index('XBB')] + x_vocs[all_vocs.index('XBB.1.5')])

        voc_here = ['ALPHA','DELTA','BETA','EPSILON','IOTA','MU','OMICRON','GAMMA']
        x_wt.append(1 - np.sum([x_vocs[all_vocs.index(v)] for v in voc_here]))

    freq_wt = np.array(x_wt)
    freq_delta = np.array(x_delta)
    freq_omi = np.array(x_omi) + np.array(x_ba1) + np.array(x_ba2) + np.array(x_ba45)
    freq_alpha = np.array(x_alpha)
    freq_ba1 = np.array(x_ba1)
    freq_ba2 = np.array(x_ba2)
    freq_ba45 = np.array(x_ba45)
    freq_bq1 = np.array(x_bq1)
    freq_xbb = np.array(x_xbb)

    cases_full = [meta_country.loc[t]['new_cases']/meta_country.loc[t]['population'] for t in list(meta_country.index)] #FOR US STATES: new cases is per week
    if country in ['NY','TX','CA']:
        for i in range(0,len(cases_full)-1,7):
            week = cases_full[i] / 7
            cases_full[i:i+7] = np.ones(7) * week
    else:
        cases_full = clean_vac(list(meta_country.index),cases_full)
    cases_full_alpha = cases_full * freq_alpha
    cases_full_wt = cases_full * freq_wt
    cases_full_delta = cases_full * freq_delta
    cases_full_omi = cases_full * freq_omi
    cases_full_ba1 = cases_full * freq_ba1
    cases_full_ba2 = cases_full * freq_ba2
    cases_full_ba45 = cases_full * freq_ba45
    cases_full_bq1 = cases_full * freq_bq1
    cases_full_xbb = cases_full * freq_xbb

    recov_tot = [[np.sum(cases_full[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_delta = [[np.sum(cases_full_delta[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_omi = [[np.sum(cases_full_omi[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_ba1 = [[np.sum(cases_full_ba1[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_ba2 = [[np.sum(cases_full_ba2[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_ba45 = [[np.sum(cases_full_ba45[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_bq1 = [[np.sum(cases_full_bq1[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_xbb = [[np.sum(cases_full_xbb[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]

    recov_alpha = [[np.sum(cases_full_alpha[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_wt = [[np.sum(cases_full_wt[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    country2immuno2time[country]['RECOV_DELTA'] = {a[1]: a[0] for a in recov_delta}
    country2immuno2time[country]['RECOV_OMI'] = {a[1]: a[0] for a in recov_omi}
    country2immuno2time[country]['RECOV_ALPHA'] = {a[1]: a[0] for a in recov_alpha}
    country2immuno2time[country]['RECOV_WT'] = {a[1]: a[0] for a in recov_wt}
    country2immuno2time[country]['RECOV_BA1'] = {a[1]: a[0] for a in recov_ba1}
    country2immuno2time[country]['RECOV_BA2'] = {a[1]: a[0] for a in recov_ba2}
    country2immuno2time[country]['RECOV_BA45'] = {a[1]: a[0] for a in recov_ba45}
    country2immuno2time[country]['RECOV_BQ1'] = {a[1]: a[0] for a in recov_bq1}
    country2immuno2time[country]['RECOV_XBB'] = {a[1]: a[0] for a in recov_xbb}
    country2immuno2time[country]['RECOV_TOT'] = defaultdict(lambda: 0.0, {a[1]: a[0] for a in recov_tot})

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
dT['BQ.1']['BA.4/5'] = 0.3 


dT['XBB']['XBB'] = 1.

k=3.0 #2.2-- 4.2 -> sigma = 1/1.96
n50 = np.log10(0.2 * 94) #0.14 -- 0.28 -> sigma 0.06/1.96
time_decay = 90
tau_decay = time_decay



def sigmoid_func(t,mean=n50,s=k):
    val = 1 / (1 + np.exp(-s * (t - mean)))
    return val

def integrate_S_component(country,time_50,tau_decay,country2immuno2time, immunisation,dTiter_1, dTiter_2):
    times = sorted(list(country2immuno2time[country][immunisation].keys()))
    if immunisation == 'VAC_cor':
        country2immuno2time[country][immunisation] = country2immuno2time[country][time_50]
        times = sorted(list(country2immuno2time[country][immunisation].keys()))
        T0 = 223
    elif immunisation == 'VAC':
        T0 = 223
    elif 'RECOV' in immunisation:
        T0 = 94
    elif immunisation == 'BOOST_cor':
        country2immuno2time[country][immunisation] = country2immuno2time_bivalent[country][time_50]
        times = sorted(list(country2immuno2time[country][immunisation].keys()))
        T0 = 223 * 4
    elif immunisation == 'BOOST':
        T0 = 223 * 4
    elif immunisation == 'BIVALENT':
        T0 = 223 * 4
    
    time_index = 0
    while country2immuno2time[country][immunisation][times[time_index]] == 0.0:
        time_index += 1

    Cbar_1 = 0.0
    Cbar_2 = 0.0
    while times[time_index] < time_50:
        dt_vac = time_50 - times[time_index]
        weight = country2immuno2time[country][immunisation][times[time_index]] - country2immuno2time[country][immunisation][times[time_index-1]]
        C_1 = sigmoid_func(np.log10(T0) - np.log10(np.exp(dt_vac/tau_decay)) - np.log10(dTiter_1), n50, k)
        C_2 = sigmoid_func(np.log10(T0) - np.log10(np.exp(dt_vac/tau_decay)) - np.log10(dTiter_2), n50, k)
        Cbar_1 += C_1 * weight
        Cbar_2 += C_2 * weight
        time_index += 1

    return Cbar_1, Cbar_2

def approx_immune_weight(country,time_50,tau_decay,country2immuno2time, immunisation):
    times = sorted(list(country2immuno2time[country][immunisation].keys()))
    time_index = 0
    while country2immuno2time[country][immunisation][times[time_index]] == 0.0:
        time_index += 1
        if time_index > len(times)-1:
            return 0.0

    R_k = 0.0
    while times[time_index] < time_50:
        dt_vac = time_50 - times[time_index]
        weight = country2immuno2time[country][immunisation][times[time_index]] - country2immuno2time[country][immunisation][times[time_index-1]]
        W = sigmoid_func(np.log10(94) - np.log10(np.exp(dt_vac/tau_decay)), n50, k)
        # R_k += np.exp(- dt_vac / tau_decay) * weight
        R_k += W * weight
        time_index += 1

    return R_k



def integrate_S_components(country,time_50,tau_decay,country2immuno2time, immunisation,dTiters):
    times = sorted(list(country2immuno2time[country][immunisation].keys()))
    if immunisation == 'VAC_cor':
        country2immuno2time[country][immunisation] = country2immuno2time[country][time_50]
        times = sorted(list(country2immuno2time[country][immunisation].keys()))
        T0 = 223
    elif immunisation == 'VAC':
        T0 = 223
    elif 'RECOV' in immunisation:
        T0 = 94
    elif immunisation == 'BOOST_cor':
        country2immuno2time[country][immunisation] = country2immuno2time_bivalent[country][time_50]
        times = sorted(list(country2immuno2time[country][immunisation].keys()))
        T0 = 223 * 4
    elif immunisation == 'BOOST':
        T0 = 223 * 4
    elif immunisation == 'BIVALENT':
        T0 = 223 * 4

    time_index = 0
    while country2immuno2time[country][immunisation][times[time_index]] == 0.0:
        time_index += 1

    Cbar = np.zeros(len(dTiters))
    while times[time_index] < time_50:
        dt_vac = time_50 - times[time_index]
        weight = country2immuno2time[country][immunisation][times[time_index]] - country2immuno2time[country][immunisation][times[time_index-1]]
        for i, dT in enumerate(dTiters):
            Cbar[i] += weight * sigmoid_func(np.log10(T0) - np.log10(np.exp(dt_vac/tau_decay)) - np.log10(dT), n50, k)
        time_index += 1

    return Cbar


data_set = []
data_fitness = []
s_window = []
for c in countries:
    meta_country= meta_df.loc[list(meta_df['location'] == c[0] + c[1:].lower())]
    if len(c) <= 3:
        meta_country= meta_df.loc[list(meta_df['location'] == c)]
    if c == 'NORWAY':
        meta_country = meta_country.iloc[1:]
    meta_country.index = meta_country['FLAI_time']
    with open("DATA/2023_04_01/freq_traj_" + c.upper() + ".json",'r') as f:
        freq_traj = json.load(f)
    with open("DATA/2023_04_01/multiplicities_" + c.upper() + ".json",'r') as f:
        counts = json.load(f)
    with open("DATA/2023_04_01/multiplicities_Z_" + c.upper() + ".json",'r') as f:
        Z = json.load(f)
    for t in list(meta_country.index):
        if t > Time.dateToCoordinate("2023-03-29"):
            continue
        Cbar = defaultdict(lambda:defaultdict(lambda:0.0))

        if np.sum(list(country2immuno2time[c][t].values())) != 0.0:
            voc_here = ['DELTA','ALPHA','OMICRON','WT']
            Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'VAC_cor',[dT['VAC'][voc] for voc in voc_here])
            for i, voc in enumerate(voc_here):
                Cbar['VAC'][voc] = Cout[i]
        else:
            voc_here = ['DELTA','ALPHA','OMICRON','WT']
            for i, voc in enumerate(voc_here):
                Cbar['VAC'][voc] = 0.0
        Cbar['VAC0']['OMICRON'], Cbar['VAC0']['DELTA'] = integrate_S_components(c,t, tau_decay,country2immuno2time, 'VAC', [dT['VAC']['OMICRON'],dT['VAC']['DELTA']])

        if np.sum(list(country2immuno2time_bivalent[c][t].values())) != 0.0:
            voc_here = sorted(list(dT['BOOST'].keys()))
            Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'BOOST_cor',[dT['BOOST'][voc] for voc in voc_here])
            for i, voc in enumerate(voc_here):
                Cbar['BOOST'][voc] = Cout[i]
        else:
            voc_here = sorted(list(dT['BOOST'].keys()))
            for i, voc in enumerate(voc_here):
                Cbar['BOOST'][voc] = 0.0

        if np.sum(list(country2immuno2time[c]['BIVALENT'].values())) != 0.0:
            voc_here = sorted(list(dT['BIVALENT'].keys()))
            Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'BIVALENT',[dT['BIVALENT'][voc] for voc in voc_here])
            for i, voc in enumerate(voc_here):
                Cbar['BIVALENT'][voc] = Cout[i]
        else:
            voc_here = sorted(list(dT['BIVALENT'].keys()))
            for i, voc in enumerate(voc_here):
                Cbar['BIVALENT'][voc] = 0.0

        voc_here = ['WT','ALPHA','DELTA','OMICRON']
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_ALPHA',[dT['ALPHA'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_ALPHA'][voc] = Cout[i]

        voc_here = ['WT','ALPHA','DELTA','OMICRON']
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_DELTA',[dT['DELTA'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_DELTA'][voc] = Cout[i]

        voc_here = ['WT','ALPHA','DELTA','OMICRON']
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_DELTA',[dT['DELTA'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_DELTA'][voc] = Cout[i]

        voc_here = sorted(list(dT['OMICRON'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_OMI',[dT['OMICRON'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_OMI'][voc] = Cout[i]

        voc_here = sorted(list(dT['BA.1'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_BA1',[dT['BA.1'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_BA1'][voc] = Cout[i]

        voc_here = sorted(list(dT['BA.2'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_BA2',[dT['BA.2'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_BA2'][voc] = Cout[i]

        voc_here = sorted(list(dT['BA.4/5'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_BA45',[dT['BA.4/5'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_BA45'][voc] = Cout[i]

        voc_here = sorted(list(dT['BQ.1'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_BQ1',[dT['BQ.1'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_BQ1'][voc] = Cout[i]

        voc_here = sorted(list(dT['XBB'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_XBB',[dT['XBB'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_XBB'][voc] = Cout[i]

        #Immunity functions for the s_window:
        if np.sum(list(country2immuno2time[c][t].values())) != 0.0:
            voc_here = ['DELTA','ALPHA','OMICRON','WT']
            Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'VAC_cor',[dT['VAC'][voc] * 4 for voc in voc_here])
            for i, voc in enumerate(voc_here):
                Cbar['VAC_adv'][voc] = Cout[i]
        else:
            for i, voc in enumerate(voc_here):
                Cbar['VAC_adv'][voc] = 0.0
        Cbar['VAC0_adv']['OMICRON'], Cbar['VAC0_adv']['DELTA'] = integrate_S_components(c,t, tau_decay,country2immuno2time, 'VAC', [dT['VAC']['OMICRON']*4,dT['VAC']['DELTA']*4])
        if np.sum(list(country2immuno2time_bivalent[c][t].values())) != 0.0:
            voc_here = sorted(list(dT['BOOST'].keys()))
            Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'BOOST_cor',[dT['BOOST'][voc] * 4 for voc in voc_here])
            for i, voc in enumerate(voc_here):
                Cbar['BOOST_adv'][voc] = Cout[i]
        else:
            voc_here = sorted(list(dT['BOOST'].keys()))
            for i, voc in enumerate(voc_here):
                Cbar['BOOST_adv'][voc] = 0.0

        if np.sum(list(country2immuno2time[c]['BIVALENT'].values())) != 0.0:
            voc_here = sorted(list(dT['BIVALENT'].keys()))
            Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'BIVALENT',[dT['BIVALENT'][voc] * 4 for voc in voc_here])
            for i, voc in enumerate(voc_here):
                Cbar['BIVALENT_adv'][voc] = Cout[i]
        else:
            voc_here = sorted(list(dT['BIVALENT'].keys()))
            for i, voc in enumerate(voc_here):
                Cbar['BIVALENT_adv'][voc] = 0.0

        voc_here = ['WT','ALPHA','DELTA','OMICRON']
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_ALPHA',[dT['ALPHA'][voc] * 4 for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_ALPHA_adv'][voc] = Cout[i]

        voc_here = ['WT','ALPHA','DELTA','OMICRON']
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_DELTA',[dT['DELTA'][voc] * 4 for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_DELTA_adv'][voc] = Cout[i]

        voc_here = ['WT','ALPHA','DELTA','OMICRON']
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_DELTA',[dT['DELTA'][voc] * 4 for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_DELTA_adv'][voc] = Cout[i]

        voc_here = sorted(list(dT['OMICRON'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_OMI',[dT['OMICRON'][voc] * 4 for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_OMI_adv'][voc] = Cout[i]

        voc_here = sorted(list(dT['BA.1'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_BA1',[dT['BA.1'][voc] * 4 for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_BA1_adv'][voc] = Cout[i]

        voc_here = sorted(list(dT['BA.2'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_BA2',[dT['BA.2'][voc] * 4 for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_BA2_adv'][voc] = Cout[i]

        voc_here = sorted(list(dT['BA.4/5'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_BA45',[dT['BA.4/5'][voc] * 4 for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_BA45_adv'][voc] = Cout[i]


        voc_here = sorted(list(dT['BQ.1'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_BQ1',[dT['BQ.1'][voc] * 4 for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_BQ1_adv'][voc] = Cout[i]

        voc_here = sorted(list(dT['XBB'].keys()))
        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_XBB',[dT['XBB'][voc] * 4 for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_XBB_adv'][voc] = Cout[i]

        s_window.append([c,t, Cbar['VAC']['DELTA'] - Cbar['VAC_adv']['DELTA'], Cbar['VAC0']['DELTA'] - Cbar['VAC0_adv']['DELTA'], Cbar['RECOV_DELTA']['DELTA'] - Cbar['RECOV_DELTA_adv']['DELTA'], Cbar['BOOST']['DELTA'] - Cbar['BOOST_adv']['DELTA'],
        Cbar['RECOV_BA1']['BA.2'] - Cbar['RECOV_BA1_adv']['BA.2'], Cbar['RECOV_BA2']['BA.2'] - Cbar['RECOV_BA2_adv']['BA.2'], Cbar['BOOST']['BA.2'] - Cbar['BOOST_adv']['BA.2'],
        Cbar['RECOV_BA1']['BA.4/5'] -Cbar['RECOV_BA1_adv']['BA.4/5'],Cbar['RECOV_BA2']['BA.4/5'] -Cbar['RECOV_BA2_adv']['BA.4/5'], Cbar['RECOV_BA45']['BA.4/5'] -Cbar['RECOV_BA45_adv']['BA.4/5'],
        Cbar['BOOST']['BA.4/5'] -Cbar['BOOST_adv']['BA.4/5'], Cbar['BOOST']['OMICRON'] -Cbar['BOOST_adv']['OMICRON'], Cbar['RECOV_BA1']['BA.1'] -Cbar['RECOV_BA1_adv']['BA.1'],
        Cbar['VAC']['ALPHA'] -Cbar['VAC_adv']['ALPHA'], Cbar['RECOV_ALPHA']['ALPHA'] -Cbar['RECOV_ALPHA_adv']['ALPHA'], Cbar['VAC']['WT'] -Cbar['VAC_adv']['WT'],
        Cbar['BIVALENT']['BA.4/5'] - Cbar['BIVALENT_adv']['BA.4/5'],Cbar['BOOST']['BQ.1'] - Cbar['BOOST_adv']['BQ.1'],Cbar['BIVALENT']['BQ.1'] - Cbar['BIVALENT_adv']['BQ.1'],Cbar['RECOV_BA1']['BQ.1'] - Cbar['RECOV_BA1_adv']['BQ.1']
        ,Cbar['RECOV_BA2']['BQ.1'] - Cbar['RECOV_BA2_adv']['BQ.1'],Cbar['RECOV_BA45']['BQ.1'] - Cbar['RECOV_BA45_adv']['BQ.1'], Cbar['RECOV_BQ1']['BQ.1'] - Cbar['RECOV_BQ1_adv']['BQ.1'], 
        Cbar['BOOST']['XBB'] - Cbar['BOOST_adv']['XBB'],Cbar['BIVALENT']['XBB'] - Cbar['BIVALENT_adv']['XBB'],Cbar['RECOV_BQ1']['XBB'] - Cbar['RECOV_BQ1_adv']['XBB'],Cbar['RECOV_BA45']['XBB'] - Cbar['RECOV_BA45_adv']['XBB'], Cbar['RECOV_XBB']['XBB'] - Cbar['RECOV_XBB_adv']['XBB']])

        R_wt  = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_WT')
        R_vac = approx_immune_weight(c,t,tau_decay,country2immuno2time,'VAC')
        R_boost = approx_immune_weight(c,t,tau_decay,country2immuno2time,'BOOST')
        R_bivalent = approx_immune_weight(c,t,tau_decay,country2immuno2time,'BIVALENT')
        R_alpha = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_ALPHA')
        R_delta = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_DELTA')
        R_omi = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_OMI')
        R_ba1 = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_BA1')
        R_ba2 = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_BA2')
        R_ba45 = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_BA45')
        R_bq1 = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_BQ1')
        R_xbb = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_XBB')

        all_vocs = sorted(list(WHOlabels.values()))
        x_vocs = []
        for voc in all_vocs:
            if pango2flupredict[voc] not in freq_traj.keys():
                x_vocs.append(0.0)
                continue
            if str(t) in freq_traj[pango2flupredict[voc]].keys():
                x_vocs.append(freq_traj[pango2flupredict[voc]][str(t)])
            else:
                x_vocs.append(0.0)



        line = [c,t] + x_vocs 
        line += [Cbar['VAC'][x] for x in sorted(list(Cbar['VAC'].keys()))] 
        line += [Cbar['VAC0'][x] for x in sorted(list(Cbar['VAC0'].keys()))]
        line += [Cbar['BOOST'][x] for x in sorted(list(Cbar['BOOST'].keys()))]
        line += [Cbar['BIVALENT'][x] for x in sorted(list(Cbar['BIVALENT'].keys()))]
        line += [Cbar['RECOV_ALPHA'][x] for x in sorted(list(Cbar['RECOV_ALPHA'].keys()))]
        line += [Cbar['RECOV_DELTA'][x] for x in sorted(list(Cbar['RECOV_DELTA'].keys()))]
        line += [Cbar['RECOV_OMI'][x] for x in sorted(list(Cbar['RECOV_OMI'].keys()))]
        line += [Cbar['RECOV_BA1'][x] for x in sorted(list(Cbar['RECOV_BA1'].keys()))]
        line += [Cbar['RECOV_BA2'][x] for x in sorted(list(Cbar['RECOV_BA2'].keys()))]
        line += [Cbar['RECOV_BA45'][x] for x in sorted(list(Cbar['RECOV_BA45'].keys()))]
        line += [Cbar['RECOV_BQ1'][x] for x in sorted(list(Cbar['RECOV_BQ1'].keys()))]
        line += [Cbar['RECOV_XBB'][x] for x in sorted(list(Cbar['RECOV_XBB'].keys()))]
        line += [R_wt,R_vac,R_boost,R_bivalent,R_alpha,R_delta,R_omi,R_ba1, R_ba2,R_ba45,R_bq1,R_xbb]
        line += [country2immuno2time[c]['RECOV_TOT'][t], country2immuno2time[c]['VAC'][t], country2immuno2time[c]['BOOST'][t], country2immuno2time[c]['BIVALENT'][t]]
        line += [country2immuno2time[c]['RECOV_TOT'][t] - country2immuno2time[c]['RECOV_TOT'][t-7], country2immuno2time[c]['VAC'][t] - country2immuno2time[c]['VAC'][t-7], 
        country2immuno2time[c]['BOOST'][t] - country2immuno2time[c]['BOOST'][t-7], country2immuno2time[c]['BIVALENT'][t] - country2immuno2time[c]['BIVALENT'][t-7]]
        data_set.append(line)

columns_fig4_data = ['country','time'] + [f'x_{voc}' for voc in sorted(list(WHOlabels.values()))]
columns_fig4_data += [f'C_VAC_{alpha}' for alpha in sorted(list(Cbar['VAC'].keys()))]
columns_fig4_data += [f'C_VAC0_{alpha}' for alpha in sorted(list(Cbar['VAC0'].keys()))]
columns_fig4_data += [f'C_BOOST_{alpha}' for alpha in sorted(list(Cbar['BOOST'].keys()))]
columns_fig4_data += [f'C_BIVALENT_{alpha}' for alpha in sorted(list(Cbar['BIVALENT'].keys()))]
columns_fig4_data += [f'C_RECOV_ALPHA_{alpha}' for alpha in sorted(list(Cbar['RECOV_ALPHA'].keys()))]
columns_fig4_data += [f'C_RECOV_DELTA_{alpha}' for alpha in sorted(list(Cbar['RECOV_DELTA'].keys()))]
columns_fig4_data += [f'C_RECOV_OMI_{alpha}' for alpha in sorted(list(Cbar['RECOV_OMI'].keys()))]
columns_fig4_data += [f'C_RECOV_BA1_{alpha}' for alpha in sorted(list(Cbar['RECOV_BA1'].keys()))]
columns_fig4_data += [f'C_RECOV_BA2_{alpha}' for alpha in sorted(list(Cbar['RECOV_BA2'].keys()))]
columns_fig4_data += [f'C_RECOV_BA45_{alpha}' for alpha in sorted(list(Cbar['RECOV_BA45'].keys()))]
columns_fig4_data += [f'C_RECOV_BQ1_{alpha}' for alpha in sorted(list(Cbar['RECOV_BQ1'].keys()))]
columns_fig4_data += [f'C_RECOV_XBB_{alpha}' for alpha in sorted(list(Cbar['RECOV_XBB'].keys()))]
columns_fig4_data += ['R_wt','R_vac','R_boost','R_biv','R_alpha','R_delta','R_omi','R_ba1','R_ba2','R_ba45','R_bq1','R_xbb']
columns_fig4_data += ['tot_cases','vac','boost','bivalent']
columns_fig4_data += ['tot_cases_rate','vac_rate','boost_rate','bivalent_rate']
df = pd.DataFrame(data_set,columns=columns_fig4_data)
df.to_csv("output/data_immune_trajectories.txt",'\t',index=False)

s_window = pd.DataFrame(s_window, columns = ['country','time','sw_delta_vac','sw_delta_vac0','sw_delta_delta','sw_delta_bst','sw_ba2_ba1','sw_ba2_ba2','sw_ba2_bst',
        'sw_ba5_ba1','sw_ba5_ba2','sw_ba5_ba5','sw_ba5_bst','sw_ba1_bst','sw_ba1_ba1','sw_alpha_vac','sw_alpha_alpha','sw_wt_vac','sw_ba5_biv','sw_bq1_bst','sw_bq1_biv','sw_bq1_ba1','sw_bq1_ba2','sw_bq1_ba5','sw_bq1_bq1','sw_xbb_bst','sw_xbb_biv','sw_xbb_bq1','sw_xbb_ba5','sw_xbb_xbb'])
s_window.to_csv("output/selection_potentials.txt",'\t',index=False)


c_channels=columns_fig4_data[len(['country','time'] + [f'x_{voc}' for voc in sorted(list(WHOlabels.values()))]):columns_fig4_data.index("R_wt")]
s_cols = ['sw_delta_vac','sw_delta_vac0','sw_delta_delta','sw_delta_bst','sw_ba2_ba1','sw_ba2_ba2','sw_ba2_bst','sw_ba5_ba1','sw_ba5_ba2','sw_ba5_ba5','sw_ba5_bst','sw_ba1_bst','sw_ba1_ba1','sw_alpha_vac','sw_alpha_alpha','sw_wt_vac','sw_ba5_biv','sw_bq1_bst','sw_bq1_biv','sw_bq1_ba1','sw_bq1_ba2','sw_bq1_ba5','sw_bq1_bq1','sw_xbb_bst','sw_xbb_biv','sw_xbb_bq1','sw_xbb_ba5','sw_xbb_xbb']
#Average the R weights over all the countries
time2R = defaultdict(lambda: defaultdict(lambda: []))
df = pd.read_csv("output/data_immune_trajectories.txt",'\t',index_col=False)
s_window = pd.read_csv("output/selection_potentials.txt",'\t',index_col=False)

for c in countries:
    df_c = df.loc[list(df.country == c)]
    sw_c = s_window.loc[list(s_window.country==c)]
    for line in df_c.iterrows():
        line = line[1]

        time2R['R_wt'][line.time].append(line.R_wt)
        time2R['R_vac'][line.time].append(line.R_vac)
        time2R['R_boost'][line.time].append(line.R_boost)
        time2R['R_biv'][line.time].append(line.R_biv)
        time2R['R_alpha'][line.time].append(line.R_alpha)
        time2R['R_delta'][line.time].append(line.R_delta)
        time2R['R_omi'][line.time].append(line.R_omi)
        time2R['R_ba1'][line.time].append(line.R_ba1)
        time2R['R_ba2'][line.time].append(line.R_ba2)
        time2R['R_ba45'][line.time].append(line.R_ba45)
        time2R['R_bq1'][line.time].append(line.R_bq1)
        time2R['R_xbb'][line.time].append(line.R_xbb)

        time2R['tot_cases'][line.time].append(line.tot_cases)
        time2R['vac'][line.time].append(line.vac)
        time2R['boost'][line.time].append(line.boost)
        time2R['bivalent'][line.time].append(line.bivalent)

        time2R['tot_cases_rate'][line.time].append(line.tot_cases_rate)
        time2R['vac_rate'][line.time].append(line.vac_rate)
        time2R['boost_rate'][line.time].append(line.boost_rate)
        time2R['bivalent_rate'][line.time].append(line.bivalent_rate)

        x_wt = 1 - line.x_ALPHA - line.x_DELTA - line.x_BETA - line.x_EPSILON - line.x_IOTA - line.x_MU - line.x_OMICRON - line.x_GAMMA
        if line.time > Time.dateToCoordinate("2021-09-01"):
            x_wt = 0.0
        all_vocs = sorted(list(WHOlabels.values()))
        x_vocs = []
        for voc in all_vocs:
            time2R[f'x_{voc}'][line.time].append(line[f'x_{voc}'])
        time2R['x_wt'][line.time].append(x_wt)
        
        for C in c_channels:
            time2R[C][line.time].append(float(line[C]))
    for line in sw_c.iterrows():
        line = line[1]
        for C in s_cols:
            time2R[C][line.time].append(float(line[C]))

all_vocs = sorted(list(WHOlabels.values()))
R_av = []
for t in sorted(list(time2R['R_vac'].keys())):
    X = [np.mean(time2R['x_wt'][t])] + [np.mean(time2R[f'x_{voc}'][t]) for voc in all_vocs]
    X = list(np.array(X) / np.sum(X))
    C_all = []
    for CC in c_channels:
        C_all.append(np.mean(time2R[CC][t]))

    R_av.append([int(t)] + X  + [np.mean(time2R['R_wt'][t]), np.mean(time2R['R_vac'][t]), np.mean(time2R['R_boost'][t]),np.mean(time2R['R_biv'][t]), np.mean(time2R['R_alpha'][t]), np.mean(time2R['R_delta'][t]), np.mean(time2R['R_omi'][t]), 
        np.mean(time2R['R_ba1'][t]), np.mean(time2R['R_ba2'][t]),np.mean(time2R['R_ba45'][t]),np.mean(time2R['R_bq1'][t]),np.mean(time2R['R_xbb'][t]),np.mean(time2R['tot_cases'][t]), np.mean(time2R['vac'][t]),np.mean(time2R['boost'][t]),np.mean(time2R['bivalent'][t]),
        np.mean(time2R['tot_cases_rate'][t]), np.mean(time2R['vac_rate'][t]),np.mean(time2R['boost_rate'][t]),np.mean(time2R['bivalent_rate'][t])] + C_all)# + F_all)
R_av_columns=['time'] + ['x_wt'] + [f'x_{voc}' for voc in all_vocs] + ['R_wt','R_vac','R_boost','R_biv','R_alpha','R_delta','R_omi','R_ba1','R_ba2','R_ba45','R_bq1','R_xbb','tot_cases','vac','boost','bivalent','tot_cases_rate','vac_rate','boost_rate','bivalent_rate'] + c_channels
R_av = pd.DataFrame(R_av,columns=R_av_columns)# + f_channels)
R_av.to_csv("output/R_average_may.txt",'\t',index=False)

R_av_sw = []
for t in sorted(list(time2R['sw_delta_vac'].keys())):
    mean_s = []
    for CC in s_cols:
        mean_s.append(np.mean(time2R[CC][t]))
    R_av_sw.append([int(t)] + mean_s)
R_av_sw = pd.DataFrame(R_av_sw, columns=['time'] + s_cols)
R_av_sw.to_csv("output/selection_potentials_average.txt",'\t',index=False)
