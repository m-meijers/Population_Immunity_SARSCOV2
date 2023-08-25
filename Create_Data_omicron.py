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

s_hat_alpha = pd.read_csv("output/s_hat_alpha_wt.txt",'\t',index_col=False)
s_hat_delta = pd.read_csv("output/s_hat_delta_alpha.txt",'\t',index_col=False)
s_hat_omi = pd.read_csv("output/s_hat_omi_delta.txt",'\t',index_col=False)
s_hat_ba2 = pd.read_csv("output/s_hat_ba2_ba1.txt",'\t',index_col=False)
s_hat_ba45 = pd.read_csv("output/s_hat_ba45_ba2.txt",'\t',index_col=False)
s_hat_bq1 = pd.read_csv("output/s_hat_bq1_ba45.txt",'\t',index_col=False)

meta_df = pd.read_csv("DATA/clean_data.txt",sep='\t',index_col=False)   

countries = list(set(s_hat_ba2.country)) + list(set(s_hat_ba45.country)) + list(set(s_hat_bq1.country))
countries = sorted(list(set(countries)))


country2immuno2time = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
country2immuno2time_bivalent = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
for country in countries:
    # for country in ['Netherlands']:
    x_limit = 0.01
    # country = 'ITALY'
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
    country2immuno2time[country]['VAC'] = {list(meta_country.index)[i] : vinterp[i] for i in range(len(list(meta_country.index)))}
    booster = np.array([meta_country.loc[t]['total_boosters_per_hundred']/100. for t in meta_country.index])
    booster = np.where(booster == 0.0, np.ones(len(booster)) * float('NaN'), booster)
    if np.sum(np.isnan(booster)) == len(booster):
        boosterp = np.zeros(len(booster))
    else:
        boosterp = clean_vac(list(meta_country.index), booster)
    country2immuno2time[country]['BOOST'] = {list(meta_country.index)[i]:boosterp[i] for i in range(len(list(meta_country.index)))}
    bivalent = np.array([meta_country.loc[t]['total_bivalent_boosters_per_hundred']/100. for t in meta_country.index])
    bivalent = np.where(bivalent == 0.0, np.ones(len(bivalent)) * float('NaN'), bivalent)
    if np.sum(np.isnan(bivalent)) == len(bivalent):
        bivalentp = np.zeros(len(bivalent))
    else:
        bivalentp = clean_vac(list(meta_country.index), bivalent)
    country2immuno2time[country]['BIVALENT'] = {list(meta_country.index)[i]:bivalentp[i] for i in range(len(list(meta_country.index)))}

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
        x_ba2.append(x_vocs[all_vocs.index('BA.2')] +  x_vocs[all_vocs.index('BA.2.12.1')])
        x_ba45.append(x_vocs[all_vocs.index('BA.4')] + x_vocs[all_vocs.index('BA.5')] + x_vocs[all_vocs.index('BA.5.9')])
        x_bq1.append(x_vocs[all_vocs.index('BQ.1')] + x_vocs[all_vocs.index('BQ.1.1')])

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

    recov_tot = [[np.sum(cases_full[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_delta = [[np.sum(cases_full_delta[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_omi = [[np.sum(cases_full_omi[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_ba1 = [[np.sum(cases_full_ba1[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_ba2 = [[np.sum(cases_full_ba2[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_ba45 = [[np.sum(cases_full_ba45[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_bq1 = [[np.sum(cases_full_bq1[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]

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
    country2immuno2time[country]['RECOV_TOT'] = {a[1]: a[0] for a in recov_tot}

#Make average C
averageImmuno2time2recov_list = defaultdict(lambda: defaultdict(lambda: []))
averageImmuno2time2recov = defaultdict(lambda: defaultdict(lambda: 0.0))
for k_channel in ['RECOV_BA1','RECOV_BA2','RECOV_BA45','RECOV_BQ1']:
    for country in countries:
        for t in country2immuno2time[country][k_channel].keys():
            averageImmuno2time2recov_list[k_channel][t].append(country2immuno2time[country][k_channel][t])
for k_channel in ['RECOV_BA1','RECOV_BA2','RECOV_BA45','RECOV_BQ1']:
    for t in averageImmuno2time2recov_list[k_channel].keys():
        averageImmuno2time2recov[k_channel][t] = np.mean(averageImmuno2time2recov_list[k_channel][t])

dT = defaultdict(lambda:defaultdict(lambda:1.0)) #Titer drop matrix in fold dT[RHO][ALPHA]
dT['VAC']['ALPHA'] = 1.8
dT['VAC']['DELTA'] = 3.2
dT['VAC']['OMICRON'] = 47
dT['VAC']['WT'] = 1.
dT['BOOST']['WT']=1.
dT['BOOST']['ALPHA']=1.8
dT['BOOST']['DELTA']=2.8
dT['BOOST']['OMICRON']=6.7 #was 6.4
# dT['BOOST']['OMICRON']=6.4 #was 6.4
dT['BOOST']['BA.1']=6.7 #same as omicron
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

k=3.0 #2.2-- 4.2 -> sigma = 1/1.96
n50 = np.log10(0.2 * 94) #0.14 -- 0.28 -> sigma 0.06/1.96
time_decay = 90
# time_decay = 70
tau_decay = time_decay


def integrate_S_components_INF(time2y,time_here,tau_decay, T0,dTiters):
    times = sorted(list(time2y.keys()))
    time_index = 0
    while time2y[times[time_index]] == 0.0:
        time_index += 1

    Cbar = np.zeros(len(dTiters))
    while times[time_index] < time_here:
        dt_vac = time_here - times[time_index]
        weight = time2y[times[time_index]] - time2y[times[time_index-1]]
        for i, dT in enumerate(dTiters):
            Cbar[i] += weight * sigmoid_func(np.log10(T0) - np.log10(np.exp(dt_vac/tau_decay)) - np.log10(dT), n50, k)
        time_index += 1

    return Cbar

def integrate_S_components_VAC(country,time_50,tau_decay,country2immuno2time, immunisation,dTiters):
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

lines = []
for line in s_hat_ba2.itertuples():
    country = line.country
    time_50 = int(line.FLAI_time)
    s_hat = line.s_hat
    s_var = line.s_var
    C_ba2_bst, C_ba1_bst = integrate_S_components_VAC(country,time_50,tau_decay,country2immuno2time, 'BOOST',[dT['BOOST']['BA.2'],dT['BOOST']['BA.1']])
    C_ba2_ba1, C_ba1_ba1 = integrate_S_components_INF(averageImmuno2time2recov['RECOV_BA1'],time_50,tau_decay,94,[dT['BA.1']['BA.2'],dT['BA.1']['BA.1']])
    C_ba2_ba2, C_ba1_ba2 = integrate_S_components_INF(averageImmuno2time2recov['RECOV_BA2'],time_50,tau_decay,94,[dT['BA.2']['BA.2'],dT['BA.2']['BA.1']])
    lines.append([country, time_50, s_hat, s_var, C_ba2_bst, C_ba1_bst, C_ba2_ba1, C_ba1_ba1, C_ba2_ba2, C_ba1_ba2])
lines = pd.DataFrame(lines, columns=['country','time','s_hat','s_var','C_ba2_bst', 'C_ba1_bst', 'C_ba2_ba1', 'C_ba1_ba1', 'C_ba2_ba2', 'C_ba1_ba2'])
lines.to_csv("output/Data_ba2_shift.txt",'\t',index=False)

lines = []
for line in s_hat_ba45.itertuples():
    country = line.country
    time_50 = int(line.FLAI_time)
    s_hat = line.s_hat
    s_var = line.s_var

    C_ba45_bst, C_ba2_bst = integrate_S_components_VAC(country,time_50,tau_decay,country2immuno2time, 'BOOST',[dT['BOOST']['BA.4/5'],dT['BOOST']['BA.2']])
    C_ba45_ba1, C_ba2_ba1 = integrate_S_components_INF(averageImmuno2time2recov['RECOV_BA1'],time_50,tau_decay,94,[dT['BA.1']['BA.4/5'],dT['BA.1']['BA.2']])
    C_ba45_ba2, C_ba2_ba2 = integrate_S_components_INF(averageImmuno2time2recov['RECOV_BA2'],time_50,tau_decay,94,[dT['BA.2']['BA.4/5'],dT['BA.2']['BA.2']])
    C_ba45_ba45, C_ba2_ba45 = integrate_S_components_INF(averageImmuno2time2recov['RECOV_BA45'],time_50,tau_decay,94,[dT['BA.4/5']['BA.4/5'],dT['BA.4/5']['BA.2']])
    lines.append([country, time_50, s_hat, s_var, C_ba45_bst,C_ba2_bst,C_ba45_ba1,C_ba2_ba1,C_ba45_ba2, C_ba2_ba2, C_ba45_ba45, C_ba2_ba45])
lines = pd.DataFrame(lines, columns=['country','time','s_hat','s_var','C_ba45_bst','C_ba2_bst','C_ba45_ba1','C_ba2_ba1','C_ba45_ba2', 'C_ba2_ba2', 'C_ba45_ba45', 'C_ba2_ba45'])
lines.to_csv("output/Data_ba45_shift.txt",'\t',index=False)

lines = []
for line in s_hat_bq1.itertuples():
    country = line.country
    time_50 = int(line.FLAI_time)
    s_hat = line.s_hat
    s_var = line.s_var

    C_bq1_bst, C_ba45_bst = integrate_S_components_VAC(country,time_50,tau_decay,country2immuno2time, 'BOOST_cor',[dT['BOOST']['BQ.1'],dT['BOOST']['BA.4/5']])
    if np.sum(list(country2immuno2time[country]['BIVALENT'].values())) > 0.0:
        C_bq1_biv, C_ba45_biv = integrate_S_components_VAC(country,time_50,tau_decay,country2immuno2time, 'BIVALENT',[dT['BIVALENT']['BQ.1'],dT['BIVALENT']['BA.4/5']])
    else:
        C_bq1_biv, C_ba45_biv = [0.0,0.0]
    C_bq1_ba1, C_ba45_ba1 = integrate_S_components_INF(averageImmuno2time2recov['RECOV_BA1'],time_50,tau_decay,94,[dT['BA.1']['BQ.1'],dT['BA.1']['BA.4/5']])
    C_bq1_ba2, C_ba45_ba2 = integrate_S_components_INF(averageImmuno2time2recov['RECOV_BA2'],time_50,tau_decay,94,[dT['BA.2']['BQ.1'],dT['BA.2']['BA.4/5']])
    C_bq1_ba45, C_ba45_ba45 = integrate_S_components_INF(averageImmuno2time2recov['RECOV_BA45'],time_50,tau_decay,94,[dT['BA.4/5']['BQ.1'],dT['BA.4/5']['BA.4/5']])
    C_bq1_bq1, C_ba45_bq1 = integrate_S_components_INF(averageImmuno2time2recov['RECOV_BQ1'],time_50,tau_decay,94,[dT['BQ.1']['BQ.1'],dT['BQ.1']['BA.4/5']])
    lines.append([country, time_50, s_hat, s_var, C_bq1_bst,C_ba45_bst, C_bq1_biv, C_ba45_biv, C_bq1_ba1,C_ba45_ba1,C_bq1_ba2,C_ba45_ba2,C_bq1_ba45, C_ba45_ba45,C_bq1_bq1, C_ba45_bq1])
lines = pd.DataFrame(lines, columns=['country','time','s_hat','s_var','C_bq1_bst','C_ba45_bst', 'C_bq1_biv', 'C_ba45_biv', 'C_bq1_ba1','C_ba45_ba1','C_bq1_ba2','C_ba45_ba2','C_bq1_ba45', 'C_ba45_ba45','C_bq1_bq1', 'C_ba45_bq1'])
lines.to_csv("output/Data_bq1_shift.txt",'\t',index=False)

