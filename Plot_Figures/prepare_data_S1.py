import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from matplotlib.lines import Line2D
import scipy.integrate as si
import scipy.stats as ss
import scipy.optimize as so
import json
from flai.util.Time import Time
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import matplotlib as mpl


df_s_da = pd.read_csv("../output/s_hat_delta_alpha.txt",sep='\t',index_col=False)
df_s_od = pd.read_csv("../output/s_hat_omi_delta.txt",sep='\t',index_col=False)
countries_da = sorted(list(set(df_s_da.country)))
countries_od = sorted(list(set(df_s_od.country)))
countries = sorted(list(set(countries_da + countries_od)))

country_variance = []
for c in countries_da:
    df_c = df_s_da.loc[list(df_s_da.country == c)]
    meansvar = np.mean(df_c.s_var)
    country_variance.append(meansvar)
median_svar_da = np.median(country_variance)
country_variance = []
for c in countries_od:
    df_c = df_s_od.loc[list(df_s_od.country == c)]
    meansvar = np.mean(df_c.s_var)
    country_variance.append(meansvar)
median_svar_od = np.median(country_variance)
df_s_da['s_var'] = np.array(df_s_da['s_var'] + median_svar_da)
df_s_od['s_var'] = np.array(df_s_od['s_var'] + median_svar_od)



#============================================================
#Prepare data 
#============================================================

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


meta_df = pd.read_csv("../DATA/clean_data.txt",sep='\t',index_col=False)   
country2immuno2time = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
country2immuno2time_bivalent = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
for country in countries:
    x_limit = 0.01
    with open("../DATA/2023_04_01/freq_traj_" + country.upper() + ".json",'r') as f:
        freq_traj = json.load(f)
    with open("../DATA/2023_04_01/multiplicities_" + country.upper() + ".json",'r') as f:
        counts = json.load(f)
    with open("../DATA/2023_04_01/multiplicities_Z_" + country.upper() + ".json",'r') as f:
        Z = json.load(f)
    meta_country= meta_df.loc[list(meta_df['location'] == country[0] + country[1:].lower())]
    if len(country) <= 3:
        meta_country= meta_df.loc[list(meta_df['location'] == country)]
    if country == 'NORWAY':
        meta_country = meta_country.iloc[1:]
    meta_country.index = meta_country['FLAI_time']
    meta_country = meta_country.loc[meta_country.FLAI_time < Time.dateToCoordinate("2022-04-01")]

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

        voc_here = ['ALPHA','DELTA','BETA','EPSILON','IOTA','MU','OMICRON','GAMMA']
        x_wt.append(1 - np.sum([x_vocs[all_vocs.index(v)] for v in voc_here]))

    freq_wt = np.array(x_wt)
    freq_delta = np.array(x_delta)
    freq_omi = np.array(x_omi) + np.array(x_ba1) + np.array(x_ba2)
    freq_alpha = np.array(x_alpha)
    freq_ba1 = np.array(x_ba1)
    freq_ba2 = np.array(x_ba2)

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


    recov_tot = [[np.sum(cases_full[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_delta = [[np.sum(cases_full_delta[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_omi = [[np.sum(cases_full_omi[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_ba1 = [[np.sum(cases_full_ba1[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_ba2 = [[np.sum(cases_full_ba2[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]

    recov_alpha = [[np.sum(cases_full_alpha[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_wt = [[np.sum(cases_full_wt[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    country2immuno2time[country]['RECOV_DELTA'] = {a[1]: a[0] for a in recov_delta}
    country2immuno2time[country]['RECOV_OMI'] = {a[1]: a[0] for a in recov_omi}
    country2immuno2time[country]['RECOV_ALPHA'] = {a[1]: a[0] for a in recov_alpha}
    country2immuno2time[country]['RECOV_WT'] = {a[1]: a[0] for a in recov_wt}
    country2immuno2time[country]['RECOV_BA1'] = {a[1]: a[0] for a in recov_ba1}
    country2immuno2time[country]['RECOV_BA2'] = {a[1]: a[0] for a in recov_ba2}
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


k=3.0 #2.2-- 4.2 -> sigma = 1/1.96
n50 = np.log10(0.2 * 94) #0.14 -- 0.28 -> sigma 0.06/1.96
time_decay = 90
tau_decay = time_decay


def sigmoid_func(t,mean=n50,s=k):
    val = 1 / (1 + np.exp(-s * (t - mean)))
    return val

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


lines = []
for c in countries:
    meta_country = meta_df.loc[list(meta_df['location'] == c[0] + c[1:].lower())]
    if len(c) <= 3:
        meta_country = meta_df.loc[list(meta_df['location'] == c)]
    if c == 'NORWAY':
        meta_country = meta_country.iloc[1:]
    meta_country.index = meta_country['FLAI_time']
    meta_country = meta_country.loc[meta_country.FLAI_time < Time.dateToCoordinate("2022-04-01")]
    with open("../DATA/2023_04_01/freq_traj_" + c.upper() + ".json",'r') as f:
        freq_traj = json.load(f)
    with open("../DATA/2023_04_01/multiplicities_" + c.upper() + ".json",'r') as f:
        counts = json.load(f)
    with open("../DATA/2023_04_01/multiplicities_Z_" + c.upper() + ".json",'r') as f:
        Z = json.load(f)

    for t in list(meta_country.index):
        Cbar = defaultdict(lambda:defaultdict(lambda:0.0))

        voc_here = ['DELTA','ALPHA','OMICRON','WT']

        if np.sum(list(country2immuno2time[c][t].values())) != 0.0:
            Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'VAC_cor',[dT['VAC'][voc] for voc in voc_here])
            for i, voc in enumerate(voc_here):
                Cbar['VAC'][voc] = Cout[i]
        else:
            for i, voc in enumerate(voc_here):
                Cbar['VAC'][voc] = 0.0
        Cbar['VAC0']['OMICRON'], Cbar['VAC0']['DELTA'] = integrate_S_components(c,t, tau_decay,country2immuno2time, 'VAC', [dT['VAC']['OMICRON'],dT['VAC']['DELTA']])

        for i, voc in enumerate(voc_here):
            Cbar['BOOST'][voc] = 0.0
       

        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_ALPHA',[dT['ALPHA'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_ALPHA'][voc] = Cout[i]

        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_DELTA',[dT['DELTA'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_DELTA'][voc] = Cout[i]

        Cout = integrate_S_components(c,t,tau_decay,country2immuno2time,'RECOV_OMI',[dT['OMICRON'][voc] for voc in voc_here])
        for i, voc in enumerate(voc_here):
            Cbar['RECOV_OMI'][voc] = Cout[i]

        R_wt  = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_WT')
        R_vac = approx_immune_weight(c,t,tau_decay,country2immuno2time,'VAC')
        R_boost = approx_immune_weight(c,t,tau_decay,country2immuno2time,'BOOST')
        R_alpha = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_ALPHA')
        R_delta = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_DELTA')
        R_omi = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_OMI')
        
        x_delta = []
        x_omi = []
        x_alpha = []
        x_ba1 = []
        x_ba2 = []
        x_wt = []
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

        x_delta.append(x_vocs[all_vocs.index('DELTA')])
        x_omi.append(x_vocs[all_vocs.index('OMICRON')])
        x_alpha.append(x_vocs[all_vocs.index('ALPHA')])
        x_ba1.append(x_vocs[all_vocs.index('BA.1')] + x_vocs[all_vocs.index('BA.1.1')])
        x_ba2.append(x_vocs[all_vocs.index('BA.2')] + x_vocs[all_vocs.index('BA.2.12.1')])

        voc_here = ['ALPHA','DELTA','BETA','EPSILON','IOTA','MU','OMICRON','GAMMA']
        x_wt.append(1 - np.sum([x_vocs[all_vocs.index(v)] for v in voc_here]))

        freq_wt = np.array(x_wt)
        freq_delta = np.array(x_delta)
        freq_omi = np.array(x_omi) + np.array(x_ba1) + np.array(x_ba2)
        freq_alpha = np.array(x_alpha)
        freq_ba1 = np.array(x_ba1)
        freq_ba2 = np.array(x_ba2)

        lines.append([c, t, freq_wt, freq_alpha, freq_delta, freq_omi, R_vac, R_boost, R_wt, R_alpha, R_delta, R_omi, Cbar['VAC']['ALPHA'], Cbar['VAC']['DELTA'], Cbar['VAC']['OMICRON'],
            Cbar['VAC0']['DELTA'], Cbar['VAC0']['OMICRON'], Cbar['BOOST']['ALPHA'], Cbar['BOOST']['DELTA'], Cbar['BOOST']['OMICRON'], Cbar['RECOV_ALPHA']['ALPHA'], Cbar['RECOV_ALPHA']['DELTA'],
            Cbar['RECOV_DELTA']['ALPHA'], Cbar['RECOV_DELTA']['DELTA'], Cbar['RECOV_DELTA']['OMICRON'], Cbar['RECOV_OMI']['DELTA'], Cbar['RECOV_OMI']['OMICRON']])

lines = pd.DataFrame(lines, columns=['country','time','freq_wt','freq_alpha','freq_delta','freq_omi','R_vac','R_boost','R_wt','R_alpha','R_delta','R_omi','C_vac_alpha','C_vac_delta','C_vac_omi',
    'C_vac0_delta','C_vac0_omi','C_boost_alpha','C_boost_delta','C_boost_omi','C_alpha_alpha','C_alpha_delta','C_delta_alpha','C_delta_delta','C_delta_omi','C_omi_delta','C_omi_omi'])
lines.to_csv("data_figS1.txt",'\t',index=False)


