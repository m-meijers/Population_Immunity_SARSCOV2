import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flai.util.Time import Time
import json
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from collections import defaultdict
import glob


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

df_da = pd.read_csv("../output/s_hat_delta_alpha.txt",'\t',index_col=False)
df_od = pd.read_csv("../output/s_hat_omi_delta.txt",'\t',index_col=False)
meta_df = pd.read_csv("../DATA/clean_data.txt",sep='\t',index_col=False)   

countries_da = sorted(list(set(df_da.country)))
countries_od = sorted(list(set(df_od.country)))
countries = sorted(list(set(countries_da).intersection(set(countries_od))))

x_limit = 0.01
dt = 30.0
min_count = 500
x_tot_min = 0.5

country_min_count = defaultdict(lambda: [])

country2immuno2time = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
lines = []
for country in countries:
    with open("../DATA/2023_04_01/freq_traj_" + country.upper() + ".json",'r') as f:
        freq_traj = json.load(f)
    with open("../DATA/2023_04_01/multiplicities_" + country.upper() + ".json",'r') as f:
        counts = json.load(f)
    with open("../DATA/2023_04_01/multiplicities_Z_" + country.upper() + ".json",'r') as f:
        Z = json.load(f)


    dates_ba2 = list(counts[pango2flupredict['BA.2']].keys())
    dates_ba22 = list(counts[pango2flupredict['BA.2.12.1']].keys())
    dates_ba4 = list(counts[pango2flupredict['BA.4']].keys())
    dates_ba5 = list(counts[pango2flupredict['BA.5']].keys())
    dates_ba2 = [int(a) for a in dates_ba2]
    dates_ba22 = [int(a) for a in dates_ba22]
    dates_ba4 = [int(a) for a in dates_ba4]
    dates_ba5 = [int(a) for a in dates_ba5]
    dates_ba2 = sorted(dates_ba2)
    dates_ba22 = sorted(dates_ba22)
    dates_ba4 = sorted(dates_ba4)
    dates_ba5 = sorted(dates_ba5)
    dates_ba4 = [a for a in dates_ba4 if a < Time.dateToCoordinate("2022-10-01") and a > Time.dateToCoordinate("2022-04-01")]
    dates_ba5 = [a for a in dates_ba5 if a < Time.dateToCoordinate("2022-10-01") and a > Time.dateToCoordinate("2022-04-01")]
    dates_ba45 = sorted(list(set(dates_ba4 + dates_ba5)))
    dates_ba2 = [a for a in dates_ba2 if a < Time.dateToCoordinate("2022-10-01") and a > 44320] #before 2022-08-01
    dates_ba22 = [a for a in dates_ba22 if a < Time.dateToCoordinate("2022-10-01") and a > 44320] #before 2022-08-01
    dates_ba2 = sorted(list(set(dates_ba2 + dates_ba22)))
    if country == 'NY' or country == 'USA':
        dates_ba2 = [a for a in dates_ba2 if a < Time.dateToCoordinate("2022-10-01") and a > 44600] #before 2022-08-01


    tmin = min(set(dates_ba45).intersection(set(dates_ba2)))
    tmax = max(set(dates_ba45).intersection(set(dates_ba2)))
    t_range = np.arange(tmin,tmax)
    ba2_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts[pango2flupredict['BA.2']].keys():
            c+= np.exp(counts[pango2flupredict['BA.2']][str(t)])
        if str(t) in counts[pango2flupredict['BA.2.12.1']].keys():
            c+= np.exp(counts[pango2flupredict['BA.2.12.1']][str(t)])
        ba2_count.append(int(c))
    ba45_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts[pango2flupredict['BA.4']].keys():
            c += np.exp(counts[pango2flupredict['BA.4']][str(t)])
        if str(t) in counts[pango2flupredict['BA.5']].keys():
            c += np.exp(counts[pango2flupredict['BA.5']][str(t)])
        if str(t) in counts[pango2flupredict['BA.5.9']].keys():
            c += np.exp(counts[pango2flupredict['BA.5.9']][str(t)])
        ba45_count.append(int(c))

    N_tot = np.array(ba2_count) + np.array(ba45_count)
    check = 'not_okay'
    for t in t_range:
        x_ba45 = ba45_count[t-tmin] / N_tot[t-tmin]
        x_ba2 = ba2_count[t-tmin] / N_tot[t-tmin]
        if x_ba45 > x_limit:
            tminnew = t
            check = 'okay'
            break
    for t in t_range:
        x_ba45 = ba45_count[t-tmin] / N_tot[t-tmin]
        x_ba2 = ba2_count[t-tmin] / N_tot[t-tmin]
        if x_ba45 > 1 - x_limit:
            tmaxnew = t
            break
    tmin = tminnew
    tmax = tmaxnew
    t_range= np.arange(tmin,tmax)

    tmax = tmaxnew
    t_range= np.arange(tmin,tmax)

    ba2_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts[pango2flupredict['BA.2']].keys():
            c+= np.exp(counts[pango2flupredict['BA.2']][str(t)])
        if str(t) in counts[pango2flupredict['BA.2.12.1']].keys():
            c+= np.exp(counts[pango2flupredict['BA.2.12.1']][str(t)])
        ba2_count.append(int(c))

    ba45_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts[pango2flupredict['BA.4']].keys():
            c += np.exp(counts[pango2flupredict['BA.4']][str(t)])
        if str(t) in counts[pango2flupredict['BA.5']].keys():
            c += np.exp(counts[pango2flupredict['BA.5']][str(t)])
        if str(t) in counts[pango2flupredict['BA.5.9']].keys():
            c += np.exp(counts[pango2flupredict['BA.5.9']][str(t)])
        ba45_count.append(int(c))

    N_tot = np.array(ba2_count) + np.array(ba45_count)

    Ztot =np.array([int(np.exp(Z[str(t)])) for t in t_range])
    print(f"{country}, {min(Ztot)}")
    country_min_count[country].append(min(Ztot))
    if np.sum(Ztot > min_count) != len(Ztot):
        print(country)
        continue

    t1 = 0 
    t2 = int(t1 + dt)
    while t2 + tmin < tmax:
        FLAI_time = int((t1 + t2 + 2*tmin)/2)
        N_tot1 = ba2_count[t1] + ba45_count[t1]
        N_tot2 = ba2_count[t2] + ba45_count[t2]
        if ba2_count[t1] < 10 or ba2_count[t2] < 10 or ba45_count[t1] < 10 or ba45_count[t2] < 10:
            t1 = t1 + 7
            t2 = int(t1 + dt)
            print(country)
            continue
        p_t1 = [ba2_count[t1] / N_tot1, ba45_count[t1] / N_tot1]
        p_t2 = [ba2_count[t2] / N_tot2, ba45_count[t2] / N_tot2]

        x_ba2_hat_t1 = np.random.binomial(N_tot1,p_t1[0],1000) / N_tot1
        x_ba45_hat_t1 = np.ones(len(x_ba2_hat_t1)) - x_ba2_hat_t1
        x_ba2_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_ba2_hat_t1])
        x_ba45_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_ba45_hat_t1])

        x_ba2_hat_t2 = np.random.binomial(N_tot2,p_t2[0],1000) / N_tot2
        x_ba45_hat_t2 = np.ones(len(x_ba2_hat_t2)) - x_ba2_hat_t2
        x_ba2_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_ba2_hat_t2])
        x_ba45_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_ba45_hat_t2])

        result = (np.log(x_ba45_hat_t2 / x_ba2_hat_t2) - np.log(x_ba45_hat_t1 / x_ba2_hat_t1)) / dt
        s_hat = np.mean(result)
        s_hat_var = np.var(result)

        t_range_here = np.arange(t1 + tmin,t2 + tmin)
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.2']].keys():
            x_ba2 = freq_traj[pango2flupredict['BA.2']][str(FLAI_time)]

        else:
            x_ba2 = 0.0
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.2.12.1']].keys():
            x_ba2 += freq_traj[pango2flupredict['BA.2.12.1']][str(FLAI_time)]

        x_ba4 = 0.0
        x_ba5 = 0.0
        x_ba59 = 0.0
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.4']].keys():
            x_ba4 = freq_traj[pango2flupredict['BA.4']][str(FLAI_time)]
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.5']].keys():
            x_ba5 = freq_traj[pango2flupredict['BA.5']][str(FLAI_time)]
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.5.9']].keys():
            x_ba59 = freq_traj[pango2flupredict['BA.5.9']][str(FLAI_time)]
        x_ba45 = x_ba4 + x_ba5 + x_ba59 #- x_bq1 - x_bf7


        lines.append([country,FLAI_time, Time.coordinateToStringDate(int(t1+tmin)),Time.coordinateToStringDate(int(t2+tmin)), np.round(s_hat,3), 
        np.round(s_hat_var,7), ba2_count[t1], ba2_count[t2],ba45_count[t1],ba45_count[t2],tmin,tmax,x_ba2,x_ba45])

        t1 = t1 + 7
        t2 = int(t1 + dt)


lines = pd.DataFrame(lines,columns = ['country','FLAI_time','t1','t2','s_hat','s_var','ba2_count_t1','ba2_count_t2','ba45_count_t1','ba45_count_t2','tmin','tmax','x_ba2','x_ba45'])

savename = '../output/s_hat_ba45_ba2.txt'
lines.to_csv(savename,'\t',index=False)