import sys
import numpy as np
import pandas as pd
from util.time import Time
import json
from scipy.interpolate import interp1d
from collections import defaultdict
sys.path.insert(0, "..")


def clean_vac(times, vac_vec):
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
    v_func = interp1d(vac_times, vac_rep, fill_value='extrapolate')
    v_interp = v_func(times)
    return v_interp


df_da = pd.read_csv("../output/s_hat_delta_alpha.txt",
                    sep='\t', index_col=False)
df_od = pd.read_csv("../output/s_hat_omi_delta.txt",
                    sep='\t', index_col=False)
meta_df = pd.read_csv("../DATA/clean_data.txt",
                      sep='\t', index_col=False)

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
             '1C.2B.3J.4E.5C.6F': ' BN.1',
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
pango2flupredict = {a: b for b, a in WHOlabels.items()}

countries_da = sorted(list(set(df_da.country)))
countries_od = sorted(list(set(df_od.country)))
countries = sorted(list(set(countries_da).intersection(set(countries_od))))

x_limit = 0.01
dt = 30.0
min_count = 500
x_tot_min = 0.5
country2immuno2time = defaultdict(lambda:
                                  defaultdict(lambda:
                                              defaultdict(lambda: 0.0)))
lines = []
folder1 = '../DATA/2023_04_01'
for country in countries:
    with open(f"{folder1}/freq_traj_{country.upper()}.json", 'r') as f:
        freq_traj = json.load(f)
    with open(f"{folder1}/multiplicities_{country.upper()}.json", 'r') as f:
        counts = json.load(f)
    with open(f"{folder1}/multiplicities_Z_{country.upper()}.json", 'r') as f:
        Z = json.load(f)

    dates_bq1 = list(counts[pango2flupredict['BQ.1']].keys())
    dates_bq11 = list(counts[pango2flupredict['BQ.1.1']].keys())
    dates_ba4 = list(counts[pango2flupredict['BA.4']].keys())
    dates_ba5 = list(counts[pango2flupredict['BA.5']].keys())
    dates_bq1 = [int(a) for a in dates_bq1]
    dates_bq11 = [int(a) for a in dates_bq11]
    dates_ba4 = [int(a) for a in dates_ba4]
    dates_ba5 = [int(a) for a in dates_ba5]
    dates_bq1 = sorted(dates_bq1)
    dates_bq11 = sorted(dates_bq11)
    dates_ba4 = sorted(dates_ba4)
    dates_ba5 = sorted(dates_ba5)
    dates_ba4 = [a for a in dates_ba4
                 if a < Time.dateToCoordinate("2023-03-01") and a > 44569]
    dates_ba5 = [a for a in dates_ba5
                 if a < Time.dateToCoordinate("2023-03-01") and a > 44569]
    dates_ba45 = sorted(list(set(dates_ba4 + dates_ba5)))
    dates_bq1 = [a for a in dates_bq1
                 if a < Time.dateToCoordinate("2023-03-01") and a > 44753]
    dates_bq11 = [a for a in dates_bq11
                  if a < Time.dateToCoordinate("2023-03-01") and a > 44753]
    if country == 'NY' or country == 'USA':
        dates_bq1 = [a for a in dates_bq1
                     if a < Time.dateToCoordinate("2023-03-01") and a > 44600]
    dates_bq1 = sorted(list(set(dates_bq1 + dates_bq11)))

    tmin = min(set(dates_ba45).intersection(set(dates_bq1)))
    tmax = max(set(dates_ba45).intersection(set(dates_bq1)))

    t_range = np.arange(tmin, tmax)
    bq1_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts[pango2flupredict['BQ.1']].keys():
            c += np.exp(counts[pango2flupredict['BQ.1']][str(t)])
        if str(t) in counts[pango2flupredict['BQ.1.1']].keys():
            c += np.exp(counts[pango2flupredict['BQ.1.1']][str(t)])
        bq1_count.append(int(c))
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

    N_tot = np.array(bq1_count) + np.array(ba45_count)
    check = 'not_okay'
    for t in t_range:
        x_ba45 = ba45_count[t - tmin] / N_tot[t - tmin]
        x_bq1 = bq1_count[t - tmin] / N_tot[t - tmin]
        if x_bq1 > x_limit:
            tminnew = t
            check = 'okay'
            break

    tmaxnew = tmax
    for t in t_range:
        x_ba45 = ba45_count[t - tmin] / N_tot[t - tmin]
        x_bq1 = bq1_count[t - tmin] / N_tot[t - tmin]
        if x_bq1 > 1 - x_limit:
            tmaxnew = t
            break

    tmin = tminnew
    tmax = tmaxnew
    t_range = np.arange(tmin, tmax)
    bq1_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts[pango2flupredict['BQ.1']].keys():
            c += np.exp(counts[pango2flupredict['BQ.1']][str(t)])
        if str(t) in counts[pango2flupredict['BQ.1.1']].keys():
            c += np.exp(counts[pango2flupredict['BQ.1.1']][str(t)])
        bq1_count.append(int(c))
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

    N_tot = np.array(bq1_count) + np.array(ba45_count)

    ba45_freq = ba45_count / N_tot
    bq1_freq = bq1_count / N_tot

    Ztot = np.array([int(np.exp(Z[str(t)])) for t in t_range])
    if np.sum(Ztot > min_count) != len(Ztot):
        print(f"Low seq counts in {country}")
        continue

    t1 = 0
    t2 = int(t1 + dt)
    if len(t_range) < dt:
        t1 = 0
        t2 = len(t_range) - 1
    while t2 + tmin < tmax:
        FLAI_time = int((t1 + t2 + 2 * tmin) / 2)
        N_tot1 = bq1_count[t1] + ba45_count[t1]
        N_tot2 = bq1_count[t2] + ba45_count[t2]
        if bq1_count[t1] < 10\
            or bq1_count[t2] < 10\
            or ba45_count[t1] < 10\
                or ba45_count[t2] < 10:
            t1 = t1 + 7
            t2 = int(t1 + dt)
            continue
        p_t1 = [bq1_count[t1] / N_tot1, ba45_count[t1] / N_tot1]
        p_t2 = [bq1_count[t2] / N_tot2, ba45_count[t2] / N_tot2]

        x_bq1_hat_t1 = np.random.binomial(N_tot1, p_t1[0], 1000) / N_tot1
        x_ba45_hat_t1 = np.ones(len(x_bq1_hat_t1)) - x_bq1_hat_t1
        x_bq1_hat_t1 = np.array([x if x != 0 else 1 / (N_tot1 + 1)
                                 for x in x_bq1_hat_t1])
        x_ba45_hat_t1 = np.array([x if x != 0 else 1 / (N_tot1 + 1)
                                  for x in x_ba45_hat_t1])

        x_bq1_hat_t2 = np.random.binomial(N_tot2, p_t2[0], 1000) / N_tot2
        x_ba45_hat_t2 = np.ones(len(x_bq1_hat_t2)) - x_bq1_hat_t2
        x_bq1_hat_t2 = np.array([x if x != 0 else 1 / (N_tot2 + 1)
                                 for x in x_bq1_hat_t2])
        x_ba45_hat_t2 = np.array([x if x != 0 else 1 / (N_tot2 + 1)
                                  for x in x_ba45_hat_t2])

        result = (np.log(x_bq1_hat_t2 / x_ba45_hat_t2)
                  - np.log(x_bq1_hat_t1 / x_ba45_hat_t1)) / dt
        s_hat = np.mean(result)
        s_hat_var = np.var(result)

        t_range_here = np.arange(t1 + tmin, t2 + tmin)
        if str(FLAI_time) in freq_traj[pango2flupredict['BQ.1']].keys():
            x_bq1 = freq_traj[pango2flupredict['BQ.1']][str(FLAI_time)]
        else:
            x_bq1 = 0.0
        if str(FLAI_time) in freq_traj[pango2flupredict['BQ.1.1']].keys():
            x_bq11 = freq_traj[pango2flupredict['BQ.1.1']][str(FLAI_time)]
        else:
            x_bq11 = 0.0
        x_bq1 += x_bq11

        x_ba4 = 0.0
        x_ba5 = 0.0
        x_ba59 = 0.0
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.4']].keys():
            x_ba4 = freq_traj[pango2flupredict['BA.4']][str(FLAI_time)]
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.5']].keys():
            x_ba5 = freq_traj[pango2flupredict['BA.5']][str(FLAI_time)]
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.5.9']].keys():
            x_ba59 = freq_traj[pango2flupredict['BA.5.9']][str(FLAI_time)]
        x_ba45 = x_ba4 + x_ba5 + x_ba59

        lines.append([country, FLAI_time,
                      Time.coordinateToStringDate(int(t1 + tmin)),
                      Time.coordinateToStringDate(int(t2 + tmin)),
                      np.round(s_hat, 3), np.round(s_hat_var, 7),
                      bq1_count[t1], bq1_count[t2],
                      ba45_count[t1], ba45_count[t2],
                      tmin, tmax, x_bq1, x_ba45])
        t1 = t1 + 7
        t2 = int(t1 + dt)
lines = pd.DataFrame(lines, columns=['country', 'FLAI_time', 't1', 't2',
                                     's_hat', 's_var', 'bq11_count_t1',
                                     'bq11_count_t2', 'ba45_count_t1',
                                     'ba45_count_t2', 'tmin',
                                     'tmax', 'x_bq11', 'x_ba45'])

savename = '../output/s_hat_bq1_ba45.txt'
lines.to_csv(savename, sep='\t', index=False)
