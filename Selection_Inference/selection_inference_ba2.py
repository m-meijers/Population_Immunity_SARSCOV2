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


df_da = pd.read_csv("../output/s_hat_delta_alpha.txt", sep='\t',
                    index_col=False)
df_od = pd.read_csv("../output/s_hat_omi_delta.txt", sep='\t',
                    index_col=False)
meta_df = pd.read_csv("../DATA/clean_data.txt", sep='\t',
                      index_col=False)

countries_da = sorted(list(set(df_da.country)))
countries_od = sorted(list(set(df_od.country)))
countries = sorted(list(set(countries_da).intersection(set(countries_od))))

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

x_limit = 0.01
dt = 30.0
min_count = 500


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
    country_case = country[0] + country[1:].lower()
    meta_country = meta_df.loc[list(meta_df['location'] == country_case)]
    if len(country) <= 3:
        meta_country = meta_df.loc[list(meta_df['location'] == country)]
    meta_country.index = meta_country['FLAI_time']

    dates_ba2 = list(counts[pango2flupredict['BA.2']].keys())
    dates_ba1 = list(counts[pango2flupredict['BA.1']].keys())
    dates_ba11 = list(counts[pango2flupredict['BA.1.1']].keys())
    dates_ba2 = [int(a) for a in dates_ba2]
    dates_ba1 = [int(a) for a in dates_ba1]
    dates_ba11 = [int(a) for a in dates_ba11]
    dates_ba2 = sorted(dates_ba2)
    dates_ba1 = sorted(dates_ba1)
    dates_ba11 = sorted(dates_ba11)
    dates_ba1 = [a for a in dates_ba1
                 if a < 44773 and a > Time.dateToCoordinate("2021-12-01")]
    dates_ba11 = [a for a in dates_ba11
                  if a < 44773 and a > Time.dateToCoordinate("2021-12-01")]
    dates_ba2 = [a for a in dates_ba2
                 if a < 44773 and a > Time.dateToCoordinate("2021-12-01")]
    if country == 'NY' or country == 'USA':
        dates_ba2 = [a for a in dates_ba2 if a < 44773 and a > 44534]
    dates_ba1 = sorted(list(set(dates_ba1 + dates_ba11)))

    tmin = min(set(dates_ba2).intersection(set(dates_ba1)))
    tmax = max(set(dates_ba2).intersection(set(dates_ba1)))
    t_range = np.arange(tmin, tmax)
    ba1_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts[pango2flupredict['BA.1']].keys():
            c += np.exp(counts[pango2flupredict['BA.1']][str(t)])
        if str(t) in counts[pango2flupredict['BA.1.1']].keys():
            c += np.exp(counts[pango2flupredict['BA.1.1']][str(t)])
        ba1_count.append(int(c))
    ba2_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts[pango2flupredict['BA.2']].keys():
            c += np.exp(counts[pango2flupredict['BA.2']][str(t)])
        if str(t) in counts[pango2flupredict['BA.2.12.1']].keys():
            c += np.exp(counts[pango2flupredict['BA.2.12.1']][str(t)])
        ba2_count.append(int(c))
    N_tot = np.array(ba1_count) + np.array(ba2_count)
    check = 'not_okay'
    for t in t_range:
        x_ba2 = ba2_count[t - tmin] / N_tot[t - tmin]
        x_ba1 = ba1_count[t - tmin] / N_tot[t - tmin]
        if x_ba2 > x_limit:
            tminnew = t
            check = 'okay'
            break
    for t in t_range:
        x_ba2 = ba2_count[t - tmin] / N_tot[t - tmin]
        x_ba1 = ba1_count[t - tmin] / N_tot[t - tmin]
        if x_ba2 > 1 - x_limit:
            tmaxnew = t
            break
    tmin = tminnew
    tmax = tmaxnew
    t_range = np.arange(tmin, tmax)

    Ztot = np.array([int(np.exp(Z[str(t)])) for t in t_range])
    if np.sum(Ztot > min_count) != len(Ztot):
        print(f"{country} dropped due to insufficient count")
        continue

    ba1_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts[pango2flupredict['BA.1']].keys():
            c += np.exp(counts[pango2flupredict['BA.1']][str(t)])
        if str(t) in counts[pango2flupredict['BA.1.1']].keys():
            c += np.exp(counts[pango2flupredict['BA.1.1']][str(t)])
        ba1_count.append(int(c))
    ba2_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts[pango2flupredict['BA.2']].keys():
            c += np.exp(counts[pango2flupredict['BA.2']][str(t)])
        if str(t) in counts[pango2flupredict['BA.2.12.1']].keys():
            c += np.exp(counts[pango2flupredict['BA.2.12.1']][str(t)])
        ba2_count.append(int(c))

    Ztot = np.array([int(np.exp(Z[str(t)])) for t in t_range])
    if np.sum(Ztot > min_count) != len(Ztot):
        continue

    t1 = 0
    t2 = int(t1 + dt)
    while t2 + tmin < tmax:
        FLAI_time = int((t1 + t2 + 2 * tmin) / 2)
        N_tot1 = ba1_count[t1] + ba2_count[t1]
        N_tot2 = ba1_count[t2] + ba2_count[t2]
        if ba1_count[t1] < 10\
            or ba1_count[t2] < 10\
            or ba2_count[t1] < 10\
                or ba2_count[t2] < 10:
            t1 = t1 + 7
            t2 = int(t1 + dt)
            continue
        p_t1 = [ba1_count[t1] / N_tot1, ba2_count[t1] / N_tot1]
        p_t2 = [ba1_count[t2] / N_tot2, ba2_count[t2] / N_tot2]

        x_ba1_hat_t1 = np.random.binomial(N_tot1, p_t1[0], 1000) / N_tot1
        x_ba2_hat_t1 = np.ones(len(x_ba1_hat_t1)) - x_ba1_hat_t1
        x_ba1_hat_t1 = np.array([x if x != 0 else 1 / (N_tot1 + 1)
                                 for x in x_ba1_hat_t1])
        x_ba2_hat_t1 = np.array([x if x != 0 else 1 / (N_tot1 + 1)
                                 for x in x_ba2_hat_t1])

        x_ba1_hat_t2 = np.random.binomial(N_tot2, p_t2[0], 1000) / N_tot2
        x_ba2_hat_t2 = np.ones(len(x_ba1_hat_t2)) - x_ba1_hat_t2
        x_ba1_hat_t2 = np.array([x if x != 0 else 1 / (N_tot2 + 1)
                                 for x in x_ba1_hat_t2])
        x_ba2_hat_t2 = np.array([x if x != 0 else 1 / (N_tot2 + 1)
                                 for x in x_ba2_hat_t2])

        result = (np.log(x_ba2_hat_t2 / x_ba1_hat_t2)
                  - np.log(x_ba2_hat_t1 / x_ba1_hat_t1)) / dt
        s_hat = np.mean(result)
        s_hat_var = np.var(result)

        t_range_here = np.arange(t1 + tmin, t2 + tmin)
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.1']].keys():
            x_ba1 = freq_traj[pango2flupredict['BA.1']][str(FLAI_time)]
        else:
            x_ba1 = 0.0
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.1.1']].keys():
            x_ba11 = freq_traj[pango2flupredict['BA.1.1']][str(FLAI_time)]
        else:
            x_ba11 = 0.0
        x_ba1 += x_ba11
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.2']].keys():
            x_ba2 = freq_traj[pango2flupredict['BA.2']][str(FLAI_time)]
        if str(FLAI_time) in freq_traj[pango2flupredict['BA.2.12.1']].keys():
            x_ba2121 = freq_traj[pango2flupredict['BA.2.12.1']][str(FLAI_time)]
        x_ba2 += x_ba2121

        lines.append([country, FLAI_time,
                      Time.coordinateToStringDate(int(t1 + tmin)),
                      Time.coordinateToStringDate(int(t2 + tmin)),
                      np.round(s_hat, 3), np.round(s_hat_var, 7),
                      ba1_count[t1], ba1_count[t2], ba2_count[t1],
                      ba2_count[t2], tmin, tmax, x_ba1, x_ba2])
        t1 = t1 + 7
        t2 = int(t1 + dt)


lines = pd.DataFrame(lines, columns=['country', 'FLAI_time', 't1', 't2',
                                     's_hat', 's_var', 'ba1_count_t1',
                                     'ba1_count_t2', 'ba2_count_t1',
                                     'ba2_count_t2', 'tmin', 'tmax',
                                     'x_ba1', 'x_ba2'])

savename = '../output/s_hat_ba2_ba1.txt'
lines.to_csv(savename, '\t', index=False)
