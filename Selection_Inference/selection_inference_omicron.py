import sys
import numpy as np
import pandas as pd
from util.time import Time
import json
from collections import defaultdict
import glob
from scipy.interpolate import interp1d
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


VOCs = ['ALPHA', 'BETA', 'GAMMA', 'DELTA', 'EPSILON', 'KAPPA', 'LAMBDA']
countries = ['BELGIUM', 'CA', 'CANADA', 'FINLAND', 'FRANCE',
             'GERMANY', 'ITALY', 'NETHERLANDS', 'NORWAY', 'NY',
             'SPAIN', 'SWITZERLAND', 'USA', 'JAPAN']
x_limit = 0.01
folder1 = '../DATA/2022_04_26'
country_min_count = defaultdict(lambda: [])
files = glob.glob(f"{folder1}/freq_traj_*")
meta_df = pd.read_csv(f"{folder1}/clean_data.txt", sep='\t', index_col=False)
print("=====================Delta - Omicron ============================")
dt = 30.0
min_count = 500
country2immuno2time = defaultdict(lambda:
                                  defaultdict(lambda:
                                              defaultdict(lambda: 0.0)))
lines = []
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

    dates_delta = list(counts['DELTA'].keys())
    dates_omi = list(counts['OMICRON'].keys())
    dates_delta = [int(a) for a in dates_delta]
    dates_omi = [int(a) for a in dates_omi]
    dates_delta = sorted(dates_delta)
    dates_omi = sorted(dates_omi)
    dates_omi = [a for a in dates_omi if a < 44620 and a > 44469]
    dates_delta = [a for a in dates_delta if a < 44620 and a > 44377]
    tmin = min(set(dates_delta).intersection(set(dates_omi)))
    tmax = max(set(dates_delta).intersection(set(dates_omi)))
    t_range = np.arange(tmin, tmax)
    omi_count = [int(np.exp(counts['OMICRON'][str(a)])) for a in t_range]
    delta_count = [int(np.exp(counts['DELTA'][str(a)])) for a in t_range]
    N_tot = np.array(omi_count) + np.array(delta_count)
    check = 'not_okay'
    for t in t_range:
        x_d = delta_count[t - tmin] / N_tot[t - tmin]
        x_o = omi_count[t - tmin] / N_tot[t - tmin]
        if x_o > x_limit:
            tminnew = t
            check = 'okay'
            break
    tmaxnew = tmax
    for t in t_range:
        x_d = delta_count[t - tmin] / N_tot[t - tmin]
        x_o = omi_count[t - tmin] / N_tot[t - tmin]
        if x_o > 1 - x_limit:
            tmaxnew = t
            break
    tmin = tminnew
    tmax = tmaxnew
    t_range = np.arange(tmin, tmax)

    Ztot = np.array([int(np.exp(Z[str(t)])) for t in t_range])
    country_min_count[country].append(min(Ztot))

    meta_times = list(meta_country.index)
    t_0 = meta_times[0]
    vac_full = [meta_country.loc[t]['people_fully_vaccinated_per_hundred']
                / 100. for t in meta_times]
    vinterp = clean_vac(list(meta_country.index), vac_full)
    country2immuno2time[country]['VAC'] = {meta_times[i]: vinterp[i]
                                           for i in range(len(meta_times))}

    booster = [meta_country.loc[t]['total_boosters_per_hundred']
               / 100. for t in meta_times]
    if np.sum(np.isnan(booster)) == len(booster):
        boosterp = np.zeros(len(booster))
    else:
        boosterp = clean_vac(meta_times, booster)
    country2immuno2time[country]['BOOST'] = {meta_times[i]: boosterp[i]
                                             for i in range(len(meta_times))}
    cases_full = [meta_country.loc[t]['new_cases']
                  / meta_country.loc[t]['population'] for t in meta_times]
    cases_full = clean_vac(list(meta_country.index), cases_full)
    country2immuno2time[country]['CASES'] = {meta_times[i]: cases_full[i]
                                             for i in range(len(meta_times))}
    recov_tot = [[np.sum(cases_full[0:t - list(meta_country.index)[0]]), t]
                 for t in meta_times]
    country2immuno2time[country]['RECOV_TOT'] = {a[1]: a[0] for a in recov_tot}
    cor = Time.dateToCoordinate("2022-01-01")
    ytot = country2immuno2time[country]['RECOV_TOT'][cor]

    x_delta = []
    x_omi = []
    for t in list(meta_country.index):
        if str(t) in freq_traj['DELTA'].keys():
            x_delta.append(freq_traj['DELTA'][str(t)])
        else:
            x_delta.append(0.0)
        if str(t) in freq_traj['OMICRON'].keys()\
                and t > Time.dateToCoordinate("2021-10-01"):
            x_omi.append(freq_traj['OMICRON'][str(t)])
        else:
            x_omi.append(0.0)
    freq_delta = np.array(x_delta)
    freq_omi = np.array(x_omi)

    cases_full_delta = cases_full * freq_delta
    recov_delta = [[np.sum(cases_full_delta[0:t - t_0]), t]
                   for t in meta_times]
    country2immuno2time[country]['RECOV_DELTA_0'] = {a[1]: a[0]
                                                     for a in recov_delta}

    cases_full_omi = cases_full * freq_omi
    recov_omi = [[np.sum(cases_full_omi[0:t - t_0]), t] for t in meta_times]
    country2immuno2time[country]['RECOV_OMI_0'] = {a[1]: a[0]
                                                   for a in recov_omi}

    omi_count = [int(np.exp(counts['OMICRON'][str(a)])) for a in t_range]
    delta_count = [int(np.exp(counts['DELTA'][str(a)])) for a in t_range]

    t1 = 0
    t2 = int(t1 + dt)
    while t2 + tmin < tmax:
        FLAI_time = int((t1 + t2 + 2 * tmin) / 2)
        N_tot1 = omi_count[t1] + delta_count[t1]
        N_tot2 = omi_count[t2] + delta_count[t2]
        if omi_count[t1] < 10\
            or omi_count[t2] < 10\
            or delta_count[t1] < 10\
                or delta_count[t2] < 10:
            t1 = t1 + 7
            t2 = int(t1 + dt)
            continue
        p_t1 = [omi_count[t1] / N_tot1, delta_count[t1] / N_tot1]
        p_t2 = [omi_count[t2] / N_tot2, delta_count[t2] / N_tot2]

        x_omi_hat_t1 = np.random.binomial(N_tot1, p_t1[0], 1000) / N_tot1
        x_delta_hat_t1 = np.ones(len(x_omi_hat_t1)) - x_omi_hat_t1
        x_omi_hat_t1 = np.array([x if x != 0 else 1 / (N_tot1 + 1)
                                 for x in x_omi_hat_t1])
        x_delta_hat_t1 = np.array([x if x != 0 else 1 / (N_tot1 + 1)
                                   for x in x_delta_hat_t1])

        x_omi_hat_t2 = np.random.binomial(N_tot2, p_t2[0], 1000) / N_tot2
        x_delta_hat_t2 = np.ones(len(x_omi_hat_t2)) - x_omi_hat_t2
        x_omi_hat_t2 = np.array([x if x != 0 else 1 / (N_tot2 + 1)
                                 for x in x_omi_hat_t2])
        x_delta_hat_t2 = np.array([x if x != 0 else 1 / (N_tot2 + 1)
                                   for x in x_delta_hat_t2])

        result = (np.log(x_omi_hat_t2 / x_delta_hat_t2)
                  - np.log(x_omi_hat_t1 / x_delta_hat_t1)) / dt

        s_hat = np.mean(result)
        s_hat_var = np.var(result)

        for t in list(country2immuno2time[country]['VAC'].keys()):
            if country2immuno2time[country]['VAC'][t]\
                    < country2immuno2time[country]['BOOST'][FLAI_time]:
                country2immuno2time[country]['VAC_cor'][t] = 0.0
            else:
                country2immuno2time[country]['VAC_cor'][t]\
                    = country2immuno2time[country]['VAC'][t]\
                    - country2immuno2time[country]['BOOST'][FLAI_time]

        t_range_here = np.arange(t1 + tmin, t2 + tmin)
        vac_av = np.mean([country2immuno2time[country]['VAC'][int(t)]
                          for t in t_range_here])
        vac_cor_av = np.mean([country2immuno2time[country]['VAC_cor'][int(t)]
                              for t in t_range_here])
        recov = np.mean([country2immuno2time[country]['RECOV_DELTA_0'][int(t)]
                         for t in t_range_here])
        cases_av = np.mean([country2immuno2time[country]['CASES'][int(t)]
                            * 1000000
                            for t in t_range_here])
        boosted_av = np.mean([country2immuno2time[country]['BOOST'][int(t)]
                              for t in t_range_here])
        delta_recov\
            = np.mean([country2immuno2time[country]['RECOV_DELTA_0'][int(t)]
                       for t in t_range_here])
        omi_recov\
            = np.mean([country2immuno2time[country]['RECOV_OMI_0'][int(t)]
                       for t in t_range_here])

        if str(FLAI_time) in freq_traj['DELTA'].keys():
            x_delta = freq_traj['DELTA'][str(FLAI_time)]
        else:
            x_delta = 0.0
        if str(FLAI_time) in freq_traj['OMICRON'].keys():
            x_omi = freq_traj['OMICRON'][str(FLAI_time)]
        else:
            x_omi = 0.0

        lines.append([country, FLAI_time,
                      Time.coordinateToStringDate(int(t1 + tmin)),
                      Time.coordinateToStringDate(int(t2 + tmin)),
                      np.round(s_hat, 3), np.round(s_hat_var, 7),
                      np.round(vac_av, 3), np.round(vac_cor_av, 3),
                      np.round(recov, 3), np.round(boosted_av, 3),
                      np.round(delta_recov, 3), np.round(omi_recov, 3),
                      omi_count[t1], omi_count[t2],
                      delta_count[t1], delta_count[t2],
                      np.round(cases_av, 2), x_delta, x_omi])
        t1 = t1 + 7
        t2 = int(t1 + dt)

lines = pd.DataFrame(lines, columns=['country', 'FLAI_time', 't1', 't2',
                                     's_hat', 's_var', 'vaccinated',
                                     'vac_cor', 'recovered', 'boosted',
                                     'delta_recov', 'omi_recov',
                                     'omi_count_t1', 'omi_count_t2',
                                     'delta_count_t1', 'delta_count_t2',
                                     'av_cases', 'x_delta', 'x_omi'])
savename = '../output/s_hat_omi_delta.txt'
lines.to_csv(savename, sep='\t', index=False)
