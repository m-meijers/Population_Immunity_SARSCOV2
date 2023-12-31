import sys
import numpy as np
import pandas as pd
from util.time import Time
import json
from scipy.interpolate import interp1d
from collections import defaultdict
import glob
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
countries = ['BELGIUM', 'CA', 'CANADA', 'FINLAND', 'FRANCE', 'GERMANY',
             'ITALY', 'NETHERLANDS', 'NORWAY', 'NY', 'SPAIN',
             'SWITZERLAND', 'USA', 'JAPAN']
x_limit = 0.01
dt = 30.0
min_count = 500

country_min_count = defaultdict(lambda: [])
print("=====================Alpha - Delta ============================")
folder1 = '../DATA/2022_04_26"'
files = glob.glob(f"{folder1}/freq_traj_*")
meta_df = pd.read_csv(f"{folder1}/clean_data.txt", sep='\t', index_col=False)
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
    dates_alpha = list(counts['ALPHA'].keys())
    dates_delta = [int(a) for a in dates_delta]
    dates_alpha = [int(a) for a in dates_alpha]
    dates_delta = sorted(dates_delta)
    dates_alpha = sorted(dates_alpha)
    dates_alpha = [a for a in dates_alpha if a < 44470]
    dates_delta = [a for a in dates_delta if a > 44255]

    if country == 'ITALY':
        dates_delta = [a for a in dates_delta if a > 44287]
    if country == 'CA':
        dates_delta = [a for a in dates_delta if a > 44274]

    tmin = min(set(dates_delta).intersection(set(dates_alpha)))
    tmax = max(set(dates_delta).intersection(set(dates_alpha)))
    t_range = np.arange(tmin, tmax)
    alpha_count = [int(np.exp(counts['ALPHA'][str(a)])) for a in t_range]
    delta_count = [int(np.exp(counts['DELTA'][str(a)])) for a in t_range]
    N_tot = np.array(alpha_count) + np.array(delta_count)
    check = 'not_okay'
    for t in t_range:
        x_d = delta_count[t - tmin] / N_tot[t - tmin]
        x_a = alpha_count[t - tmin] / N_tot[t - tmin]
        if x_d > x_limit:
            tminnew = t
            check = 'okay'
            break
    for t in t_range:
        x_d = delta_count[t - tmin] / N_tot[t - tmin]
        x_a = alpha_count[t - tmin] / N_tot[t - tmin]
        if x_d > 1 - x_limit:
            tmaxnew = t
            break
    tmin = tminnew
    tmax = tmaxnew
    t_range = np.arange(tmin, tmax)

    x_alpha = []
    x_delta = []
    for t in list(meta_country.index):
        if str(t) in freq_traj['DELTA'].keys():
            x_delta.append(freq_traj['DELTA'][str(t)])
        else:
            x_delta.append(0.0)

        if str(t) in freq_traj['ALPHA'].keys():
            x_alpha.append(freq_traj['ALPHA'][str(t)])
        else:
            x_alpha.append(0.0)
    freq_delta = np.array(x_delta)
    freq_alpha = np.array(x_alpha)
    freq_wt = 1 - freq_delta - freq_alpha

    meta_times = list(meta_country.index)
    t_0 = meta_times[0]
    cases_full = [meta_country.loc[t]['new_cases']
                  / meta_country.loc[t]['population']
                  for t in meta_times]
    cases_full = clean_vac(meta_times, cases_full)

    cases_full_alpha = cases_full * freq_alpha
    cases_full_wt = cases_full * freq_wt
    cases_full_delta = cases_full * freq_delta
    recov_tot = [[np.sum(cases_full_alpha[t_range[0] - t_0:t - t_0]),
                  np.sum(cases_full_delta[t_range[0] - t_0:t - t_0]), t]
                 for t in t_range]
    recov_df = pd.DataFrame(recov_tot,
                            columns=['recov_alpha', 'recov_delta', 't'])
    recov_delta = [[np.sum(cases_full_delta[0:t - t_0]), t]
                   for t in meta_times]
    country2immuno2time[country]['RECOV_DELTA'] = {a[1]: a[0]
                                                   for a in recov_delta}
    recov_alpha = [[np.sum(cases_full_alpha[0:t - t_0]), t]
                   for t in meta_times]
    recov_wt = [[np.sum(cases_full_wt[0:t - t_0]), t]
                for t in meta_times]
    country2immuno2time[country]['RECOV_ALPHA'] = {a[1]: a[0]
                                                   for a in recov_alpha}
    country2immuno2time[country]['RECOV_WT'] = {a[1]: a[0] for a in recov_wt}
    recovered = [meta_country.loc[t]['total_cases']
                 / meta_country.loc[t]['population']
                 for t in t_range]
    cases = [meta_country.loc[t]['new_cases']
             / meta_country.loc[t]['population'] * 1000000
             for t in t_range]
    vaccinated = [meta_country.loc[t]['people_fully_vaccinated_per_hundred']
                  / 100. for t in t_range]

    vaccinated = clean_vac(t_range, vaccinated)
    country2immuno2time[country]['VAC'] = {t_range[i]: vaccinated[i]
                                           for i in range(len(t_range))}

    alpha_count = [int(np.exp(counts['ALPHA'][str(a)])) for a in t_range]
    delta_count = [int(np.exp(counts['DELTA'][str(a)])) for a in t_range]

    t1 = 0
    t2 = int(t1 + dt)
    while t2 + tmin < tmax:
        FLAI_time = int((t1 + t2 + 2 * tmin) / 2)
        N_tot1 = alpha_count[t1] + delta_count[t1]
        N_tot2 = alpha_count[t2] + delta_count[t2]
        if alpha_count[t1] < 10\
            or alpha_count[t2] < 10\
            or delta_count[t1] < 10\
                or delta_count[t2] < 10:
            t1 = t1 + 7
            t2 = int(t1 + dt)
            continue
        p_t1 = [alpha_count[t1] / N_tot1, delta_count[t1] / N_tot1]
        p_t2 = [alpha_count[t2] / N_tot2, delta_count[t2] / N_tot2]

        x_alpha_hat_t1 = np.random.binomial(N_tot1, p_t1[0], 1000) / N_tot1
        x_delta_hat_t1 = np.ones(len(x_alpha_hat_t1)) - x_alpha_hat_t1
        x_alpha_hat_t1 = np.array([x if x != 0 else 1 / (N_tot1 + 1)
                                   for x in x_alpha_hat_t1])
        x_delta_hat_t1 = np.array([x if x != 0 else 1 / (N_tot1 + 1)
                                   for x in x_delta_hat_t1])

        x_alpha_hat_t2 = np.random.binomial(N_tot2, p_t2[0], 1000) / N_tot2
        x_delta_hat_t2 = np.ones(len(x_alpha_hat_t2)) - x_alpha_hat_t2
        x_alpha_hat_t2 = np.array([x if x != 0 else 1 / (N_tot2 + 1)
                                   for x in x_alpha_hat_t2])
        x_delta_hat_t2 = np.array([x if x != 0 else 1 / (N_tot2 + 1)
                                   for x in x_delta_hat_t2])

        result = (np.log(x_delta_hat_t2 / x_alpha_hat_t2)
                  - np.log(x_delta_hat_t1 / x_alpha_hat_t1)) / dt
        s_hat = np.mean(result)
        s_hat_var = np.var(result)

        vac_av = np.mean(vaccinated[t1:t2])
        recov = np.mean(recovered[t1:t2])
        cases_av = np.mean(cases[t1:t2])
        t_range_here = np.arange(t1 + tmin, t2 + tmin)
        t_range_here = [int(t) for t in t_range_here]
        delta_recov = np.mean([country2immuno2time[country]['RECOV_DELTA'][t]
                               for t in t_range_here])
        alpha_recov = np.mean([country2immuno2time[country]['RECOV_ALPHA'][t]
                               for t in t_range_here])
        t_range_here = np.arange(t1 + tmin, t2 + tmin)

        if str(FLAI_time) in freq_traj['DELTA'].keys():
            x_delta = freq_traj['DELTA'][str(FLAI_time)]
        else:
            x_delta = 0.0

        if str(FLAI_time) in freq_traj['ALPHA'].keys():
            x_alpha = freq_traj['ALPHA'][str(FLAI_time)]
        else:
            x_alpha = 0.0
        freq_wt = 1 - x_delta - x_alpha

        lines.append([country, FLAI_time,
                      Time.coordinateToStringDate(int(t1 + tmin)),
                      Time.coordinateToStringDate(int(t2 + tmin)),
                      np.round(s_hat, 3), np.round(s_hat_var, 7),
                      np.round(vac_av, 3), np.round(recov, 3),
                      delta_recov, alpha_recov, alpha_count[t1],
                      alpha_count[t2], delta_count[t1], delta_count[t2],
                      np.round(cases_av, 2), freq_wt, x_delta, x_alpha])
        t1 = t1 + 7
        t2 = int(t1 + dt)
lines = pd.DataFrame(lines, columns=['country', 'FLAI_time', 't1', 't2',
                                     's_hat', 's_var', 'vaccinated',
                                     'recovered', 'delta_recov',
                                     'alpha_recov', 'alpha_count_t1',
                                     'alpha_count_t2', 'delta_count_t1',
                                     'delta_count_t2', 'av_cases',
                                     'x_wt', 'x_delta', 'x_alpha'])
savename = '../output/s_hat_delta_alpha.txt'
lines.to_csv(savename, '\t', index=False)
