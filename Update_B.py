import numpy as np
import pandas as pd
import sys
from util.time import Time
from sklearn import linear_model
from copy import copy
sys.path.insert(0, "..")

df_alpha = pd.read_csv("output/Data_alpha_shift.txt", sep='\t',
                       index_col=False)
df_delta = pd.read_csv("output/Data_delta_shift.txt", sep='\t',
                       index_col=False)
df_omi = pd.read_csv("output/Data_omi_shift.txt", sep='\t',
                     index_col=False)
df_ba2 = pd.read_csv("output/Data_ba2_shift.txt", sep='\t',
                     index_col=False)
df_ba45 = pd.read_csv("output/Data_ba45_shift.txt", sep='\t',
                      index_col=False)
df_bq1 = pd.read_csv("output/Data_bq1_shift.txt", sep='\t',
                     index_col=False)

df_alpha = df_alpha.sort_values(by=['country', 'time'])
df_delta = df_delta.sort_values(by=['country', 'time'])
df_omi = df_omi.sort_values(by=['country', 'time'])
df_ba2 = df_ba2.sort_values(by=['country', 'time'])
df_ba45 = df_ba45.sort_values(by=['country', 'time'])
df_bq1 = df_bq1.sort_values(by=['country', 'time'])


df_omi = df_omi.loc[[a != 'CA' for a in df_omi.country]]

# Add up a country-variance for all measurements.
country_variance = []
for c in list(set(df_delta.country)):
    meansvar = np.mean(df_delta.loc[df_delta.country == c, 's_var'])
    country_variance.append(meansvar)
median_svar_da = np.median(country_variance)

country_variance = []
for c in list(set(df_omi.country)):
    meansvar = np.mean(df_omi.loc[df_omi.country == c, 's_var'])
    country_variance.append(meansvar)
median_svar_od = np.median(country_variance)

country_variance = []
for c in list(set(df_ba2.country)):
    meansvar = np.mean(df_ba2.loc[df_ba2.country == c, 's_var'])
    country_variance.append(meansvar)
median_svar_ba2 = np.median(country_variance)

country_variance = []
for c in list(set(df_ba45.country)):
    meansvar = np.mean(df_ba45.loc[df_ba45.country == c, 's_var'])
    country_variance.append(meansvar)
median_svar_ba45 = np.median(country_variance)

country_variance = []
for c in list(set(df_bq1.country)):
    meansvar = np.mean(df_bq1.loc[df_bq1.country == c, 's_var'])
    country_variance.append(meansvar)
median_svar_bq1 = np.median(country_variance)


df_delta['s_var'] = np.array(df_delta['s_var'] + median_svar_da)
df_omi['s_var'] = np.array(df_omi['s_var'] + median_svar_od)
df_ba2['s_var'] = np.array(df_ba2['s_var'] + median_svar_ba2)
df_ba45['s_var'] = np.array(df_ba45['s_var'] + median_svar_ba45)
df_bq1['s_var'] = np.array(df_bq1['s_var'] + median_svar_bq1)

df_full = pd.DataFrame([], columns=['country',
                                    'time',
                                    's_hat',
                                    's_var',
                                    'dC_vac',
                                    'dC_inf'])
for line in df_ba45.itertuples():
    dC_vac = line.C_ba2_bst - line.C_ba45_bst
    dC_inf = (line.C_ba2_ba1 - line.C_ba45_ba1)\
        + (line.C_ba2_ba2 - line.C_ba45_ba2)\
        + (line.C_ba2_ba45 - line.C_ba45_ba45)
    new_row = pd.DataFrame([[line.country, line.time, line.s_hat,
                             line.s_var, dC_vac, dC_inf]],
                           columns=df_full.columns)
    df_full = pd.concat([new_row, df_full])
for line in df_bq1.itertuples():
    dC_vac = (line.C_ba45_bst - line.C_bq1_bst)\
        + (line.C_ba45_biv - line.C_bq1_biv)
    dC_inf = (line.C_ba45_ba1 - line.C_bq1_ba1)\
        + (line.C_ba45_ba2 - line.C_bq1_ba2)\
        + (line.C_ba45_ba45 - line.C_bq1_ba45)\
        + (line.C_ba45_bq1 - line.C_bq1_bq1)
    new_row = pd.DataFrame([[line.country, line.time, line.s_hat,
                             line.s_var, dC_vac, dC_inf]],
                           columns=df_full.columns)
    df_full = pd.concat([new_row, df_full])


time_window = 120  # days
lines = []
# first data 08-05-2022, update weekly
for time_cut_off in np.arange(Time.dateToCoordinate("2022-05-09"),
                              Time.dateToCoordinate("2023-03-01"),
                              1):
    dt = time_window - (time_cut_off - Time.dateToCoordinate("2022-05-08"))
    if dt > 0:
        ratio = dt / time_window
    else:
        ratio = 0

    if ratio == 0:
        df_data = copy(df_full.loc[(df_full.time < time_cut_off)
                                   & (df_full.time
                                      > time_cut_off - time_window)])
    else:
        df_data = copy(df_full.loc[df_full.time < time_cut_off])
    # only one parameter to fit..
    gamma_vac = 0.27859
    gamma_inf_old = 0.65718
    LR = linear_model.LinearRegression(fit_intercept=False)
    X = np.array(list(df_data.dC_inf))
    X = X.reshape((-1, 1))
    Y = list(df_data.s_hat - df_data.dC_vac * gamma_vac)
    LR.fit(X, Y, sample_weight=list(1 / df_data.s_var))
    gamma_inf = LR.coef_[0]
    gamma_inf_weighted = (1 - ratio) * gamma_inf + ratio * gamma_inf_old

    lines.append([time_cut_off,
                  str(Time.coordinateToDate(int(time_cut_off))),
                  len(df_data),
                  gamma_inf_weighted,
                  gamma_inf_weighted / gamma_vac])
lines = pd.DataFrame(lines,
                     columns=['time_cut_off',
                              'date_cut_off',
                              'data_points',
                              'gamma_inf_new',
                              'b_new'])
lines.to_csv("output/Update_gamma_inf.txt", '\t', index=False)
