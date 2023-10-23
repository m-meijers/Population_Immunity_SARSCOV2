import numpy as np
import pandas as pd
import sys
import scipy.stats as ss
from cvxopt import solvers
from cvxopt import matrix

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

all_countries = sorted(list(set(df_delta.country)))\
    + sorted(list(set(df_omi.country)))\
    + sorted(list(set(df_ba2.country)))\
    + sorted(list(set(df_ba45.country)))\
    + sorted(list(set(df_bq1.country)))
num_data = len(df_delta) + len(df_omi) + len(df_ba2)\
    + len(df_ba45) + len(df_bq1)
num_data_early = len(df_delta) + len(df_omi)
num_data_late = len(df_ba2) + len(df_ba45) + len(df_bq1)
L_result = []
BIC_result = []
# ==========================================================================
# Infer on Alpha-Delta and Delta-Omicron cross-overs. TIME INDEPENDENT ONLY
# ==========================================================================
lendelta = len(set(df_delta.country))
lenomi = len(set(df_omi.country))
delta_countries = sorted(list(set(df_delta.country)))
omi_countries = sorted(list(set(df_omi.country)))
delta_country2index = {c: int(i) for c, i in
                       zip(delta_countries, np.arange(len(delta_countries)))}
omi_country2index = {c: int(i) for c, i in
                     zip(omi_countries, np.arange(len(omi_countries)))}

n = lendelta + lenomi  # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(1, n))  # constraints no constrains yet...
b = matrix(0.0, size=(1, 1))  # Constraints no constrains yet...

Xdata = []
for line in df_delta.itertuples():
    W = 1 / line.s_var
    idx = delta_country2index[line.country]
    Q[idx, idx] += 2 * W
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_omi.itertuples():
    W = 1 / line.s_var
    idx = omi_country2index[line.country] + lendelta
    Q[idx, idx] += 2 * W
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xdata.append(Xvec)

sol = solvers.qp(Q, P)
p = np.array(list(sol['x']))

s_model = np.matmul(p, np.transpose(Xdata))
var = list(df_delta.s_var) + list(df_omi.s_var)
Y = list(df_delta.s_hat) + list(df_omi.s_hat)
L_s0 = np.sum(ss.norm.logpdf(np.array(s_model), np.array(Y),
                             np.sqrt(np.array(var))))

L_delta_s0 = np.sum(ss.norm.logpdf(np.array(s_model[:len(df_delta)]),
                                   np.array(Y[:len(df_delta)]),
                                   np.sqrt(np.array(var[:len(df_delta)]))))
L_omi_s0 = np.sum(ss.norm.logpdf(np.array(s_model[len(df_delta):]),
                                 np.array(Y[len(df_delta):]),
                                 np.sqrt(np.array(var[len(df_delta):]))))

# Validation on later shifts
lenba2 = len(set(df_ba2.country))
lenba45 = len(set(df_ba45.country))
lenbq1 = len(set(df_bq1.country))
ba2_countries = sorted(list(set(df_ba2.country)))
ba45_countries = sorted(list(set(df_ba45.country)))
bq1_countries = sorted(list(set(df_bq1.country)))
ba2_country2index = {c: int(i) for c, i in zip(ba2_countries,
                                               np.arange(len(ba2_countries)))}
ba45_country2index = {c: int(i) for c, i
                      in zip(ba45_countries,
                             np.arange(len(ba45_countries)))}
bq1_country2index = {c: int(i) for c, i in zip(bq1_countries,
                                               np.arange(len(bq1_countries)))}

n = lenba2 + lenba45 + lenbq1  # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(1, n))  # constraints no constrains yet...
b = matrix(0.0, size=(1, 1))  # Constraints no constrains yet...
Xdata = []

for line in df_ba2.itertuples():
    W = 1 / line.s_var
    idx = ba2_country2index[line.country]
    Q[idx, idx] += 2 * W
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_ba45.itertuples():
    W = 1 / line.s_var
    idx = ba45_country2index[line.country] + lenba2
    Q[idx, idx] += 2 * W
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_bq1.itertuples():
    W = 1 / line.s_var
    idx = bq1_country2index[line.country] + lenba2 + lenba45
    Q[idx, idx] += 2 * W
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xdata.append(Xvec)

sol = solvers.qp(Q, P)
p_val = np.array(list(sol['x']))

s_model_val = np.matmul(p_val, np.transpose(Xdata))
var_val = list(df_ba2.s_var) + list(df_ba45.s_var) + list(df_bq1.s_var)
Y_val = list(df_ba2.s_hat) + list(df_ba45.s_hat) + list(df_bq1.s_hat)
N_ba2 = len(df_ba2)
N_ba45 = N_ba2 + len(df_ba45)
L_s0_val = np.sum(ss.norm.logpdf(np.array(s_model_val),
                                 np.array(Y_val),
                                 np.sqrt(np.array(var_val))))
L_ba2_s0 = np.sum(ss.norm.logpdf(np.array(s_model_val[:len(df_ba2)]),
                                 np.array(Y_val[:len(df_ba2)]),
                                 np.sqrt(np.array(var_val[:len(df_ba2)]))))
L_ba45_s0 = np.sum(ss.norm.logpdf(np.array(s_model_val[N_ba2:N_ba45]),
                                  np.array(Y_val[N_ba2:N_ba45]),
                                  np.sqrt(np.array(var_val[N_ba2:N_ba45]))))
L_bq1_s0 = np.sum(ss.norm.logpdf(np.array(s_model_val[N_ba45:]),
                                 np.array(Y_val[N_ba45:]),
                                 np.sqrt(np.array(var_val[N_ba45:]))))

s0_delta = list(p[:lendelta])
s0_omi = list(p[lendelta:])
s0_ba2 = list(p_val[:lenba2])
s0_ba45 = list(p_val[lenba2:lenba2 + lenba45])
s0_bq1 = list(p_val[lenba2 + lenba45:])


BIC_early = (lendelta + lenomi) * np.log(num_data_early) - 2 * (L_s0)
BIC_late = (lenba2 + lenba45 + lenbq1) * np.log(num_data_late)\
    - 2 * (L_s0_val)
L_result.append([L_delta_s0, L_omi_s0, L_s0, L_ba2_s0, L_ba45_s0, L_bq1_s0,
                 L_s0 + L_s0_val, BIC_early, BIC_early + BIC_late,
                 0.0, 0.0, 0.0, 0.0, 0.0,
                 np.mean(s0_delta),
                 np.mean(s0_omi),
                 np.mean(s0_ba2),
                 np.mean(s0_ba45),
                 np.mean(s0_bq1)])


# ============================================================================
# Infer on Alpha-Delta and Delta-Omicron cross-overs
# ============================================================================
lendelta = len(set(df_delta.country))
lenomi = len(set(df_omi.country))
delta_countries = sorted(list(set(df_delta.country)))
omi_countries = sorted(list(set(df_omi.country)))
delta_country2index = {c: int(i) for c, i
                       in zip(delta_countries,
                              np.arange(len(delta_countries)))}
omi_country2index = {c: int(i) for c, i in zip(omi_countries,
                                               np.arange(len(omi_countries)))}

n = 3 + lendelta + lenomi  # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(1, n))  # constraints no constrains yet...
b = matrix(0.0, size=(1, 1))  # Constraints no constrains yet...
G = matrix(0.0, size=(n, n))
h = matrix(0.0, size=(n, 1))
kappa_da = 2.
Xdata = []
for line in df_delta.itertuples():
    W = 1 / line.s_var
    idx = 3 + delta_country2index[line.country]

    dC_vac = (line.C_alpha_vac - line.C_delta_vac)\
        + kappa_da * (line.C_alpha_alpha - line.C_delta_alpha)\
        + kappa_da * (line.C_alpha_delta - line.C_delta_delta)

    Q[0, 0] += 2 * W * dC_vac * dC_vac
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_vac * 1
    Q[idx, 0] += 2 * W * dC_vac * 1

    # linear terms
    P[0] += -2 * W * dC_vac * line.s_hat
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xvec[0] = dC_vac
    Xdata.append(np.array(Xvec))

for line in df_omi.itertuples():
    W = 1 / line.s_var
    idx = 3 + omi_country2index[line.country] + lendelta

    dC_vac = (line.C_delta_vac - line.C_omi_vac)\
        + (line.C_delta_bst - line.C_omi_bst)
    dC_inf = (line.C_delta_omi - line.C_omi_omi)\
        + (line.C_delta_delta - line.C_omi_delta)\
        + (line.C_delta_alpha - line.C_omi_alpha)

    Q[1, 1] += 2 * W * dC_vac * dC_vac
    Q[2, 2] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W

    # cross terms
    Q[1, idx] += 2 * W * dC_vac * 1
    Q[idx, 1] += 2 * W * dC_vac * 1
    Q[2, idx] += 2 * W * dC_inf * 1
    Q[idx, 2] += 2 * W * dC_inf * 1
    Q[1, 2] += 2 * W * dC_vac * dC_inf
    Q[2, 1] += 2 * W * dC_vac * dC_inf

    # linear terms
    P[1] += -2 * W * dC_vac * line.s_hat
    P[2] += -2 * W * dC_inf * line.s_hat
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xvec[1] = dC_vac
    Xvec[2] = dC_inf
    Xdata.append(np.array(Xvec))

# Fill in non-negativity constraints for s_0:
for i in range(lendelta + lenomi):
    G[i + 3, i + 3] = -1

sol = solvers.qp(Q, P, G=G, h=h)
p = np.array(list(sol['x']))

s_model = np.matmul(p, np.transpose(Xdata))
var = list(df_delta.s_var) + list(df_omi.s_var)
Y = list(df_delta.s_hat) + list(df_omi.s_hat)
L = np.sum(ss.norm.logpdf(np.array(s_model),
                          np.array(Y),
                          np.sqrt(np.array(var))))

L_delta = np.sum(ss.norm.logpdf(np.array(s_model[:len(df_delta)]),
                                np.array(Y[:len(df_delta)]),
                                np.sqrt(np.array(var[:len(df_delta)]))))
L_omi = np.sum(ss.norm.logpdf(np.array(s_model[len(df_delta):]),
                              np.array(Y[len(df_delta):]),
                              np.sqrt(np.array(var[len(df_delta):]))))

# Validation on later shifts
lenba2 = len(set(df_ba2.country))
lenba45 = len(set(df_ba45.country))
lenbq1 = len(set(df_bq1.country))
ba2_countries = sorted(list(set(df_ba2.country)))
ba45_countries = sorted(list(set(df_ba45.country)))
bq1_countries = sorted(list(set(df_bq1.country)))
ba2_country2index = {c: int(i) for c, i in
                     zip(ba2_countries, np.arange(len(ba2_countries)))}
ba45_country2index = {c: int(i) for c, i in
                      zip(ba45_countries, np.arange(len(ba45_countries)))}
bq1_country2index = {c: int(i) for c, i in
                     zip(bq1_countries, np.arange(len(bq1_countries)))}

n = lenba2 + lenba45 + lenbq1  # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(1, n))  # constraints no constrains yet...
b = matrix(0.0, size=(1, 1))  # Constraints no constrains yet...
Xdata = []

gamma_vac = p[1]
gamma_inf = p[2]
cor_list = []
for line in df_ba2.itertuples():
    W = 1 / line.s_var
    idx = ba2_country2index[line.country]
    cor = gamma_vac * (line.C_ba1_bst - line.C_ba2_bst)\
        + gamma_inf * (line.C_ba1_ba1 - line.C_ba2_ba1)\
        + gamma_inf * (line.C_ba1_ba2 - line.C_ba2_ba2)
    cor_list.append(cor)
    y = line.s_hat - cor
    Q[idx, idx] += 2 * W
    P[idx] += -2 * W * y
    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_ba45.itertuples():
    W = 1 / line.s_var
    idx = ba45_country2index[line.country] + lenba2
    cor = gamma_vac * (line.C_ba2_bst - line.C_ba45_bst)\
        + gamma_inf * (line.C_ba2_ba1 - line.C_ba45_ba1)\
        + gamma_inf * (line.C_ba2_ba2 - line.C_ba45_ba2)\
        + gamma_inf * (line.C_ba2_ba45 - line.C_ba45_ba45)
    cor_list.append(cor)
    y = line.s_hat - cor
    Q[idx, idx] += 2 * W
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_bq1.itertuples():
    W = 1 / line.s_var
    idx = bq1_country2index[line.country] + lenba2 + lenba45
    cor = gamma_vac * (line.C_ba45_bst - line.C_bq1_bst
                       + line.C_ba45_biv - line.C_bq1_biv)\
        + gamma_inf * (line.C_ba45_ba1 - line.C_bq1_ba1
                       + line.C_ba45_ba2 - line.C_bq1_ba2
                       + line.C_ba45_ba45 - line.C_bq1_ba45
                       + line.C_ba45_bq1 - line.C_bq1_bq1)
    cor_list.append(cor)
    y = line.s_hat - cor
    Q[idx, idx] += 2 * W
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xdata.append(Xvec)

sol = solvers.qp(Q, P)
p_val = np.array(list(sol['x']))

s_model_val = np.matmul(p_val, np.transpose(Xdata))
s_model_val += np.array(cor_list)
var_val = list(df_ba2.s_var) + list(df_ba45.s_var) + list(df_bq1.s_var)
Y_val = list(df_ba2.s_hat) + list(df_ba45.s_hat) + list(df_bq1.s_hat)

N_ba2 = len(df_ba2)
N_ba45 = N_ba2 + len(df_ba45)

L_val = np.sum(ss.norm.logpdf(np.array(s_model_val),
                              np.array(Y_val),
                              np.sqrt(np.array(var_val))))
L_ba2 = np.sum(ss.norm.logpdf(np.array(s_model_val[:N_ba2]),
                              np.array(Y_val[:N_ba2]),
                              np.sqrt(np.array(var_val[:N_ba2]))))
L_ba45 = np.sum(ss.norm.logpdf(np.array(s_model_val[N_ba2:N_ba45]),
                               np.array(Y_val[N_ba2:N_ba45]),
                               np.sqrt(np.array(var_val[N_ba2:N_ba45]))))
L_bq1 = np.sum(ss.norm.logpdf(np.array(s_model_val[N_ba45:]),
                              np.array(Y_val[N_ba45:]),
                              np.sqrt(np.array(var_val[N_ba45:]))))

s0_delta = list(p[3:3 + lendelta])
s0_omi = list(p[3 + lendelta:])
s0_ba2 = list(p_val[:lenba2])
s0_ba45 = list(p_val[lenba2:lenba2 + lenba45])
s0_bq1 = list(p_val[lenba2 + lenba45:])

BIC = (lendelta + lenomi + lenba2 + lenba45 + lenbq1 + 3) * np.log(num_data)\
    - 2 * (L_val + L)
BIC_early = (lendelta + lenomi + 3) * np.log(num_data_early) - 2 * (L)
BIC_late = (lenba2 + lenba45 + lenbq1) * np.log(num_data_late) - 2 * (L_val)
L_result.append([L_delta, L_omi, L, L_ba2, L_ba45, L_bq1, L + L_val,
                 BIC_early, BIC_early + BIC_late, p[0], p[1], p[2],
                 0.0, 0.0,
                 np.mean(s0_delta),
                 np.mean(s0_omi),
                 np.mean(s0_ba2),
                 np.mean(s0_ba45),
                 np.mean(s0_bq1)])

# ==============================================================================
# Infer on Alpha-Delta and Delta-Omicron cross-overs.
# One additional parameter for gamma_inf for later shifts. Mean s0 must be zero
# ==============================================================================
lendelta = len(set(df_delta.country))
lenomi = len(set(df_omi.country))
delta_countries = sorted(list(set(df_delta.country)))
omi_countries = sorted(list(set(df_omi.country)))
delta_country2index = {c: int(i) for c, i in
                       zip(delta_countries, np.arange(len(delta_countries)))}
omi_country2index = {c: int(i) for c, i in
                     zip(omi_countries, np.arange(len(omi_countries)))}

n = 3 + lendelta + lenomi  # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(1, n))
b = matrix(0.0, size=(1, 1))
G = matrix(0.0, size=(n, n))
h = matrix(0.0, size=(n, 1))
kappa_da = 2.
Xdata = []
for line in df_delta.itertuples():
    W = 1 / line.s_var
    idx = 3 + delta_country2index[line.country]

    dC_vac = (line.C_alpha_vac - line.C_delta_vac)\
        + kappa_da * (line.C_alpha_alpha - line.C_delta_alpha)\
        + kappa_da * (line.C_alpha_delta - line.C_delta_delta)

    Q[0, 0] += 2 * W * dC_vac * dC_vac
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_vac * 1
    Q[idx, 0] += 2 * W * dC_vac * 1

    # linear terms
    P[0] += -2 * W * dC_vac * line.s_hat
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xvec[0] = dC_vac
    Xdata.append(np.array(Xvec))

for line in df_omi.itertuples():
    W = 1 / line.s_var
    idx = 3 + omi_country2index[line.country] + lendelta

    dC_vac = (line.C_delta_vac - line.C_omi_vac)\
        + (line.C_delta_bst - line.C_omi_bst)
    dC_inf = (line.C_delta_omi - line.C_omi_omi)\
        + (line.C_delta_delta - line.C_omi_delta)\
        + (line.C_delta_alpha - line.C_omi_alpha)

    Q[1, 1] += 2 * W * dC_vac * dC_vac
    Q[2, 2] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W

    # cross terms
    Q[1, idx] += 2 * W * dC_vac * 1
    Q[idx, 1] += 2 * W * dC_vac * 1
    Q[2, idx] += 2 * W * dC_inf * 1
    Q[idx, 2] += 2 * W * dC_inf * 1
    Q[1, 2] += 2 * W * dC_vac * dC_inf
    Q[2, 1] += 2 * W * dC_vac * dC_inf

    # linear terms
    P[1] += -2 * W * dC_vac * line.s_hat
    P[2] += -2 * W * dC_inf * line.s_hat
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xvec[1] = dC_vac
    Xvec[2] = dC_inf
    Xdata.append(np.array(Xvec))

# Fill in non-negativity constraints for s_0:
for i in range(lendelta + lenomi):
    G[i + 3, i + 3] = -1

sol = solvers.qp(Q, P, G=G, h=h)
p = np.array(list(sol['x']))

s_model = np.matmul(p, np.transpose(Xdata))
var = list(df_delta.s_var) + list(df_omi.s_var)
Y = list(df_delta.s_hat) + list(df_omi.s_hat)
L = np.sum(ss.norm.logpdf(np.array(s_model),
                          np.array(Y),
                          np.sqrt(np.array(var))))

L_delta = np.sum(ss.norm.logpdf(np.array(s_model[:len(df_delta)]),
                                np.array(Y[:len(df_delta)]),
                                np.sqrt(np.array(var[:len(df_delta)]))))
L_omi = np.sum(ss.norm.logpdf(np.array(s_model[len(df_delta):]),
                              np.array(Y[len(df_delta):]),
                              np.sqrt(np.array(var[len(df_delta):]))))

# Inference on later shifts
lenba2 = len(set(df_ba2.country))
lenba45 = len(set(df_ba45.country))
lenbq1 = len(set(df_bq1.country))
ba2_countries = sorted(list(set(df_ba2.country)))
ba45_countries = sorted(list(set(df_ba45.country)))
bq1_countries = sorted(list(set(df_bq1.country)))
ba2_country2index = {c: int(i) for c, i in
                     zip(ba2_countries, np.arange(len(ba2_countries)))}
ba45_country2index = {c: int(i) for c, i in
                      zip(ba45_countries, np.arange(len(ba45_countries)))}
bq1_country2index = {c: int(i) for c, i in
                     zip(bq1_countries, np.arange(len(bq1_countries)))}

n = 1 + lenba2 + lenba45 + lenbq1  # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(2, n))  # constraints no constrains yet...
b = matrix(0.0, size=(2, 1))  # Constraints no constrains yet...
G = matrix(0.0, size=(n, n))
h = matrix(0.0, size=(n, 1))
Xdata = []

gamma_vac = p[1]
gamma_inf = p[2]
y_cor_list = []
for line in df_ba2.itertuples():
    W = 1 / line.s_var
    idx = 1 + ba2_country2index[line.country]
    cor = gamma_vac * (line.C_ba1_bst - line.C_ba2_bst)\
        + gamma_inf * (line.C_ba1_ba1 - line.C_ba2_ba1
                       + line.C_ba1_ba2 - line.C_ba2_ba2)
    y_cor_list.append(cor)
    dC_inf = 0.0
    y = line.s_hat - cor

    Q[0, 0] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_inf * 1
    Q[idx, 0] += 2 * W * dC_inf * 1

    # linear terms
    P[0] += -2 * W * dC_inf * y
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[0] = dC_inf
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_ba45.itertuples():
    W = 1 / line.s_var
    idx = 1 + ba45_country2index[line.country] + lenba2
    cor = gamma_vac * (line.C_ba2_bst - line.C_ba45_bst)
    y_cor_list.append(cor)
    dC_inf = (line.C_ba2_ba1 - line.C_ba45_ba1)\
        + (line.C_ba2_ba2 - line.C_ba45_ba2)\
        + (line.C_ba2_ba45 - line.C_ba45_ba45)
    y = line.s_hat - cor
    Q[0, 0] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_inf * 1
    Q[idx, 0] += 2 * W * dC_inf * 1

    # linear terms
    P[0] += -2 * W * dC_inf * y
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[0] = dC_inf
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_bq1.itertuples():
    W = 1 / line.s_var
    idx = 1 + bq1_country2index[line.country] + lenba2 + lenba45
    cor = gamma_vac * (line.C_ba45_bst - line.C_bq1_bst
                       + line.C_ba45_biv - line.C_bq1_biv)
    y_cor_list.append(cor)
    dC_inf = (line.C_ba45_ba1 - line.C_bq1_ba1
              + line.C_ba45_ba2 - line.C_bq1_ba2
              + line.C_ba45_ba45 - line.C_bq1_ba45
              + line.C_ba45_bq1 - line.C_bq1_bq1)
    y = line.s_hat - cor
    Q[0, 0] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_inf * 1
    Q[idx, 0] += 2 * W * dC_inf * 1

    # linear terms
    P[0] += -2 * W * dC_inf * y
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[0] = dC_inf
    Xvec[idx] = 1
    Xdata.append(Xvec)

# Fill in non-negativity constraints for s_0 for ba2:
for i in range(lenba2):  # + lenba45 + lenbq1):
    G[i + 1, i + 1] = -1

# Fill constraint mean s_0 for ba45 mean s_0 for bq1 = 0:
for i in range(lenba45):
    A[0, 1 + lenba2 + i] = 1 / (lenba45)

for i in range(lenbq1):
    A[1, 1 + lenba2 + lenba45 + i] = 1 / (lenbq1)


sol = solvers.qp(Q, P, G=G, h=h, A=A, b=b)
p_val = np.array(list(sol['x']))

s_model_val = np.matmul(p_val, np.transpose(Xdata))
var_val = list(df_ba2.s_var) + list(df_ba45.s_var) + list(df_bq1.s_var)
Y_val = list(df_ba2.s_hat) + list(df_ba45.s_hat) + list(df_bq1.s_hat)
Y_val = np.array(Y_val) - np.array(y_cor_list)


N_ba2 = len(df_ba2)
N_ba45 = N_ba2 + len(df_ba45)
L_val = np.sum(ss.norm.logpdf(np.array(s_model_val),
                              np.array(Y_val),
                              np.sqrt(np.array(var_val))))
L_ba2 = np.sum(ss.norm.logpdf(np.array(s_model_val[:N_ba2]),
                              np.array(Y_val[:N_ba2]),
                              np.sqrt(np.array(var_val[:N_ba2]))))
L_ba45 = np.sum(ss.norm.logpdf(np.array(s_model_val[N_ba2:N_ba45]),
                               np.array(Y_val[N_ba2:N_ba45]),
                               np.sqrt(np.array(var_val[N_ba2:N_ba45]))))
L_bq1 = np.sum(ss.norm.logpdf(np.array(s_model_val[N_ba45:]),
                              np.array(Y_val[N_ba45:]),
                              np.sqrt(np.array(var_val[N_ba45:]))))

s0_delta = list(p[3:3 + lendelta])
s0_omi = list(p[3 + lendelta:])
s0_ba2 = list(p_val[1:1 + lenba2])
s0_ba45 = list(p_val[1 + lenba2:1 + lenba2 + lenba45])
s0_bq1 = list(p_val[1 + lenba2 + lenba45:])

BIC = (lendelta + lenomi + lenba2 + lenba45 + lenbq1 + 4) * np.log(num_data)\
    - 2 * (L_val + L)
BIC_early = (lendelta + lenomi + 3) * np.log(num_data_early) - 2 * (L)
BIC_late = (lenba2 + lenba45 + lenbq1 + 1) * np.log(num_data_late)\
    - 2 * (L_val)
L_result.append([L_delta, L_omi, L, L_ba2, L_ba45, L_bq1, L + L_val,
                 BIC_early, BIC_early + BIC_late, p[0], p[1], p[2], p_val[0],
                 0.0,
                 np.mean(s0_delta),
                 np.mean(s0_omi),
                 np.mean(s0_ba2),
                 np.mean(s0_ba45),
                 np.mean(s0_bq1)])

# ============================================================================
# Infer on Alpha-Delta and Delta-Omicron cross-overs.
# One additional parameter for gamma_inf for later. Mean s0 must be zero.
# No BQ.1 shift
# ===========================================================================
lendelta = len(set(df_delta.country))
lenomi = len(set(df_omi.country))
delta_countries = sorted(list(set(df_delta.country)))
omi_countries = sorted(list(set(df_omi.country)))
delta_country2index = {c: int(i) for c, i in
                       zip(delta_countries, np.arange(len(delta_countries)))}
omi_country2index = {c: int(i) for c, i in
                     zip(omi_countries, np.arange(len(omi_countries)))}

n = 3 + lendelta + lenomi  # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(1, n))
b = matrix(0.0, size=(1, 1))
G = matrix(0.0, size=(n, n))
h = matrix(0.0, size=(n, 1))
kappa_da = 2.
Xdata = []
for line in df_delta.itertuples():
    W = 1 / line.s_var
    idx = 3 + delta_country2index[line.country]

    dC_vac = (line.C_alpha_vac - line.C_delta_vac)\
        + kappa_da * (line.C_alpha_alpha - line.C_delta_alpha)\
        + kappa_da * (line.C_alpha_delta - line.C_delta_delta)

    Q[0, 0] += 2 * W * dC_vac * dC_vac
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_vac * 1
    Q[idx, 0] += 2 * W * dC_vac * 1

    # linear terms
    P[0] += -2 * W * dC_vac * line.s_hat
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xvec[0] = dC_vac
    Xdata.append(np.array(Xvec))

for line in df_omi.itertuples():
    W = 1 / line.s_var
    idx = 3 + omi_country2index[line.country] + lendelta

    dC_vac = (line.C_delta_vac - line.C_omi_vac)\
        + (line.C_delta_bst - line.C_omi_bst)
    dC_inf = (line.C_delta_omi - line.C_omi_omi)\
        + (line.C_delta_delta - line.C_omi_delta)\
        + (line.C_delta_alpha - line.C_omi_alpha)

    Q[1, 1] += 2 * W * dC_vac * dC_vac
    Q[2, 2] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W

    # cross terms
    Q[1, idx] += 2 * W * dC_vac * 1
    Q[idx, 1] += 2 * W * dC_vac * 1
    Q[2, idx] += 2 * W * dC_inf * 1
    Q[idx, 2] += 2 * W * dC_inf * 1
    Q[1, 2] += 2 * W * dC_vac * dC_inf
    Q[2, 1] += 2 * W * dC_vac * dC_inf

    # linear terms
    P[1] += -2 * W * dC_vac * line.s_hat
    P[2] += -2 * W * dC_inf * line.s_hat
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xvec[1] = dC_vac
    Xvec[2] = dC_inf
    Xdata.append(np.array(Xvec))

# Fill in non-negativity constraints for s_0:
for i in range(lendelta + lenomi):
    G[i + 3, i + 3] = -1

sol = solvers.qp(Q, P, G=G, h=h)
p = np.array(list(sol['x']))

s_model = np.matmul(p, np.transpose(Xdata))
var = list(df_delta.s_var) + list(df_omi.s_var)
Y = list(df_delta.s_hat) + list(df_omi.s_hat)
L = np.sum(ss.norm.logpdf(np.array(s_model),
                          np.array(Y),
                          np.sqrt(np.array(var))))

L_delta = np.sum(ss.norm.logpdf(np.array(s_model[:len(df_delta)]),
                                np.array(Y[:len(df_delta)]),
                                np.sqrt(np.array(var[:len(df_delta)]))))
L_omi = np.sum(ss.norm.logpdf(np.array(s_model[len(df_delta):]),
                              np.array(Y[len(df_delta):]),
                              np.sqrt(np.array(var[len(df_delta):]))))

# Inference on later shifts
lenba2 = len(set(df_ba2.country))
lenba45 = len(set(df_ba45.country))
lenbq1 = len(set(df_bq1.country))
ba2_countries = sorted(list(set(df_ba2.country)))
ba45_countries = sorted(list(set(df_ba45.country)))
bq1_countries = sorted(list(set(df_bq1.country)))
ba2_country2index = {c: int(i) for c, i in
                     zip(ba2_countries, np.arange(len(ba2_countries)))}
ba45_country2index = {c: int(i) for c, i in
                      zip(ba45_countries, np.arange(len(ba45_countries)))}
bq1_country2index = {c: int(i) for c, i in
                     zip(bq1_countries, np.arange(len(bq1_countries)))}

n = 1 + lenba2 + lenba45   # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(1, n))  # constraints no constrains yet...
b = matrix(0.0, size=(1, 1))  # Constraints no constrains yet...
G = matrix(0.0, size=(n, n))
h = matrix(0.0, size=(n, 1))
Xdata = []

gamma_vac = p[1]
gamma_inf = p[2]
y_cor_list = []
for line in df_ba2.itertuples():
    W = 1 / line.s_var
    idx = 1 + ba2_country2index[line.country]
    cor = gamma_vac * (line.C_ba1_bst - line.C_ba2_bst)\
        + gamma_inf * (line.C_ba1_ba1 - line.C_ba2_ba1
                       + line.C_ba1_ba2 - line.C_ba2_ba2)
    y_cor_list.append(cor)
    dC_inf = 0.0
    y = line.s_hat - cor

    Q[0, 0] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_inf * 1
    Q[idx, 0] += 2 * W * dC_inf * 1

    # linear terms
    P[0] += -2 * W * dC_inf * y
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[0] = dC_inf
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_ba45.itertuples():
    W = 1 / line.s_var
    idx = 1 + ba45_country2index[line.country] + lenba2
    cor = gamma_vac * (line.C_ba2_bst - line.C_ba45_bst)
    y_cor_list.append(cor)
    dC_inf = (line.C_ba2_ba1 - line.C_ba45_ba1)\
        + (line.C_ba2_ba2 - line.C_ba45_ba2)\
        + (line.C_ba2_ba45 - line.C_ba45_ba45)
    y = line.s_hat - cor
    Q[0, 0] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_inf * 1
    Q[idx, 0] += 2 * W * dC_inf * 1

    # linear terms
    P[0] += -2 * W * dC_inf * y
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[0] = dC_inf
    Xvec[idx] = 1
    Xdata.append(Xvec)


# Fill in non-negativity constraints for s_0 for ba2:
for i in range(lenba2):
    G[i + 1, i + 1] = -1
# for i in range(lenba45 + lenbq1):
#   h[1+lenba2+i] = -means0
# Fill constraint mean s_0 for ba45 mean s_0 for bq1 = 0:
for i in range(lenba45):
    A[0, 1 + lenba2 + i] = 1 / (lenba45)
# for i in range(lenbq1):
#   A[1,1+lenba2+lenba45+i] = 1 / lenbq1

sol = solvers.qp(Q, P, G=G, h=h, A=A, b=b)
p_val = np.array(list(sol['x']))

s_model_val = np.matmul(p_val, np.transpose(Xdata))
var_val = list(df_ba2.s_var) + list(df_ba45.s_var)
Y_val = list(df_ba2.s_hat) + list(df_ba45.s_hat)
Y_val = np.array(Y_val) - np.array(y_cor_list)

N_ba2 = len(df_ba2)
N_ba45 = N_ba2 + len(df_ba45)
L_val = np.sum(ss.norm.logpdf(np.array(s_model_val),
                              np.array(Y_val),
                              np.sqrt(np.array(var_val))))
L_ba2 = np.sum(ss.norm.logpdf(np.array(s_model_val[:N_ba2]),
                              np.array(Y_val[:N_ba2]),
                              np.sqrt(np.array(var_val[:N_ba2]))))
L_ba45 = np.sum(ss.norm.logpdf(np.array(s_model_val[N_ba2:N_ba45]),
                               np.array(Y_val[N_ba2:N_ba45]),
                               np.sqrt(np.array(var_val[N_ba2:N_ba45]))))

s0_delta = list(p[3:3 + lendelta])
s0_omi = list(p[3 + lendelta:])
s0_ba2 = list(p_val[1:1 + lenba2])
s0_ba45 = list(p_val[1 + lenba2:1 + lenba2 + lenba45])


BIC = (lendelta + lenomi + lenba2 + lenba45 + 4) * np.log(num_data)\
    - 2 * (L_val + L)
BIC_early = (lendelta + lenomi + 3) * np.log(num_data_early) - 2 * (L)
BIC_late = (lenba2 + lenba45 + 1) * np.log(num_data_late) - 2 * (L_val)
L_result.append([L_delta, L_omi, L, L_ba2, L_ba45, 0, L + L_val, BIC_early,
                 BIC_early + BIC_late, p[0], p[1], p[2], p_val[0],
                 0.0,
                 np.mean(s0_delta),
                 np.mean(s0_omi),
                 np.mean(s0_ba2),
                 np.mean(s0_ba45),
                 0])

# ===========================================================================
# Infer on Alpha-Delta and Delta-Omicron cross-overs. One additional parameter
# for gamma_inf for later shifts. Mean s0 must be zero. Infection only
# ===========================================================================
lendelta = len(set(df_delta.country))
lenomi = len(set(df_omi.country))
delta_countries = sorted(list(set(df_delta.country)))
omi_countries = sorted(list(set(df_omi.country)))
delta_country2index = {c: int(i) for c, i in
                       zip(delta_countries, np.arange(len(delta_countries)))}
omi_country2index = {c: int(i) for c, i in
                     zip(omi_countries, np.arange(len(omi_countries)))}

n = 2 + lendelta + lenomi  # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(1, n))
b = matrix(0.0, size=(1, 1))
G = matrix(0.0, size=(n, n))
h = matrix(0.0, size=(n, 1))
kappa_da = 2.
Xdata = []
for line in df_delta.itertuples():
    W = 1 / line.s_var
    idx = 2 + delta_country2index[line.country]
    dC_vac = (line.C_alpha_alpha - line.C_delta_alpha)\
        + (line.C_alpha_delta - line.C_delta_delta)

    Q[0, 0] += 2 * W * dC_vac * dC_vac
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_vac * 1
    Q[idx, 0] += 2 * W * dC_vac * 1

    # linear terms
    P[0] += -2 * W * dC_vac * line.s_hat
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xvec[0] = dC_vac
    Xdata.append(np.array(Xvec))

for line in df_omi.itertuples():
    W = 1 / line.s_var
    idx = 2 + omi_country2index[line.country] + lendelta

    dC_vac = (line.C_delta_vac - line.C_omi_vac)\
        + (line.C_delta_bst - line.C_omi_bst)
    dC_inf = (line.C_delta_omi - line.C_omi_omi)\
        + (line.C_delta_delta - line.C_omi_delta)\
        + (line.C_delta_alpha - line.C_omi_alpha)

    Q[1, 1] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W

    # cross terms
    Q[1, idx] += 2 * W * dC_inf * 1
    Q[idx, 1] += 2 * W * dC_inf * 1

    # linear terms
    P[1] += -2 * W * dC_inf * line.s_hat
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xvec[1] = dC_inf
    Xdata.append(np.array(Xvec))

for i in range(lendelta + lenomi):
    G[i + 2, i + 2] = -1

sol = solvers.qp(Q, P, G=G, h=h)
p = np.array(list(sol['x']))

s_model = np.matmul(p, np.transpose(Xdata))
var = list(df_delta.s_var) + list(df_omi.s_var)
Y = list(df_delta.s_hat) + list(df_omi.s_hat)
L = np.sum(ss.norm.logpdf(np.array(s_model),
                          np.array(Y),
                          np.sqrt(np.array(var))))

L_delta = np.sum(ss.norm.logpdf(np.array(s_model[:len(df_delta)]),
                                np.array(Y[:len(df_delta)]),
                                np.sqrt(np.array(var[:len(df_delta)]))))
L_omi = np.sum(ss.norm.logpdf(np.array(s_model[len(df_delta):]),
                              np.array(Y[len(df_delta):]),
                              np.sqrt(np.array(var[len(df_delta):]))))

# Inference on later shifts
lenba2 = len(set(df_ba2.country))
lenba45 = len(set(df_ba45.country))
lenbq1 = len(set(df_bq1.country))
ba2_countries = sorted(list(set(df_ba2.country)))
ba45_countries = sorted(list(set(df_ba45.country)))
bq1_countries = sorted(list(set(df_bq1.country)))
ba2_country2index = {c: int(i) for c, i in
                     zip(ba2_countries, np.arange(len(ba2_countries)))}
ba45_country2index = {c: int(i) for c, i in
                      zip(ba45_countries, np.arange(len(ba45_countries)))}
bq1_country2index = {c: int(i) for c, i in
                     zip(bq1_countries, np.arange(len(bq1_countries)))}

n = 1 + lenba2 + lenba45 + lenbq1  # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(2, n))  # constraints no constrains yet...
b = matrix(0.0, size=(2, 1))  # Constraints no constrains yet...
G = matrix(0.0, size=(n, n))
h = matrix(0.0, size=(n, 1))
Xdata = []

gamma_inf = p[1]
y_cor_list = []
for line in df_ba2.itertuples():
    W = 1 / line.s_var
    idx = 1 + ba2_country2index[line.country]
    cor = gamma_inf * (line.C_ba1_ba1 - line.C_ba2_ba1
                       + line.C_ba1_ba2 - line.C_ba2_ba2)
    y_cor_list.append(cor)
    dC_inf = 0.0
    y = line.s_hat - cor

    Q[0, 0] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_inf * 1
    Q[idx, 0] += 2 * W * dC_inf * 1

    # linear terms
    P[0] += -2 * W * dC_inf * y
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[0] = dC_inf
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_ba45.itertuples():
    W = 1 / line.s_var
    idx = 1 + ba45_country2index[line.country] + lenba2
    cor = 0
    y_cor_list.append(cor)
    dC_inf = (line.C_ba2_ba1 - line.C_ba45_ba1)\
        + (line.C_ba2_ba2 - line.C_ba45_ba2)\
        + (line.C_ba2_ba45 - line.C_ba45_ba45)
    y = line.s_hat - cor
    Q[0, 0] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_inf * 1
    Q[idx, 0] += 2 * W * dC_inf * 1

    # linear terms
    P[0] += -2 * W * dC_inf * y
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[0] = dC_inf
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_bq1.itertuples():
    W = 1 / line.s_var
    idx = 1 + bq1_country2index[line.country] + lenba2 + lenba45
    cor = 0
    y_cor_list.append(cor)
    dC_inf = (line.C_ba45_ba1 - line.C_bq1_ba1
              + line.C_ba45_ba2 - line.C_bq1_ba2
              + line.C_ba45_ba45 - line.C_bq1_ba45
              + line.C_ba45_bq1 - line.C_bq1_bq1)
    y = line.s_hat - cor
    Q[0, 0] += 2 * W * dC_inf * dC_inf
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_inf * 1
    Q[idx, 0] += 2 * W * dC_inf * 1

    # linear terms
    P[0] += -2 * W * dC_inf * y
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[0] = dC_inf
    Xvec[idx] = 1
    Xdata.append(Xvec)

# Fill in non-negativity constraints for s_0 for ba2:
for i in range(lenba2):
    G[i + 1, i + 1] = -1
# Fill constraint mean s_0 for ba45 mean s_0 for bq1 = 0:
for i in range(lenba45 + lenbq1):
    A[0, 1 + lenba2 + i] = 1 / (lenba45)

for i in range(lenbq1):
    A[1, 1 + lenba2 + lenba45 + i] = 1 / (lenbq1)


sol = solvers.qp(Q, P, G=G, h=h, A=A, b=b)
p_val = np.array(list(sol['x']))

s_model_val = np.matmul(p_val, np.transpose(Xdata))
var_val = list(df_ba2.s_var) + list(df_ba45.s_var) + list(df_bq1.s_var)
Y_val = list(df_ba2.s_hat) + list(df_ba45.s_hat) + list(df_bq1.s_hat)
Y_val = np.array(Y_val) - np.array(y_cor_list)

N_ba2 = len(df_ba2)
N_ba45 = N_ba2 + len(df_ba45)
L_val = np.sum(ss.norm.logpdf(np.array(s_model_val),
                              np.array(Y_val),
                              np.sqrt(np.array(var_val))))
L_ba2 = np.sum(ss.norm.logpdf(np.array(s_model_val[:N_ba2]),
                              np.array(Y_val[:N_ba2]),
                              np.sqrt(np.array(var_val[:N_ba2]))))
L_ba45 = np.sum(ss.norm.logpdf(np.array(s_model_val[N_ba2:N_ba45]),
                               np.array(Y_val[N_ba2:N_ba45]),
                               np.sqrt(np.array(var_val[N_ba2:N_ba45]))))
L_bq1 = np.sum(ss.norm.logpdf(np.array(s_model_val[N_ba45:]),
                              np.array(Y_val[N_ba45:]),
                              np.sqrt(np.array(var_val[N_ba45:]))))

s0_delta = list(p[2:2 + lendelta])
s0_omi = list(p[2 + lendelta:])
s0_ba2 = list(p_val[1:1 + lenba2])
s0_ba45 = list(p_val[1 + lenba2:1 + lenba2 + lenba45])
s0_bq1 = list(p_val[1 + lenba2 + lenba45:])

BIC = (lendelta + lenomi + lenba2 + lenba45 + 3) * np.log(num_data)\
    - 2 * (L_val + L)
BIC_early = (lendelta + lenomi + 2) * np.log(num_data_early) - 2 * (L)
BIC_late = (lenba2 + lenba45 + lenbq1 + 1) * np.log(num_data_late)\
    - 2 * (L_val)
L_result.append([L_delta, L_omi, L, L_ba2, L_ba45, 0, L + L_val,
                 BIC_early, BIC_early + BIC_late, p[0], 0.0, p[1],
                 p_val[0], 0.0,
                 np.mean(s0_delta),
                 np.mean(s0_omi),
                 np.mean(s0_ba2),
                 np.mean(s0_ba45),
                 np.mean(s0_bq1)])

# ============================================================================
# Infer on Alpha-Delta and Delta-Omicron cross-overs. One additional parameter
# for gamma_inf for later shifts. Mean s0 must be zero. Vaccination only
# ============================================================================
lenomi = len(set(df_omi.country))
delta_countries = sorted(list(set(df_delta.country)))
omi_countries = sorted(list(set(df_omi.country)))
delta_country2index = {c: int(i) for c, i in
                       zip(delta_countries, np.arange(len(delta_countries)))}
omi_country2index = {c: int(i) for c, i in
                     zip(omi_countries, np.arange(len(omi_countries)))}

n = 2 + lendelta + lenomi  # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(1, n))
b = matrix(0.0, size=(1, 1))
G = matrix(0.0, size=(n, n))
h = matrix(0.0, size=(n, 1))
kappa_da = 2.
Xdata = []
for line in df_delta.itertuples():
    W = 1 / line.s_var
    idx = 2 + delta_country2index[line.country]

    dC_vac = (line.C_alpha_vac - line.C_delta_vac)

    Q[0, 0] += 2 * W * dC_vac * dC_vac
    Q[idx, idx] += 2 * W * 1 * 1

    # cross terms
    Q[0, idx] += 2 * W * dC_vac * 1
    Q[idx, 0] += 2 * W * dC_vac * 1

    # linear terms
    P[0] += -2 * W * dC_vac * line.s_hat
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xvec[0] = dC_vac
    Xdata.append(np.array(Xvec))

for line in df_omi.itertuples():
    W = 1 / line.s_var
    idx = 2 + omi_country2index[line.country] + lendelta

    dC_vac = (line.C_delta_vac - line.C_omi_vac)\
        + (line.C_delta_bst - line.C_omi_bst)
    dC_inf = (line.C_delta_omi - line.C_omi_omi)\
        + (line.C_delta_delta - line.C_omi_delta)\
        + (line.C_delta_alpha - line.C_omi_alpha)

    Q[1, 1] += 2 * W * dC_vac * dC_vac
    Q[idx, idx] += 2 * W

    # cross terms
    Q[1, idx] += 2 * W * dC_vac * 1
    Q[idx, 1] += 2 * W * dC_vac * 1

    # linear terms
    P[1] += -2 * W * dC_vac * line.s_hat
    P[idx] += -2 * W * line.s_hat

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xvec[1] = dC_vac
    Xdata.append(np.array(Xvec))

# Fill in non-negativity constraints for s_0:
for i in range(lendelta + lenomi):
    G[i + 2, i + 2] = -1

sol = solvers.qp(Q, P, G=G, h=h)
p = np.array(list(sol['x']))

s_model = np.matmul(p, np.transpose(Xdata))
var = list(df_delta.s_var) + list(df_omi.s_var)
Y = list(df_delta.s_hat) + list(df_omi.s_hat)
L = np.sum(ss.norm.logpdf(np.array(s_model),
                          np.array(Y),
                          np.sqrt(np.array(var))))

L_delta = np.sum(ss.norm.logpdf(np.array(s_model[:len(df_delta)]),
                                np.array(Y[:len(df_delta)]),
                                np.sqrt(np.array(var[:len(df_delta)]))))
L_omi = np.sum(ss.norm.logpdf(np.array(s_model[len(df_delta):]),
                              np.array(Y[len(df_delta):]),
                              np.sqrt(np.array(var[len(df_delta):]))))

# Inference on later shifts
lenba2 = len(set(df_ba2.country))
lenba45 = len(set(df_ba45.country))
lenbq1 = len(set(df_bq1.country))
ba2_countries = sorted(list(set(df_ba2.country)))
ba45_countries = sorted(list(set(df_ba45.country)))
bq1_countries = sorted(list(set(df_bq1.country)))
ba2_country2index = {c: int(i) for c, i in
                     zip(ba2_countries, np.arange(len(ba2_countries)))}
ba45_country2index = {c: int(i) for c, i in
                      zip(ba45_countries, np.arange(len(ba45_countries)))}
bq1_country2index = {c: int(i) for c, i in
                     zip(bq1_countries, np.arange(len(bq1_countries)))}

n = 0 + lenba2 + lenba45 + lenbq1  # number of parameters
Q = matrix(0.0, size=(n, n))  # quadratic terms
P = matrix(0.0, size=(n, 1))  # linear terms
A = matrix(0.0, size=(1, n))  # constraints no constrains yet...
b = matrix(0.0, size=(1, 1))  # Constraints no constrains yet...
G = matrix(0.0, size=(n, n))
h = matrix(0.0, size=(n, 1))
Xdata = []

gamma_vac = p[1]
y_cor_list = []
for line in df_ba2.itertuples():
    W = 1 / line.s_var
    idx = ba2_country2index[line.country]
    cor = gamma_vac * (line.C_ba1_bst - line.C_ba2_bst)
    y_cor_list.append(cor)
    dC_inf = 0.0
    y = line.s_hat - cor

    Q[idx, idx] += 2 * W * 1 * 1

    # linear terms
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    # Xvec[0] = dC_inf
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_ba45.itertuples():
    W = 1 / line.s_var
    idx = ba45_country2index[line.country] + lenba2
    cor = gamma_vac * (line.C_ba2_bst - line.C_ba45_bst)
    y_cor_list.append(cor)
    y = line.s_hat - cor
    Q[idx, idx] += 2 * W * 1 * 1

    # linear terms
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xdata.append(Xvec)

for line in df_bq1.itertuples():
    W = 1 / line.s_var
    idx = bq1_country2index[line.country] + lenba2 + lenba45
    cor = gamma_vac * (line.C_ba45_bst - line.C_bq1_bst
                       + line.C_ba45_biv - line.C_bq1_biv)
    y_cor_list.append(cor)
    y = line.s_hat - cor
    Q[idx, idx] += 2 * W * 1 * 1
    # linear terms
    P[idx] += -2 * W * y

    Xvec = np.zeros(n)
    Xvec[idx] = 1
    Xdata.append(Xvec)

# Fill in non-negativity constraints for s_0 for ba2:
for i in range(lenba2):
    G[i, i] = -1

# Fill constraint mean s_0 for ba45 mean s_0 for bq1 = 0:
for i in range(lenba45 + lenbq1):
    A[0, lenba2 + i] = 1 / (lenba45 + lenbq1)

sol = solvers.qp(Q, P, G=G, h=h, A=A, b=b)
p_val = np.array(list(sol['x']))

s_model_val = np.matmul(p_val, np.transpose(Xdata))
var_val = list(df_ba2.s_var) + list(df_ba45.s_var) + list(df_bq1.s_var)
Y_val = list(df_ba2.s_hat) + list(df_ba45.s_hat) + list(df_bq1.s_hat)
Y_val = np.array(Y_val) - np.array(y_cor_list)


N_ba2 = len(df_ba2)
N_ba45 = N_ba2 + len(df_ba45)
L_val = np.sum(ss.norm.logpdf(np.array(s_model_val),
                              np.array(Y_val),
                              np.sqrt(np.array(var_val))))
L_ba2 = np.sum(ss.norm.logpdf(np.array(s_model_val[:N_ba2]),
                              np.array(Y_val[:N_ba2]),
                              np.sqrt(np.array(var_val[:N_ba2]))))
L_ba45 = np.sum(ss.norm.logpdf(np.array(s_model_val[N_ba2:N_ba45]),
                               np.array(Y_val[N_ba2:N_ba45]),
                               np.sqrt(np.array(var_val[N_ba2:N_ba45]))))
L_bq1 = np.sum(ss.norm.logpdf(np.array(s_model_val[N_ba45:]),
                              np.array(Y_val[N_ba45:]),
                              np.sqrt(np.array(var_val[N_ba45:]))))

s0_delta = list(p[2:2 + lendelta])
s0_omi = list(p[2 + lendelta:])
s0_ba2 = list(p_val[:lenba2])
s0_ba45 = list(p_val[lenba2:lenba2 + lenba45])
s0_bq1 = list(p_val[lenba2 + lenba45:])

BIC = (lendelta + lenomi + lenba2 + lenba45 + lenbq1 + 4) * np.log(num_data)\
    - 2 * (L_val + L)
BIC_early = (lendelta + lenomi + 2) * np.log(num_data_early) - 2 * (L)
BIC_late = (lenba2 + lenba45 + lenbq1) * np.log(num_data_late) - 2 * (L_val)
L_result.append([L_delta, L_omi, L, L_ba2, L_ba45, L_bq1, L + L_val, BIC_early,
                 BIC_early + BIC_late, p[0], p[1], 0.0, 0.0, 0.0,
                 np.mean(s0_delta),
                 np.mean(s0_omi),
                 np.mean(s0_ba2),
                 np.mean(s0_ba45),
                 np.mean(s0_bq1)])

L_result = pd.DataFrame(L_result,
                        columns=['L_delta',
                                 'L_omi',
                                 'L_early',
                                 'L_ba2',
                                 'L_ba45',
                                 'L_bq1',
                                 'L',
                                 'BIC_early',
                                 'BIC',
                                 'gamma_vac_ad',
                                 'gamma_vac_do',
                                 'gamma_inf_do',
                                 'gamma_inf_ba45',
                                 'gamma_inf_bq1',
                                 's0_ad',
                                 's0_do',
                                 's0_ba2',
                                 's0_ba45',
                                 's0_bq1'],
                        index=['s0',
                               'antigenic_ad_do',
                               'antigenic_omi_inf_means0_0',
                               'antigenic_omi_inf_means0_no_bq1',
                               'inf_only',
                               'vac_only'])
L_result = L_result.round(decimals=4)
L_result['L'] -= L_result.loc['s0', 'L']
L_result['L_delta'] -= L_result.loc['s0', 'L_delta']
L_result['L_omi'] -= L_result.loc['s0', ' L_omi']
L_result['L_early'] -= L_result.loc['s0', 'L_early']
L_result['L_ba2'] -= L_result.loc['s0', 'L_ba2']
L_result['L_ba45'] -= L_result.loc['s0', 'L_ba45']
L_result['L_bq1'] -= L_result.loc['s0', 'L_bq1']
L_result['BIC_early'] -= L_result.loc['s0', 'BIC_early']
L_result['BIC'] -= L_result.loc['s0', 'BIC']
