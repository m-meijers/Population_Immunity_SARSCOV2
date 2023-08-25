import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import scipy.stats as ss
import scipy
from collections import defaultdict
import time
from scipy.optimize import minimize
import scipy.integrate as si
import copy
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import scipy.integrate as integrate
import sys
sys.path.insert(0,"..")
from util.time import Time


color_list = [(76,109,166),(215,139,45),(125,165,38),(228,75,41),(116,97,164),(182,90,36),(80,141,188),(246,181,56),(125,64,119),(158,248,72)]
color_list2 = []
for i in range(len(color_list)):
    color_list2.append(np.array(color_list[i])/256.)
color_list = color_list2


df_delta = pd.read_csv("../output/Data_delta_shift.txt",'\t',index_col=False)
df_omi = pd.read_csv("../output/Data_omi_shift.txt",'\t',index_col=False)

#Add up a country-variance for all measurements.
country_variance = []
for c in list(set(df_delta.country)):
    meansvar = np.mean(df_delta.loc[df_delta.country == c,'s_var'])
    country_variance.append(meansvar)
median_svar_da = np.median(country_variance)

country_variance = []
for c in list(set(df_omi.country)):
    meansvar = np.mean(df_omi.loc[df_omi.country == c,'s_var'])
    country_variance.append(meansvar)
median_svar_od = np.median(country_variance)

df_delta['s_var'] = np.array(df_delta['s_var'] + median_svar_da)
df_omi['s_var'] = np.array(df_omi['s_var'] + median_svar_od)


F = lambda R, tau, var: (tau / var) * (R**(var / tau**2) - 1)
F2 = lambda R, tau: 1/tau * np.log(R)
gam = lambda tau, var: ss.gamma(tau**2/var, scale=var/tau)


countries = list(set(df_delta.country)) + list(set(df_omi.country))
countries = sorted(list(set(countries)))

country2color = {}
country2style = {}
country2marker = {}
for i in range(int(len(countries)/2)):
    country2color[countries[i]] = color_list[i]
    country2style[countries[i]] = '-'
    country2marker[countries[i]] = 'o'
add = int(len(countries)/2)
for i in range(int(len(countries)/2), len(countries)):
    country2color[countries[i]] = color_list[i - int(len(countries)/2)]
    country2style[countries[i]] = '-'
    country2marker[countries[i]] = 'x'


generation_t = np.linspace(0,10)

fs = 12
ls = 12
ratio = 1
plt.figure(figsize=(7,12))
sag = np.linspace(0.0,0.12)

#minimal model with gamma-distributed fgeneration intervals
ax = plt.subplot(521)
plt.plot(generation_t, gam(5.0,3.2).pdf(generation_t),color='k')
plt.plot(generation_t, 1.2 * gam(5.0,3.2).pdf(generation_t),color='grey')
plt.ylabel("Density, $P(\\tau)$",fontsize=fs)
plt.ylim([0,0.30])
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

ax = plt.subplot(522)
plt.plot(sag, F(1.2 * np.exp(5.0 * sag) * 0.9,5.0,3.2) - F(0.9,5.0,3.2), color=color_list[0])
plt.plot(sag, F(1.2 * np.exp(5.0 * sag) * 1.1,5.0,3.2) - F(1.1,5.0,3.2), color=color_list[1])
plt.plot(sag, F(1.2 * np.exp(5.0 * sag) * 1.3,5.0,3.2) - F(1.3,5.0,3.2), color=color_list[2])
plt.plot(sag, sag, 'k--')
# plt.xlabel("Antigenic selection, $s_{ag}$",fontsize=fs)
plt.ylabel("selection, $s$",fontsize=fs)
plt.ylim([0,0.15])
plt.xticks([0,0.05,0.1],['0.0','0.05','0.10'])
plt.yticks([0.0,0.05,0.10,0.15],['0.0','0.05','0.10','0.15'])
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

#variant shorter mean tau
ax = plt.subplot(523)
plt.plot(generation_t, gam(5.0,3.2).pdf(generation_t),color='k')
plt.plot(generation_t, gam(4.0,3.2).pdf(generation_t),color='grey')
plt.ylabel("Density, $P(\\tau)$",fontsize=fs)
plt.ylim([0,0.30])
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

ax = plt.subplot(524)
plt.plot(sag, F(np.exp(4.0 * sag) * 0.9,4.0,3.2) - F(0.9,5.0,3.2), color=color_list[0])
plt.plot(sag, F(np.exp(4.0 * sag) * 1.1,4.0,3.2) - F(1.1,5.0,3.2), color=color_list[1])
plt.plot(sag, F(np.exp(4.0 * sag) * 1.3,4.0,3.2) - F(1.3,5.0,3.2), color=color_list[2])
plt.plot(sag, sag, 'k--')
# plt.xlabel("Antigenic selection, $s_{ag}$",fontsize=fs)
plt.ylabel("selection, $s$",fontsize=fs)
plt.ylim([0,0.15])
plt.xticks([0,0.05,0.1],['0.0','0.05','0.10'])
plt.yticks([0.0,0.05,0.10,0.15],['0.0','0.05','0.10','0.15'])
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)


#variant with increased pathogenicity
ax = plt.subplot(525)
plt.plot(generation_t, gam(5.0,3.2).pdf(generation_t),color='k')
plt.plot(generation_t, 1.2 * gam(5.5,5).pdf(generation_t),color='grey')
plt.ylabel("Density, $P(\\tau)$",fontsize=fs)
plt.ylim([0,0.30])
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

ax = plt.subplot(526)
plt.plot(sag, F(1.2 * np.exp(5.5 * sag) * 0.9,5.5,5.) - F(0.9,5.0,3.2), color=color_list[0])
plt.plot(sag, F(1.2 * np.exp(5.5 * sag) * 1.1,5.5,5.) - F(1.1,5.0,3.2), color=color_list[1])
plt.plot(sag, F(1.2 * np.exp(5.5 * sag) * 1.3,5.5,5.) - F(1.3,5.0,3.2), color=color_list[2])
plt.plot(sag, sag, 'k--')
# plt.xlabel("Antigenic selection, $s_{ag}$",fontsize=fs)
plt.ylabel("selection, $s$",fontsize=fs)
plt.ylim([0,0.15])
plt.xticks([0,0.05,0.1],['0.0','0.05','0.10'])
plt.yticks([0.0,0.05,0.10,0.15],['0.0','0.05','0.10','0.15'])
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)



#effect of NPI measures
ax = plt.subplot(527)
plt.plot(generation_t, gam(5.0,3.2).pdf(generation_t),color=color_list[3])
plt.plot(generation_t, 1.1/ 1.3 * gam(5.,3.2).pdf(generation_t),color=color_list[4])
plt.ylabel("Density, $P(\\tau)$",fontsize=fs)
plt.ylim([0,0.30])
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

ax = plt.subplot(528)
plt.plot(sag, F(np.exp(5 * sag) * 1.3,5.0,3.2) - F(1.3,5.0,3.2), color=color_list[3])
plt.plot(sag, F(np.exp(5 * sag) * 1.1,5.0,3.2) - F(1.1,5.0,3.2), color=color_list[4])
plt.plot(sag, sag, 'k--')
# plt.xlabel("Antigenic selection, $s_{ag}$",fontsize=fs)
plt.ylabel("selection, $s$",fontsize=fs)
plt.ylim([0,0.15])
plt.xticks([0,0.05,0.1],['0.0','0.05','0.10'])
plt.yticks([0.0,0.05,0.10,0.15],['0.0','0.05','0.10','0.15'])
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)


#effect of NPI measures on tau
ax = plt.subplot(529)
plt.plot(generation_t, gam(5.0,3.2).pdf(generation_t),color=color_list[3])
plt.plot(generation_t, 0.85 * gam(4.,2.5).pdf(generation_t),color=color_list[4])
plt.ylabel("Density, $P(\\tau)$",fontsize=fs)
plt.xlabel("Generation time, $\\tau$",fontsize=fs)
plt.ylim([0,0.30])
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

ax = plt.subplot(5,2,10)
plt.plot(sag, F(np.exp(5 * sag) * 1.3,5.0,3.2) - F(1.3,5.0,3.2), color=color_list[3])
plt.plot(sag, F(np.exp(4. * sag) * 1.1,4.,2.5) - F(1.1,4.,2.5), color=color_list[4])
plt.plot(sag, sag, 'k--')
plt.xlabel("Antigenic selection, $s_{ag}$",fontsize=fs)
plt.ylabel("selection, $s$",fontsize=fs)
plt.ylim([0,0.15])
plt.xticks([0,0.05,0.1],['0.0','0.05','0.10'])
plt.yticks([0.0,0.05,0.10,0.15],['0.0','0.05','0.10','0.15'])
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

plt.subplots_adjust(wspace=0.3,hspace=0.25)
plt.savefig("FigS4_theory_plots.pdf")
plt.close()

ratio = 1/1.69
#Get correlation and P-value
ls=12
ps=12
plt.figure(figsize=(14,10))
ax = plt.subplot(221)
X = []
Y = []
Y_var = []
for country in list(set(df_delta.country)):
    df_c = df_delta.loc[list(df_delta.country == country)]
    # X += list(np.log(df_c['R_inst'])) 
    X += list(df_c['Fabs']) 
    Y += list(df_c.s_hat)
    Y_var += list(df_c.s_var)
    plt.errorbar(df_c['Fabs'], df_c.s_hat,yerr = np.sqrt(df_c.s_var),marker=country2marker[country],alpha=0.7, color=country2color[country])

X = np.array(X)
Y = np.array(Y)
Y_var = np.array(Y_var)
R = ss.linregress(X,Y)
X_range = np.linspace(min(X), max(X))
plt.plot(X_range, R.slope * X_range + R.intercept,'k-',linewidth=2.)
plt.title(f"Alpha - Delta, slope: {R.slope:.2}, pval: {R.pvalue:.2E}")
plt.xlabel("Epidemic growth, $\\hat{F}$")
plt.ylabel("Selection, $s$")
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
#Get correlation and P-value
ax = plt.subplot(222)
X = []
Y = []
Y_var = []
for country in list(set(df_omi.country)):
    df_c = df_omi.loc[list(df_omi.country == country)]
    # X += list(np.log(df_c['R_inst'])) 
    X += list(df_c['Fabs']) 
    Y += list(df_c.s_hat)
    Y_var += list(df_c.s_var)
    plt.errorbar(df_c['Fabs'], df_c.s_hat,yerr = np.sqrt(df_c.s_var),marker=country2marker[country],alpha=0.7, color=country2color[country])
X = np.array(X)
Y = np.array(Y)
Y_var = np.array(Y_var)
R = ss.linregress(X,Y)
X_range = np.linspace(min(X), max(X))
plt.plot(X_range, R.slope * X_range + R.intercept,'k-',linewidth=2.)
plt.title(f"Delta - BA.1, slope: {R.slope:.2}, pval: {R.pvalue:.2E}")
plt.xlabel("Epidemic growth, $\\hat{F}$")
plt.ylabel("Selection, $s$")
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

ax = plt.subplot(223)
X = []
Y = []
av_times = []
for country in list(set(df_delta.country)):
    df_c = df_delta.loc[list(df_delta.country == country)]
    times = np.array(df_c.time) 
    plt.plot(times - np.mean(times), df_c['Fabs'],'-',alpha=0.7, color=country2color[country],marker= country2marker[country])
    X += list(times - np.mean(times)) 
    Y += list(df_c['Fabs'])
    av_times.append(times[-1] - times[0])
X = np.array(X)
Y = np.array(Y)
R = ss.linregress(X,Y)
plt.title(f"Alpha - Delta, slope: {R.slope:.2}, av_time: {np.mean(av_times):.1f}, pval: {R.pvalue:.2E}")
plt.ylabel("Epidemic growth, $\\hat{F}$")
plt.xlabel("Time from midpoint, $t$ [days]",fontsize=fs)
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
#Get correlation and P-value
ax = plt.subplot(224)
X = []
Y = []
av_times = []
for country in list(set(df_omi.country)):
    df_c = df_omi.loc[list(df_omi.country == country)]
    times = np.array(df_c.time) 
    plt.plot(times - np.mean(times), df_c['Fabs'],'-',alpha=0.7, color=country2color[country],marker= country2marker[country])
    X += list(times - np.mean(times)) 
    Y += list(df_c['Fabs'])
    av_times.append(times[-1] - times[0])
Y = np.array(Y)
R = ss.linregress(X,Y)
plt.title(f"Delta - BA.1, slope: {R.slope:.2}, av_time: {np.mean(av_times):.1f}, pval: {R.pvalue:.2E}")
plt.ylabel("Epidemic growth, $\\hat{F}$")
plt.xlabel("Time from midpoint, $t$ [days]",fontsize=fs)
plt.tick_params(direction='in')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

legend_elements = []
for country in countries:
    legend_elements.append(Line2D([],[],marker=country2marker[country],markersize=6,color=country2color[country],linestyle=country2style[country],label=country, linewidth=2.0))
plt.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1.05,0.5),prop={'size':ls})
plt.subplots_adjust(right=0.8)
plt.savefig("FigS4_correlation_plots.pdf")
plt.close()

