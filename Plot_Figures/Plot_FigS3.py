import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from flai.util.Time import Time
from sklearn import linear_model
from matplotlib.lines import Line2D
import scipy.stats as ss
import json
import scipy.optimize as so
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

df_alpha = pd.read_csv("../output/Data_alpha_shift.txt",'\t',index_col=False)
df_delta = pd.read_csv("../output/Data_delta_shift.txt",'\t',index_col=False)
df_omi = pd.read_csv("../output/Data_omi_shift.txt",'\t',index_col=False)
df_ba2= pd.read_csv("../output/Data_ba2_shift.txt",'\t',index_col=False)
df_ba45 = pd.read_csv("../output/Data_ba45_shift.txt",'\t',index_col=False)
df_bq1 = pd.read_csv("../output/Data_bq1_shift.txt",'\t',index_col=False)
#Add up a country-variance for all measurements.
country_variance = []
for c in list(set(df_alpha.country)):
    meansvar = np.mean(df_alpha.loc[df_alpha.country == c,'s_var'])
    country_variance.append(meansvar)
median_svar_awt = np.median(country_variance)

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

country_variance = []
for c in list(set(df_ba2.country)):
    meansvar = np.mean(df_ba2.loc[df_ba2.country == c,'s_var'])
    country_variance.append(meansvar)
median_svar_ba2 = np.median(country_variance)

country_variance = []
for c in list(set(df_ba45.country)):
    meansvar = np.mean(df_ba45.loc[df_ba45.country == c,'s_var'])
    country_variance.append(meansvar)
median_svar_ba45 = np.median(country_variance)

country_variance = []
for c in list(set(df_bq1.country)):
    meansvar = np.mean(df_bq1.loc[df_bq1.country == c,'s_var'])
    country_variance.append(meansvar)
median_svar_bq1 = np.median(country_variance)

df_alpha['s_var'] = np.array(df_alpha['s_var'] + median_svar_awt)
df_delta['s_var'] = np.array(df_delta['s_var'] + median_svar_da)
df_omi['s_var'] = np.array(df_omi['s_var'] + median_svar_od)
df_ba2['s_var'] = np.array(df_ba2['s_var'] + median_svar_ba2)
df_ba45['s_var'] = np.array(df_ba45['s_var'] + median_svar_ba45)
df_bq1['s_var'] = np.array(df_bq1['s_var'] + median_svar_bq1)


df = pd.read_csv("../output/data_immune_trajectories.txt",'\t',index_col=False)


countries = list(set(list(set(df_alpha.country)) + list(set(df_delta.country)) + list(set(df_omi.country))))
countries = sorted(list(set(countries)))


color_list = [(76,109,166),(215,139,45),(125,165,38),(228,75,41),(116,97,164),(182,90,36),(80,141,188),(246,181,56),(125,64,119),(158,248,72)]
color_list2 = []
for i in range(len(color_list)):
	color_list2.append(np.array(color_list[i])/256.)
color_list = color_list2

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


cladeshift2df = {'wt-alpha':df_alpha,'alpha-delta':df_delta,'delta-omi':df_omi,'ba1-ba2':df_ba2,'ba2-ba45':df_ba45,'ba45-bq1':df_bq1}
cladeshift2ylim = {'wt-alpha':(0,0.08),'alpha-delta':(0,0.08),'delta-omi':(-0.1,0.35),'ba1-ba2':(-0.1,0.35),'ba2-ba45':(0.0,0.15),'ba45-bq1':(0.0,0.15)}
fs = 12
ls = 10
ms = 2
ratio = 1/1.68
inch2cm = 2.54
lw = 1

fig = plt.figure(figsize=(35/inch2cm,30/inch2cm))
gs0 = gridspec.GridSpec(3,2,figure=fig,hspace=0.35)

lines = []
for clade_index, clade_shift in enumerate(['wt-alpha','alpha-delta','delta-omi','ba1-ba2','ba2-ba45','ba45-bq1']):
	df_s = cladeshift2df[clade_shift]
	#selection change over time 
	gs = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=gs0[clade_index],hspace=0.1,wspace=0.1)
	ax = fig.add_subplot(gs[0,0])

	t_length = []
	slopes = []
	s_hat = []
	time_list = []
	mean_s_list = []
	for country in sorted(list(set(df_s.country))):
		mean_s = np.mean(df_s.loc[(df_s.country == country),'s_hat'])
		times = np.array(df_s.loc[(df_s.country == country),'time'])
		plt.errorbar(times - np.mean(times), np.array(df_s.loc[(df_s.country == country),'s_hat']) - mean_s, np.sqrt(np.array(df_s.loc[(df_s.country == country),'s_var']))* 1.96,alpha=0.7,color=country2color[country], linestyle=country2style[country], marker =country2marker[country])
		t_length.append(times[-1] - times[0])
		R = ss.linregress(times - np.mean(times), np.array(df_s.loc[ (df_s.country == country),'s_hat']))
		slopes.append(R.slope)

		s_hat += list(np.array(df_s.loc[(df_s.country == country),'s_hat']))
		time_list += list(times)
		mean_s_list.append(mean_s)

	a = np.array(slopes)
	print(f"{clade_shift} increase in {len(a[a>0])} out of {len(a)} regions")

	R = ss.linregress(time_list,s_hat)
	# plt.title(f"{clade_shift}, Var(s lin)={(np.mean(slopes)*np.mean(t_length))**2:.2E}, pval: {R.pvalue:.2E}")
	plt.title(f"{clade_shift}",fontsize=fs+3)
	x_range= np.linspace(-1/2. * np.mean(t_length),1/2. * np.mean(t_length))
	plt.plot(x_range, np.mean(slopes) * x_range - 0.5 * np.mean(slopes),'k-',linewidth=3)

	plt.ylim([-0.06,0.06])
	plt.xlabel("Time from midpoint, $t$ [days]",fontsize=fs)
	
	plt.xticks([-40,-20,0,20,40],[-40,-20,0,20,40])
	plt.yticks([-0.05,0.0,0.05],['-0.05','0.0','0.05'])
	plt.ylabel("$\\Delta \\hat{s}(t)$",fontsize=fs)
	# plt.xlabel("Time from $x_{\\rm inv} = 0.01$",fontsize=fs)
	plt.xlim([-40,40])
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

	plt.tick_params(direction='in',labelsize=ls)

	lines.append([clade_shift, (np.mean(slopes) * np.mean(t_length))**2, R.pvalue])

	

legend_elements = []
for country in countries:
	legend_elements.append(Line2D([],[],marker=country2marker[country],markersize=6,color=country2color[country],linestyle=country2style[country],label=country, linewidth=2.0, alpha=0.7))
legend_elements.append(Line2D([],[],color='k',linestyle='-',label='Vaccination',linewidth=2.0))
legend_elements.append(Line2D([],[],color='k',linestyle='--',label='Infection',linewidth=2.0))

plt.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1.01,2.),prop={'size':ls})
plt.subplots_adjust(bottom=0.05,top=0.95,left=0.075,right=0.85)
plt.savefig("FigS3.pdf")
plt.close()

lines = pd.DataFrame(lines, columns=['clade_shift','var_slin','pval'])
