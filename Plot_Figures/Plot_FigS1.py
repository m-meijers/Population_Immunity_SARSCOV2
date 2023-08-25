import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from matplotlib.lines import Line2D
import scipy.integrate as si
import scipy.stats as ss
import scipy.optimize as so
import json
import sys
sys.path.insert(0,"..")
from util.time import Time
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import matplotlib as mpl



month_dict = {1:'Jan.',2:'Feb.',3:'Mar.',4:'Apr.',5:'May',6:'Jun.',7:'Jul.',8:'Aug.',9:'Sept.',10:'Oct.',11:'Nov.',12:'Dec.'}

flupredict2color={ "1": "#004E98" ,
		 "1C.2A.3A.4B": "#872853",
		 "1C.2B.3D": "#CF48B1",
		 "1C.2B.3G": "#F0E68C",
		 "1C.2B.3J": "#7400B8",
		 "1C.2B.3J.4E": "#BC6C25",
		 "1C.2B.3J.4E.5B": "#B08968",
		 "1C.2B.3J.4E.5N": "#FF9F1C",
		 "1C.2B.3J.4E.5N.6J": "#D3D3D3",
		 "1C.2B.3J.4E.5C": "#BA181B",
		 "1C.2B.3J.4E.5C.6A": "#1F618D",
		 "1C.2B.3J.4E.5C.6I.7C": "#C08552",
		 "1C.2B.3J.4E.5C.6F": "#D39DC0",
		 '1C.2B.3J.4E.5C.6E':"#FF006E",
		 "1C.2B.3J.4D": "#DDA15E",
		 "1C.2B.3J.4D.5A": "#FF69B4",
		 "1C.2B.3J.4F": "#6FBA78",
		 "1C.2B.3J.4F.5D": "#335C67",
		 "1C.2B.3J.4G": "#6FBA78",
		 "1C.2B.3J.4G.5E": "#FF74FD",
		 "1C.2B.3J.4G.5F": "#AFFC41",
		 "1C.2B.3J.4G.5F.6B": "#ECF39E",
		 "1C.2B.3J.4H": "#5E2F0B",
		 "1C.2D.3F": "#FF0040"}

WHOlabels = {
'1C.2A.3A.4B':'BETA',
'1C.2A.3A.4A':'EPSILON',
'1C.2A.3A.4C':'IOTA',
'1C.2A.3I':'MU',
'1C.2B.3D':'ALPHA',
'1C.2B.3J':'OMICRON',
'1C.2B.3J.4D':'BA.1',
# '1C.2B.3J.4D.5A':'BA.1.1',
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

df_model = pd.read_csv("../output/data_immune_trajectories.txt",'\t',index_col=False)

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


ratio = 0.5
fs=7
lp = 1.5
ls = 7
rot=20
lw = 1

inch2cm = 2.54
ratio = 1/1.1

mpl.rcParams['axes.linewidth'] = 0.3 #set the value globally
plt.rcParams.update({'font.sans-serif':'Helvetica'})
gamma_vac = 1.2
gamma_inf = 2.4

countries = ['BELGIUM','CA','CANADA','FINLAND','FRANCE','GERMANY','ITALY','NETHERLANDS','NORWAY','NY','SPAIN','SWITZERLAND','USA','JAPAN']

# fig = plt.figure(figsize=(18.3/inch2cm,20/inch2cm))

countries_da = countries
countries_od = countries

plt.figure(figsize= (18.3/inch2cm, 4 * len(countries_da) / inch2cm))
index = 1
for c in countries_da:
	df_c = df_model.loc[list(df_model.country == c)]
	df_c_s = df_s_da.loc[list(df_s_da.country==c)]
	df_c['x_OMICRON'] = df_c['x_BA.1'] + df_c['x_BA.1.1'] + df_c['x_BA.2'] + df_c['x_BA.2.12.1']
	df_c['x_wt'] = 1 - df_c['x_ALPHA'] - df_c['x_DELTA'] - df_c['x_OMICRON']
	df_c = df_c.loc[df_c.time > Time.dateToCoordinate('2021-02-01')]
	t0 = df_c.loc[df_c['x_DELTA']>0.01].iloc[0].time
	tf = df_c.loc[df_c['x_ALPHA']>0.01].iloc[-1].time
	t_range = np.arange(t0,tf)
	df_c = df_c.loc[(df_c.time < tf) & (df_c.time >= t0)]


	ax = plt.subplot(len(countries_da), 4, index)
	plt.plot(t_range, df_c.x_ALPHA, color = flupredict2color[pango2flupredict['ALPHA']],linewidth=lw)
	plt.plot(t_range, df_c.x_DELTA, color = flupredict2color[pango2flupredict['DELTA']],linewidth=lw)
	plt.plot(t_range, df_c.x_wt , color = flupredict2color['1'],linewidth=lw)

	plt.ylabel("Frequency, $x(t)$",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.02,1.02])
	ax.set_yticks([0,0.5,1.0],['0.0','0.5','1.0'],fontsize=ls)
	plt.title(c,fontsize=fs)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	# xtick_pos = list(np.arange(t_range[0],t_range[-1],60)) + [t_range[-1]]
	xtick_pos = [t_range[0], t_range[int(len(t_range)/2)],t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
	ax.tick_params(direction="in",width=0.3,labelsize=ls)

	index += 1
	ax = plt.subplot(len(countries_da), 4, index)
	plt.plot(t_range, df_c.R_vac,'-',color='#00B4D8')
	plt.plot(t_range,df_c.R_alpha,'-',color=flupredict2color[pango2flupredict['ALPHA']])
	plt.ylabel("Coverage, $y_k(t)$",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.02,1.02])
	ax.set_yticks([0,0.5,1.0],['0.0','0.5','1.0'],fontsize=ls)
	plt.title(c,fontsize=fs)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = [t_range[0], t_range[int(len(t_range)/2)],t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
	ax.tick_params(direction="in",width=0.3,labelsize=ls)
	legend_elements = []
	legend_elements.append(Line2D([],[],marker='',color='#00B4D8',linestyle='-',linewidth=1.0,label='Vaccination'))
	legend_elements.append(Line2D([],[],marker='',color=flupredict2color[pango2flupredict['ALPHA']],linestyle='-',linewidth=1.0,label='Alpha recovery'))
	plt.legend(handles = legend_elements,loc='upper left', prop={'size':ls-1})

	index += 1
	ax = plt.subplot(len(countries_da), 4, index)
	plt.plot(t_range, df_c.C_VAC_ALPHA, '-',color=flupredict2color[pango2flupredict['ALPHA']])
	plt.plot(t_range, df_c.C_VAC_DELTA, '-',color=flupredict2color[pango2flupredict['DELTA']])
	plt.plot(t_range, df_c.C_RECOV_ALPHA_ALPHA,'--',color=flupredict2color[pango2flupredict['ALPHA']])
	plt.plot(t_range, df_c.C_RECOV_ALPHA_DELTA,'--',color=flupredict2color[pango2flupredict['DELTA']])
	plt.fill_between(t_range, df_c.C_VAC_DELTA,df_c.C_VAC_ALPHA,color=flupredict2color[pango2flupredict['DELTA']],alpha=0.3,linewidth=0.0)
	plt.fill_between(t_range, df_c.C_RECOV_ALPHA_DELTA,df_c.C_RECOV_ALPHA_ALPHA,color=flupredict2color[pango2flupredict['DELTA']],alpha=0.3,linewidth=0.0)
	legend_elements = []
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle='-',linewidth=1.0,label='Vaccination'))
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle='--',linewidth=1.0,label='Alpha recovery'))
	plt.legend(handles = legend_elements,loc='upper left', prop={'size':ls-1})

	plt.ylabel("Population Immunity",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.02,0.6])
	plt.title(c,fontsize=fs)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = [t_range[0], t_range[int(len(t_range)/2)],t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
	ax.tick_params(direction="in",width=0.3,labelsize=ls)

	index += 1
	ax = plt.subplot(len(countries_da), 4, index)
	times = np.array(df_c_s.FLAI_time) 
	plt.errorbar(times,np.array(df_c_s.s_hat) - np.mean(df_c_s.s_hat), np.sqrt(np.array(df_c_s.s_var)) * 1.96,alpha=0.7,color='k', fmt='o-',markersize=3)

	F_alpha = -gamma_vac * np.array(df_c.C_VAC_ALPHA) - gamma_inf * np.array(df_c.C_RECOV_ALPHA_ALPHA) - gamma_inf * np.array(df_c.C_RECOV_DELTA_ALPHA)
	F_delta = -gamma_vac * np.array(df_c.C_VAC_DELTA) - gamma_inf * np.array(df_c.C_RECOV_ALPHA_DELTA) - gamma_inf * np.array(df_c.C_RECOV_DELTA_DELTA)
	plt.plot(t_range,F_delta - F_alpha - np.mean(F_delta - F_alpha),'r--')
	plt.title(c,fontsize=fs)
	plt.ylabel("Selection coefficient,$\\Delta s_{\\delta \\alpha}$",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.05,0.05])
	ax.set_yticks([-0.04,-0.02,0.0,0.02,0.04],['-0.04','-0.02','0.0','0.02','0.04'],fontsize=ls)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = [t_range[0], t_range[int(len(t_range)/2)],t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
	ax.tick_params(direction="in",width=0.3,labelsize=ls)	

	index += 1

plt.subplots_adjust(bottom=0.01,top=0.99,left=0.05,right=0.95,hspace=0.05,wspace=0.5)
plt.savefig("FigS1a.pdf")
plt.close()


ratio = 1 / 1.1
ratio2 = ratio / 2.1
gamma_vac = 0.29
gamma_inf = 0.689
fig = plt.figure(figsize= (18.3/inch2cm, 4 * len(countries_od) / inch2cm))
gs0 = gridspec.GridSpec(len(countries_od),4,figure=fig,wspace=0.5)
index = 0
for c in countries_od:
	df_c = df_model.loc[list(df_model.country == c)]
	df_c_s = df_s_od.loc[list(df_s_od.country==c)]
	df_c['x_OMICRON'] = df_c['x_BA.1'] + df_c['x_BA.1.1'] + df_c['x_BA.2'] + df_c['x_BA.2.12.1']
	df_c['x_wt'] = 1 - df_c['x_ALPHA'] - df_c['x_DELTA'] - df_c['x_OMICRON']

	df_c = df_c.loc[df_c.time > Time.dateToCoordinate('2021-09-01')]
	t0 = df_c.loc[df_c['x_OMICRON']>0.01].iloc[0].time
	tf = df_c.loc[df_c['x_DELTA']>0.01].iloc[-1].time
	t_range = np.arange(t0,tf)
	df_c = df_c.loc[(df_c.time < tf) & (df_c.time >= t0)]
	

	# ax = plt.subplot(len(countries_od), 3, index)

	ax = fig.add_subplot(gs0[index,0])

	plt.plot(t_range, df_c.x_OMICRON, color=flupredict2color[pango2flupredict['BA.1']],linewidth=lw)
	plt.plot(t_range, df_c.x_DELTA, color=flupredict2color[pango2flupredict['DELTA']],linewidth=lw)
	plt.ylabel("Frequency, $x(t)$",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.02,1.02])
	ax.set_yticks([0,0.5,1.0],['0.0','0.5','1.0'],fontsize=ls)
	plt.title(c,fontsize=fs)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = [t_range[0], t_range[int(len(t_range)/2)],t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
	ax.tick_params(direction="in",width=0.3,labelsize=ls)

	gs00 = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs0[index,1],hspace=0.00005)
	ax = fig.add_subplot(gs00[0,0])
	t_range = list(df_c.time)
	plt.plot(t_range, df_c.R_vac,'-',color='#00B4D8')
	plt.plot(t_range, df_c.R_boost,'-',color='#0077B6')
	ax.set_ylim([-0.02,1.0])
	plt.title(c,fontsize=fs)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = [t_range[0], t_range[int(len(t_range)/2)],t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append('')
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio2)
	ax.tick_params(direction="in",width=0.3,labelsize=ls)
	legend_elements = []
	legend_elements.append(Line2D([],[],marker='',color='#00B4D8',linestyle='-',linewidth=1.0,label='Booster'))
	legend_elements.append(Line2D([],[],marker='',color='#0077B6',linestyle='-',linewidth=1.0,label='Vaccination'))
	plt.legend(handles = legend_elements,loc='upper left', prop={'size':ls-2})

	ax = fig.add_subplot(gs00[1,0])
	plt.plot(t_range, df_c.R_delta,'-',color= flupredict2color[pango2flupredict['DELTA']])
	plt.plot(t_range, df_c.R_omi,'-',color=flupredict2color[pango2flupredict['BA.1']])
	plt.ylabel("Coverage, $y_k(t)$",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.02,0.6])
	legend_elements = []
	legend_elements.append(Line2D([],[],marker='',color=flupredict2color[pango2flupredict['DELTA']],linestyle='-',linewidth=1.0,label='Delta recovery'))
	legend_elements.append(Line2D([],[],marker='',color=flupredict2color[pango2flupredict['BA.1']],linestyle='-',linewidth=1.0,label='Omicron recovery'))
	plt.legend(handles = legend_elements,loc='upper left', prop={'size':ls-2})
	# ax.set_yticks([0,0.5,1.0],['0.0','0.5','1.0'],fontsize=ls)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = [t_range[0], t_range[int(len(t_range)/2)],t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio2)
	ax.tick_params(direction="in",width=0.3,labelsize=ls)

	
	# ax = fig.add_subplot(gs0[index,1])
	gs00 = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs0[index,2],hspace=0.00005)
	ax = fig.add_subplot(gs00[0,0])
	t_range = list(df_c.time)
	plt.plot(t_range, df_c.C_VAC_DELTA,  '-',color = flupredict2color[pango2flupredict['DELTA']],linewidth=lw)
	plt.plot(t_range, df_c.C_VAC_OMICRON,   '-',color = flupredict2color[pango2flupredict['BA.1']],linewidth=lw)
	plt.plot(t_range, df_c.C_BOOST_DELTA, '--',color = flupredict2color[pango2flupredict['DELTA']],linewidth=lw)
	plt.plot(t_range, df_c.C_BOOST_OMICRON,  '--',color = flupredict2color[pango2flupredict['BA.1']],linewidth=lw)
	plt.fill_between(t_range, df_c.C_VAC_OMICRON,df_c.C_VAC_DELTA,color=flupredict2color[pango2flupredict['BA.1']],alpha=0.3,linewidth=0.0)
	plt.fill_between(t_range, df_c.C_BOOST_OMICRON,df_c.C_BOOST_DELTA,color=flupredict2color[pango2flupredict['BA.1']],alpha=0.3,linewidth=0.0)
	ax.set_ylim([-0.02,0.6])
	plt.title(c,fontsize=fs)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = [t_range[0], t_range[int(len(t_range)/2)],t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append('')
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio2)
	ax.tick_params(direction="in",width=0.3,labelsize=ls)
	legend_elements = []
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle='-',linewidth=1.0,label='Vaccination'))
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle='--',linewidth=1.0,label='Booster'))
	plt.legend(handles = legend_elements,loc='upper left', prop={'size':ls-2})

	ax = fig.add_subplot(gs00[1,0])
	plt.plot(t_range, df_c.C_RECOV_DELTA_DELTA, '-',color = flupredict2color[pango2flupredict['DELTA']],linewidth=lw)
	plt.plot(t_range, df_c.C_RECOV_DELTA_OMICRON,  '-',color = flupredict2color[pango2flupredict['BA.1']],linewidth=lw)
	plt.plot(t_range, df_c.C_RECOV_OMI_DELTA,  '--',color = flupredict2color[pango2flupredict['DELTA']],linewidth=lw)
	plt.plot(t_range, df_c['C_RECOV_OMI_BA.1'],   '--',color = flupredict2color[pango2flupredict['BA.1']],linewidth=lw)
	plt.fill_between(t_range, df_c.C_RECOV_DELTA_OMICRON,df_c.C_RECOV_DELTA_DELTA,color=flupredict2color[pango2flupredict['BA.1']],alpha=0.3,linewidth=0.0)
	plt.fill_between(t_range, df_c.C_RECOV_OMI_DELTA,df_c['C_RECOV_OMI_BA.1'],color=flupredict2color[pango2flupredict['DELTA']],alpha=0.3,linewidth=0.0)
	plt.ylabel("Population Immunity",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.02,0.6])
	legend_elements = []
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle='-',linewidth=1.0,label='Delta recovery'))
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle='--',linewidth=1.0,label='Omicron recovery'))
	plt.legend(handles = legend_elements,loc='upper left', prop={'size':ls-2})
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = [t_range[0], t_range[int(len(t_range)/2)],t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio2)
	ax.tick_params(direction="in",width=0.3,labelsize=ls)

	
	ax = fig.add_subplot(gs0[index,3])
	times = np.array(df_c_s.FLAI_time) 
	plt.errorbar(times,np.array(df_c_s.s_hat) - np.mean(df_c_s.s_hat), np.sqrt(np.array(df_c_s.s_var)) * 1.96,alpha=0.7,color='k', fmt='o-',markersize=3)
	F_delta = -gamma_vac * np.array(df_c.C_VAC_DELTA) -gamma_vac * np.array(df_c.C_BOOST_DELTA) - gamma_inf * np.array(df_c.C_RECOV_DELTA_DELTA) - gamma_inf * np.array(df_c.C_RECOV_OMI_DELTA)
	F_omi = -gamma_vac * np.array(df_c.C_VAC_OMICRON) -gamma_vac * np.array(df_c.C_BOOST_OMICRON) - gamma_inf * np.array(df_c.C_RECOV_DELTA_OMICRON) - gamma_inf * np.array(df_c['C_RECOV_OMI_BA.1'])
	plt.plot(t_range,F_omi - F_delta - np.mean(F_omi - F_delta),'r--')
	plt.title(c,fontsize=fs)
	plt.ylabel("Selection coefficient,$\\Delta s_{o \\delta}$",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.05,0.05])
	ax.set_yticks([-0.04,-0.02,0.0,0.02,0.04],['-0.04','-0.02','0.0','0.02','0.04'],fontsize=ls)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = [t_range[0], t_range[int(len(t_range)/2)],t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
	ax.tick_params(direction="in",width=0.3,labelsize=ls)	
	index += 1

plt.subplots_adjust(bottom=0.01,top=0.99,left=0.05,right=0.95)

plt.savefig("FigS1b.pdf")
plt.close()

