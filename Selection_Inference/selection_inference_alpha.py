import sys
sys.path.insert(0,"..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.time import Time
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

VOCs = ['ALPHA','BETA','GAMMA','DELTA','EPSILON','KAPPA','LAMBDA']
colors = ['b','g','r','c','m','y','k','lime','salmon','lime']

print("=====================1 - Alpha ============================")

x_limit = 0.01
dt = 30
min_count = 500

country_min_count = defaultdict(lambda: [])
files = glob.glob("../DATA/2022_04_26/freq_traj_*")
countries = [f.split("_")[-1][:-5] for f in files]
meta_df = pd.read_csv("../DATA/clean_data.txt",sep='\t',index_col=False)
countries.pop(countries.index("WALES"))
countries.pop(countries.index("NORTHERNIRELAND"))
countries.pop(countries.index("SCOTLAND"))
countries.pop(countries.index("ENGLAND")) #No mRNA vaccine in the UK
countries.pop(countries.index("ICELAND"))
countries.pop(countries.index("LUXEMBOURG"))
country2max_recov =   defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: defaultdict())))
country2immuno2time = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))

lines = []
for country in countries:
	with open("../DATA/2022_04_26/freq_traj_" + country.upper() + ".json",'r') as f:
		freq_traj = json.load(f)
	with open("../DATA/2022_04_26/multiplicities_" + country.upper() + ".json",'r') as f:
		counts = json.load(f)
	with open("../DATA/2022_04_26/multiplicities_Z_" + country.upper() + ".json",'r') as f:
		Z = json.load(f)
	meta_country= meta_df.loc[list(meta_df['location'] == country[0] + country[1:].lower())]
	if len(country) <= 3:
		meta_country= meta_df.loc[list(meta_df['location'] == country)]
	meta_country.index = meta_country['FLAI_time']


	if max(list(freq_traj['ALPHA'].values())) > 0.5:
		dates_alpha = list(counts['ALPHA'].keys())
		dates_alpha = [int(a) for a in dates_alpha]
		dates_alpha = sorted(dates_alpha)
		dates_alpha = [a for a in dates_alpha if a < 44470]
		tmin = min(dates_alpha)
		tmax = max(dates_alpha)
		t_range = np.arange(tmin,tmax)
		alpha_count = [int(np.exp(counts['ALPHA'][str(a)])) for a in t_range]
		wt_count = [int(np.exp(Z[str(a)])) - int(np.exp(counts['ALPHA'][str(a)])) for a in t_range]
		N_tot = np.array(alpha_count) + np.array(wt_count)
		alpha_freq = np.array(alpha_count) / np.array(N_tot)
		check = 'not_okay'
		for t in t_range:
			x_wt = wt_count[t-tmin] / N_tot[t-tmin]
			x_a = alpha_count[t-tmin] / N_tot[t-tmin]
			if x_a > x_limit:
				tminnew = t
				check = 'okay'
				break
		for t in t_range:
			x_wt = wt_count[t-tmin] / N_tot[t-tmin]
			x_a = alpha_count[t-tmin] / N_tot[t-tmin]
			if x_a > max(alpha_freq)-0.01:
				tmaxnew = t
				break
		tmin = tminnew
		tmax = tmaxnew
		t_range= np.arange(tmin,tmax)

		Ztot =np.array([int(np.exp(Z[str(t)])) for t in t_range])
		country_min_count[country].append(min(Ztot))
		if np.sum(Ztot > min_count) != len(Ztot):
			print(f"{country} drops out due to insufficient count")
			continue

		x_alpha = []
		x_wt = []
		freq_wt_dict = defaultdict(lambda: 0.0)
		for t in list(meta_country.index):
			if str(t) in freq_traj['ALPHA'].keys():
				x_alpha.append(freq_traj['ALPHA'][str(t)])
			else:
				x_alpha.append(0.0)
			x_voc = 0.0
			for voc in VOCs:
				if voc in freq_traj.keys():
					if str(t) in freq_traj[voc].keys():
						x_voc += freq_traj[voc][str(t)]
			x_wt.append(1-x_voc)
			freq_wt_dict[str(t)] = 1 - x_voc

		freq_alpha = np.array(x_alpha)
		freq_wt = np.array(x_wt)

		cases_full = [meta_country.loc[t]['new_cases']/meta_country.loc[t]['population'] for t in list(meta_country.index)]
		cases_full = clean_vac(list(meta_country.index),cases_full)
		country2immuno2time[country]['CASES'] = {list(meta_country.index)[i]:cases_full[i] for i in range(len(list(meta_country.index)))}
		cases_full_alpha = cases_full * freq_alpha
		recov_alpha = [[np.sum(cases_full_alpha[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['ALPHA'] = {a[1]: a[0] for a in recov_alpha}
		cases_full_wt = cases_full * freq_wt
		recov_wt = [[np.sum(cases_full_wt[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['WT'] = {a[1]: a[0] for a in recov_wt}
		vaccinated = [meta_country.loc[t]['people_fully_vaccinated_per_hundred']/100. for t in t_range]
		if not np.sum([np.isnan(a) for a in vaccinated]) == len(vaccinated):
			vaccinated = clean_vac(t_range,vaccinated)
			country2immuno2time[country]['VAC'] = {t_range[i] : vaccinated[i] for i in range(len(t_range))}

		alpha_count = [int(np.exp(counts['ALPHA'][str(a)])) for a in t_range]
		VOC_counts = defaultdict(lambda: [])
		for t in t_range:
			for VOC in VOCs:
				if VOC in counts.keys():
					if str(t) in counts[VOC].keys():
						VOC_counts[VOC].append(int(np.exp(counts[VOC][str(t)])))
					else:
						VOC_counts[VOC].append(0.0)

		wt_count = np.array([int(np.exp(Z[str(a)])) for a in t_range])
		for VOC in VOC_counts.keys():
			wt_count = wt_count - np.array(VOC_counts[VOC])
		wt_count = wt_count - np.array(alpha_count)

		t1 = 0 
		t2 = t1 + dt
		while t2 + tmin < tmax:
			N_tot1 = alpha_count[t1] + wt_count[t1]
			N_tot2 = alpha_count[t2] + wt_count[t2]
			if alpha_count[t1] < 10 or alpha_count[t2] < 10 or wt_count[t1] < 10 or wt_count[t2] < 10:
				t1 = t1 + 7
				t2 = t1 + dt
				print("t<10",country)
				continue
			p_t1 = [alpha_count[t1] / N_tot1, wt_count[t1] / N_tot1]
			p_t2 = [alpha_count[t2] / N_tot2, wt_count[t2] / N_tot2]


			x_alpha_hat_t1 = np.random.binomial(N_tot1,p_t1[0],1000) / N_tot1
			x_other_hat_t1 = np.ones(len(x_alpha_hat_t1)) - x_alpha_hat_t1
			x_alpha_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_alpha_hat_t1])
			x_other_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_other_hat_t1])

			x_alpha_hat_t2 = np.random.binomial(N_tot2,p_t2[0],1000) / N_tot2
			x_other_hat_t2 = np.ones(len(x_alpha_hat_t2)) - x_alpha_hat_t2
			x_alpha_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_alpha_hat_t2])
			x_other_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_other_hat_t2])

			result = (np.log(x_alpha_hat_t2/x_other_hat_t2) - np.log(x_alpha_hat_t1 / x_other_hat_t1)) / float(dt)

			s_hat = np.mean(result)
			s_hat_var = np.var(result)

			FLAI_time = int((t1 + t2 + 2 * tmin)/2)
			
			if str(FLAI_time) in freq_traj['ALPHA'].keys():
				x_alpha = freq_traj['ALPHA'][str(FLAI_time)]
			else:
				x_alpha = 0.0
			x_voc = 0.0
			for voc in VOCs:
				if voc in freq_traj.keys():
					if str(FLAI_time) in freq_traj[voc].keys():
						x_voc += freq_traj[voc][str(FLAI_time)]
			x_wt = 1-x_voc
			x_voc = 1 - x_wt - x_alpha

			lines.append([country,FLAI_time,Time.coordinateToStringDate(int(t1+tmin)),Time.coordinateToStringDate(int(t2+tmin)), np.round(s_hat,3), 
				np.round(s_hat_var,7), alpha_count[t1], alpha_count[t2],wt_count[t1],wt_count[t2],tmin,tmax, x_wt, x_alpha, x_voc])

			t1 = t1 + 7
			t2 = t1 + dt
	else:
		country_min_count[country].append(0)

lines = pd.DataFrame(lines,columns = ['country','FLAI_time','t1','t2','s_hat','s_var','alpha_count_t1','alpha_count_t2','wt_count_t1','wt_count_t2','tmin','tmax','x_wt','x_alpha','x_voc'])
country2pop = []
for country in list(set(lines.country)):
	df_c = lines.loc[list(lines.country == country)]
	if len(df_c) < 4:
		country2pop.append(country)
mask = [c not in country2pop for c in list(lines.country)]
lines = lines.loc[mask]
savename = '../output/s_hat_alpha_wt.txt'
lines.to_csv(savename,'\t',index=False)