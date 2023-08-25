import numpy as np
import pandas as pd
import json
from flai.util.Time import Time
from math import exp
from collections import defaultdict
from flai.util.Utils import Utils
from flai.util.TreeUtils import TreeUtils
import os

def getFrequencyLifeSpan(t0, freqspan):
	return [int(t0 - freqspan), int(t0 + freqspan + 1)]

def getLogMultiplicity(col_date, time, tbase=Utils.INF):
	'''
	Returns the log of node multiplicity at a given time point time
	
	Parameters:
		time: time at which the function is evaluated
		tbase: optional, used if tree is cut after prediction time
	'''
	[t1, t2] = getFrequencyLifeSpan(col_date,33)
	t2 = min(t2, tbase)

	mul = -Utils.INF

	if col_date <= tbase and t1 <= time < t2:
		mul = TreeUtils.weightFunction(col_date, time) + np.log(float(1.0))

	return mul

WHOlabels = {'1C.2B.3D':'ALPHA','1C.2D.3F':'DELTA','1C.2B.3J':'OMICRON','1C.2B.3J.4D':'OMICRON BA.1','1C.2B.3J.4E':'OMICRON BA.2','1C.2A.3A.4B':'BETA',
'1C.2A.3A.4A':'EPSILON','1C.2A.3A.4C':'IOTA','1C.2A.3I':'MU','1C.2B.3G':'GAMMA','1C.2B.3J.4G':'OMICRON BA.4','1C.2B.3J.4H':'OMICRON BA.5'}

region2pos2time2logMult_list = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: [])))
region2time2Z_list = defaultdict(lambda:defaultdict(lambda: []))

df = pd.read_csv("/data/flupredict/SARS/2022_06_22/CladeAssignment.txt",'\t',index_col=False)

# #====country2count
country2count = defaultdict(lambda: 0)
tmin = 44196 #2021-01-01
tmax = 44560
for line in df.iterrows():
	line = line[1]
	country = line.NAME.split("/")[1]
	state = ''
	if country == 'USA':
		state = line.NAME.split("/")[2][:2]

	date = Time.dateToCoordinate(line[3])
	if date < tmax and date > tmin:
		country2count[country] += 1
		if state != '':
			country2count[state] += 1

lines = []
for country in country2count.keys():
	lines.append([country, country2count[country]])
lines = pd.DataFrame(lines,columns=['Country','GISAID_count'])
lines = lines.sort_values(by='GISAID_count',ascending=False)
lines.to_csv("2022_06_22/Country2Count.txt",'\t',index=False)


lines = pd.read_csv("2022_06_22/Country2Count.txt",'\t',index_col=False)
country2count = {}
for l in lines.iterrows():
	l = l[1]
	country2count[l.Country]=l.GISAID_count

country_list = list(lines[:100].Country)


freqspan = 33
std_days = 11
TreeUtils.STDDAYS = 11
TreeUtils.FREQSPAN = 3 * TreeUtils.STDDAYS

for line in df.iterrows():
	line = line[1]
	col_date = Time.dateToCoordinate(line.TIME)
	name = line.NAME
	voc_here = line.CLADE

	if voc_here == 'WT' and col_date > Time.dateToCoordinate("2021-07-01"):
		continue

	country = line.NAME.split("/")[1]
	node_countries = [country]
	state = ''
	if country == 'USA':
		state = line.NAME.split("/")[2][:2]
		if state == 'CA' or state == 'TX' or state == 'NY':
			node_countries.append(state)

	for c in node_countries:
		if c in country_list:
			node_WHO_labels = [WHOlabels[VoC] for VoC in list(WHOlabels.keys()) if VoC in voc_here]
			[t1,t2] = getFrequencyLifeSpan(col_date,33)
			times = np.arange(t1,t2)
			mult_list = [(int(t), getLogMultiplicity(col_date,int(t))) for t in times]	

			for m in mult_list:
				for label in node_WHO_labels:
					region2pos2time2logMult_list[c][label][m[0]].append(m[1])
				region2time2Z_list[c][m[0]].append(m[1])

region2pos2time2logMult = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: [])))
region2time2Z = defaultdict(lambda:defaultdict(lambda: []))
for country in region2pos2time2logMult_list.keys():
	for pos in region2pos2time2logMult_list[country].keys():
		for time in region2pos2time2logMult_list[country][pos].keys():
			region2pos2time2logMult[country][pos][time] = Utils.logSum(region2pos2time2logMult_list[country][pos][time])
for country in region2time2Z_list.keys():
	for time in region2time2Z_list[country].keys():
		region2time2Z[country][time] = Utils.logSum(region2time2Z_list[country][time])

#Normalize
region2pos2time2freq = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
for country in region2pos2time2logMult.keys():
	for pos in region2pos2time2logMult[country].keys():
		for time in region2pos2time2logMult[country][pos].keys():
			region2pos2time2freq[country][pos][time] = np.exp(region2pos2time2logMult[country][pos][time] - region2time2Z[country][time])

for country in region2pos2time2freq.keys():
	with open(os.path.join('2022_06_22','freq_traj_' +country + ".json"),'w') as f:
		json.dump(region2pos2time2freq[country], f)
	with open(os.path.join('2022_06_22','multiplicities_' + country + '.json'),'w') as f:
		json.dump(region2pos2time2logMult[country],f)
	with open(os.path.join('2022_06_22','multiplicities_Z_' + country + '.json'),'w') as f:
		json.dump(region2time2Z[country],f)







