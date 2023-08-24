import numpy as np
import pandas as pd
import json
import sys
sys.path.insert(0,"..")
from util.time import Time
from util.functions import Functions as util_functions
from math import exp
from collections import defaultdict
import os

region2pos2time2logMult_list = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: [])))
region2time2Z_list = defaultdict(lambda:defaultdict(lambda: []))

df = pd.read_csv("CladeAssignment.txt",'\t',index_col=False)

country_list = ['BELGIUM','CA','CANADA','DENMARK','FINLAND','FRANCE','GERMANY',
'IRELAND','ITALY','JAPAN','NETHERLANDS','NORWAY','NY',
'SLOVENIA','SPAIN','SWITZERLAND','USA']

util_functions.STDDAYS = 11
util_functions.FREQSPAN = 3 * util_functions.STDDAYS

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
			[t1,t2] = util_functions.getFrequencyLifeSpan(col_date,33)
			times = np.arange(t1,t2)
			mult_list = [(int(t), util_functions.getLogMultiplicity(col_date,int(t))) for t in times]	

			for m in mult_list:
				region2pos2time2logMult_list[c][voc_here][m[0]].append(m[1])
				region2time2Z_list[c][m[0]].append(m[1])

region2pos2time2logMult = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: [])))
region2time2Z = defaultdict(lambda:defaultdict(lambda: []))
for country in region2pos2time2logMult_list.keys():
	for pos in region2pos2time2logMult_list[country].keys():
		for time in region2pos2time2logMult_list[country][pos].keys():
			region2pos2time2logMult[country][pos][time] = util_functions.logSum(region2pos2time2logMult_list[country][pos][time])
for country in region2time2Z_list.keys():
	for time in region2time2Z_list[country].keys():
		region2time2Z[country][time] = util_functions.logSum(region2time2Z_list[country][time])

#Normalize
region2pos2time2freq = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
for country in region2pos2time2logMult.keys():
	for pos in region2pos2time2logMult[country].keys():
		for time in region2pos2time2logMult[country][pos].keys():
			region2pos2time2freq[country][pos][time] = np.exp(region2pos2time2logMult[country][pos][time] - region2time2Z[country][time])

for country in region2pos2time2freq.keys():
	with open(os.path.join('2023_04_01', 'freq_traj_' +country + ".json"),'w') as f:
		json.dump(region2pos2time2freq[country], f)
	with open(os.path.join('multiplicities_' + country + '.json'),'w') as f:
		json.dump(region2pos2time2logMult[country],f)
	with open(os.path.join('2023_04_01', 'multiplicities_Z_' + country + '.json'),'w') as f:
		json.dump(region2time2Z[country],f)







