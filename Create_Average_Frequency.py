import sys
sys.path.insert(0,"..")
from util.time import Time
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from flai.util.Time import Time
import json
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from copy import copy
import scipy.integrate as si
import scipy.stats as ss
import scipy.optimize as so
from collections import defaultdict
import copy
import glob
import sys
from time import time
import scipy.integrate as integrate

def sigmoid_func(t,mean,s):
    val = 1 / (1 + np.exp(-s * (t - mean)))
    return val

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

def smooth(times,vec,dt):
    smooth = []
    times_smooth = []
    for i in range(dt,len(times)-dt):
        smooth.append(np.mean(vec[i-dt:i+dt]))
        times_smooth.append(times[i])
    return smooth, times_smooth


WHOlabels = {
'1C.2A.3A.4B':'BETA',
'1C.2A.3A.4A':'EPSILON',
'1C.2A.3A.4C':'IOTA',
'1C.2A.3I':'MU',
'1C.2B.3D':'ALPHA',
'1C.2B.3J':'OMICRON',
'1C.2B.3J.4D':'BA.1',
'1C.2B.3J.4D.5A':'BA.1.1',
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

meta_df = pd.read_csv("DATA/clean_data.txt",sep='\t',index_col=False)   

countries = ['BELGIUM','CA','CANADA','FINLAND','FRANCE','GERMANY','ITALY','NETHERLANDS','NORWAY','NY','SPAIN','SWITZERLAND','USA']
country2voc2freq = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
for country in countries:
    with open("DATA/2023_04_01/freq_traj_" + country.upper() + ".json",'r') as f:
        freq_traj = json.load(f)
    with open("DATA/2023_04_01/multiplicities_" + country.upper() + ".json",'r') as f:
        counts = json.load(f)
    with open("DATA/2023_04_01/multiplicities_Z_" + country.upper() + ".json",'r') as f:
        Z = json.load(f)
   
    x_delta = []
    x_omi = []
    x_alpha = []
    x_ba1 = []
    x_ba2 = []
    x_ba45 = []
    x_bq1 = []
    x_xbb = []
    x_wt = []
    x_ba46 = []
    x_bf7 = [] 
    x_ch1 = []
    x_bn1 = []
    x_bm11 = []
    all_vocs = sorted(list(WHOlabels.values()))
    for t in np.arange(Time.dateToCoordinate("2021-01-01"), Time.dateToCoordinate("2023-04-01")):
        x_vocs = []
        for voc in all_vocs:
            if pango2flupredict[voc] not in freq_traj.keys():
                x_vocs.append(0.0)
                continue
            if str(t) in freq_traj[pango2flupredict[voc]].keys():
                x_vocs.append(freq_traj[pango2flupredict[voc]][str(t)])
            else:
                x_vocs.append(0.0)

        x_delta.append(x_vocs[all_vocs.index('DELTA')])
        x_omi.append(x_vocs[all_vocs.index('OMICRON')])
        x_alpha.append(x_vocs[all_vocs.index('ALPHA')])
        x_ba1.append(x_vocs[all_vocs.index('BA.1')] + x_vocs[all_vocs.index('BA.1.1')])
        x_ba2.append(x_vocs[all_vocs.index('BA.2')] + x_vocs[all_vocs.index('BA.2.12.1')])
        x_ba45.append(x_vocs[all_vocs.index('BA.4')] + x_vocs[all_vocs.index('BA.5')] + x_vocs[all_vocs.index('BA.5.9')])
        x_bq1.append(x_vocs[all_vocs.index('BQ.1')] + x_vocs[all_vocs.index('BQ.1.1')])
        x_xbb.append(x_vocs[all_vocs.index('XBB')] + x_vocs[all_vocs.index('XBB.1.5')])
        x_ba46.append(x_vocs[all_vocs.index('BA.4.6')])
        x_bf7.append(x_vocs[all_vocs.index('BF.7')])
        x_ch1.append(x_vocs[all_vocs.index('CH.1.1')] + x_vocs[all_vocs.index('CH.1')])
        x_bn1.append(x_vocs[all_vocs.index('BN.1')])
        x_bm11.append(x_vocs[all_vocs.index('BM.1.1')])

        voc_here = ['ALPHA','DELTA','BETA','EPSILON','IOTA','MU','OMICRON','GAMMA']
        x_wt.append(1 - np.sum([x_vocs[all_vocs.index(v)] for v in voc_here]))

    country2voc2freq[country]['WT'] = np.array(x_wt)
    country2voc2freq[country]['DELTA'] = np.array(x_delta)
    country2voc2freq[country]['OMI'] = np.array(x_omi) + np.array(x_ba1) + np.array(x_ba2) + np.array(x_ba45) 
    country2voc2freq[country]['ALPHA'] = np.array(x_alpha)
    country2voc2freq[country]['BA.1'] = np.array(x_ba1)
    country2voc2freq[country]['BA.2'] = np.array(x_ba2)
    country2voc2freq[country]['BA.4/5'] = np.array(x_ba45)
    country2voc2freq[country]['BQ.1'] = np.array(x_bq1)
    country2voc2freq[country]['XBB'] = np.array(x_xbb)
    country2voc2freq[country]['BA.4.6'] = np.array(x_ba46)
    country2voc2freq[country]['BF.7'] = np.array(x_bf7)
    country2voc2freq[country]['CH.1'] = np.array(x_ch1)
    country2voc2freq[country]['BN.1'] = np.array(x_bn1)
    country2voc2freq[country]['BM.1.1'] = np.array(x_bm11)
#Average frequencies
vocs = ['WT','DELTA','OMI','ALPHA','BA.1','BA.2','BA.4/5','BQ.1','BA.4.6','BF.7','XBB','CH.1','BN.1','BM.1.1'] 
voc2freq = defaultdict(lambda: [])
for voc in vocs:
    voc2freq[voc] = np.mean([country2voc2freq[c][voc] for c in countries], axis=0)

df = pd.DataFrame(voc2freq)
df.index =np.arange(Time.dateToCoordinate("2021-01-01"), Time.dateToCoordinate("2023-04-01"))
df.to_csv("output/Average_Frequencies.txt",'\t')