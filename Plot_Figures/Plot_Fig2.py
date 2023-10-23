import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from util.time import Time

flupredict2color = {"1": "#004E98",
                    "1C.2A.3A.4B": "#872853",
                    "1C.2B.3D": "#CF48B1",
                    "1C.2B.3G": "#F0E68C",
                    "1C.2B.3J": "#7400B8",
                    "1C.2B.3J.4E": "#1E90FF",
                    "1C.2B.3J.4E.5B": "#B08968",
                    "1C.2B.3J.4E.5N": "#B1A7A6",
                    "1C.2B.3J.4E.5N.6J": "#D3D3D3",
                    "1C.2B.3J.4E.5C": "#BA181B",
                    "1C.2B.3J.4E.5C.6A": "#1F618D",
                    "1C.2B.3J.4E.5C.6I.7C": "#C08552",
                    "1C.2B.3J.4E.5C.6F": "#D39DC0",
                    "1C.2B.3J.4D": "#4CC9F0",
                    "1C.2B.3J.4D.5A": "#FF69B4",
                    "1C.2B.3J.4F": "#6FBA78",
                    "1C.2B.3J.4F.5D": "#344E41",
                    "1C.2B.3J.4G": "#6FBA78",
                    "1C.2B.3J.4G.5E": "#00AFB9",
                    "1C.2B.3J.4G.5F": "#AFFC41",
                    "1C.2B.3J.4G.5F.6B": "#ECF39E",
                    "1C.2B.3J.4H": "#5E2F0B",
                    "1C.2D.3F": "#FF0040"}
WHOlabels = {'1C.2A.3A.4B': 'BETA',
             '1C.2A.3A.4A': 'EPSILON',
             '1C.2A.3A.4C': 'IOTA',
             '1C.2A.3I': 'MU',
             '1C.2B.3D': 'ALPHA',
             '1C.2B.3J': 'OMICRON',
             '1C.2B.3J.4D': 'BA.1',
             '1C.2B.3J.4D.5A': 'BA.1.1',
             '1C.2B.3J.4E': 'BA.2',
             '1C.2B.3J.4E.5B': 'BA.2.12.1',
             '1C.2B.3J.4E.5L': 'BJ.1',
             '1C.2B.3J.4E.5C': 'BA.2.75',
             '1C.2B.3J.4E.5C.6A': 'BA.2.75.2',
             '1C.2B.3J.4E.5C.6E': 'BM.1.1',
             '1C.2B.3J.4E.5C.6F': 'BN.1',
             '1C.2B.3J.4F': 'BA.4',
             '1C.2B.3J.4F.5D': 'BA.4.6',
             '1C.2B.3J.4G': 'BA.5',
             '1C.2B.3J.4G.5K': 'BA.5.9',
             '1C.2B.3J.4G.5E': 'BF.7',
             '1C.2B.3J.4G.5F': 'BQ.1',
             '1C.2B.3J.4G.5F.6B': 'BQ.1.1',
             '1C.2D.3F': 'DELTA',
             '1C.2B.3G': 'GAMMA',
             '1C.2B.3J.4E.5N': 'XBB',
             '1C.2B.3J.4E.5N.6J': 'XBB.1.5',
             '1C.2B.3J.4E.5C.6I': 'CH.1',
             '1C.2B.3J.4E.5C.6I.7C': 'CH.1.1'}

pango2flupredict = {a: b for b, a in WHOlabels.items()}

df = pd.read_csv("../output/data_immune_trajectories.txt", '\t',
                 index_col=False)
df_R = pd.read_csv("../output/R_average.txt", '\t', index_col=False)
df_R.loc[:, 'x_BA.1'] = df_R['x_BA.1'] + df_R['x_BA.1.1']
df_R.loc[:, 'x_BA.2'] = df_R['x_BA.2'] + df_R['x_BA.2.12.1']
df_R.loc[:, 'x_BA.4/5'] = df_R['x_BA.4'] + df_R['x_BA.5'] + df_R['x_BA.5.9']
df_R.loc[:, 'x_BQ.1'] = df_R['x_BQ.1'] + df_R['x_BQ.1.1']
df_R.loc[:, 'x_XBB'] = df_R['x_XBB'] + df_R['x_XBB.1.5']
df_R.loc[:, 'x_CH.1'] = df_R['x_CH.1'] + df_R['x_CH.1.1']
df_R.loc[:, 'x_wt'] = 1 - df_R['x_ALPHA'] - df_R['x_DELTA']\
    - df_R['x_BETA'] - df_R['x_EPSILON'] - df_R['x_IOTA']\
    - df_R['x_MU'] - df_R['x_OMICRON'] - df_R['x_GAMMA']\
    - df_R['x_BA.1'] - df_R['x_BA.2'] - df_R['x_BA.4/5']\
    - df_R['x_BQ.1'] - df_R['x_XBB']
df_R.loc[:, 'x_OMICRON'] = df_R['x_BA.1'] + df_R['x_BA.2']\
    + df_R['x_BA.2.75'] + df_R['x_BA.4/5']
df_R.index = df_R.time

countries = sorted(list(set(df.country)))

times = list(df_R.time)

vocs = ['wt', 'ALPHA', 'DELTA', 'BA.1', 'BA.2', 'BA.4/5', 'BQ.1', 'XBB']
voc2era = defaultdict(lambda: [])
for voc in vocs:
    for line in df_R.iterrows():
        line = line[1]
        x = line[f'x_{voc}']
        era = True
        for voc2 in vocs:
            if voc2 == voc:
                continue
            if x < line[f'x_{voc2}']:
                era = False
        if era:
            voc2era[voc].append(line.time)

ls = 10
fs = 12
ratio = 0.2
alpha_back = 0.15
plt.figure(figsize=(14, 5))
ax = plt.subplot(211)

era2color = {'wt': flupredict2color['1'],
             'ALPHA': flupredict2color[pango2flupredict['ALPHA']],
             'DELTA': flupredict2color[pango2flupredict['DELTA']],
             'BA.1': flupredict2color[pango2flupredict['BA.1']],
             'BA.2': flupredict2color[pango2flupredict['BA.2']],
             'BA.4/5': flupredict2color[pango2flupredict['BA.5']],
             'BQ.1': flupredict2color[pango2flupredict['BQ.1']],
             'XBB': flupredict2color[pango2flupredict['XBB']]}
for voc in ['ALPHA', 'DELTA', 'BA.1', 'BA.2', 'BA.4/5', 'BQ.1']:
    plt.barh(1.07, width=max(voc2era[voc]) - min(voc2era[voc]),
             height=0.05,
             left=min(voc2era[voc]),
             color=era2color[voc])
plt.axhline(0, color='k', alpha=0.5)

# Vaccination types
for c in countries:
    plt.plot(df.loc[df.country == c, 'time'],
             list(df.loc[df.country == c, 'C_VAC_ALPHA']),
             color=flupredict2color[pango2flupredict['ALPHA']],
             alpha=alpha_back)
    plt.plot(df.loc[df.country == c, 'time'],
             list(df.loc[df.country == c, 'C_VAC_DELTA']),
             color=flupredict2color[pango2flupredict['DELTA']],
             alpha=alpha_back)
    plt.plot(df.loc[df.country == c, 'time'],
             list(df.loc[df.country == c, 'C_VAC_OMICRON']),
             color=flupredict2color[pango2flupredict['BA.1']],
             alpha=alpha_back)
plt.plot(times,
         df_R['C_VAC_ALPHA'],
         color=flupredict2color[pango2flupredict['ALPHA']], linewidth=2.)
plt.plot(times,
         df_R['C_VAC_DELTA'],
         color=flupredict2color[pango2flupredict['DELTA']], linewidth=2.)
plt.plot(times,
         df_R['C_VAC_OMICRON'],
         color=flupredict2color[pango2flupredict['BA.1']], linewidth=2.)

# Booster types
for c in countries:
    df_c = df.loc[(df.country == c)
                  & (df.time > Time.dateToCoordinate("2021-10-01"))]
    plt.plot(df_c.time, df_c.C_BOOST_ALPHA,
             color=flupredict2color[pango2flupredict['ALPHA']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c.C_BOOST_DELTA,
             color=flupredict2color[pango2flupredict['DELTA']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_BOOST_BA.1'],
             color=flupredict2color[pango2flupredict['BA.1']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_BOOST_BA.2'],
             color=flupredict2color[pango2flupredict['BA.2']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_BOOST_BA.4/5'],
             color=flupredict2color[pango2flupredict['BA.5']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_BOOST_BQ.1'],
             color=flupredict2color[pango2flupredict['BQ.1']],
             alpha=alpha_back)
df_c = df_R.loc[df_R.time > Time.dateToCoordinate("2021-10-01")]
plt.plot(df_c.time, df_c.C_BOOST_ALPHA,
         color=flupredict2color[pango2flupredict['ALPHA']], linewidth=2.)
plt.plot(df_c.time, df_c.C_BOOST_DELTA,
         color=flupredict2color[pango2flupredict['DELTA']], linewidth=2.)
plt.plot(df_c.time, df_c['C_BOOST_BA.1'],
         color=flupredict2color[pango2flupredict['BA.1']], linewidth=2.)
plt.plot(df_c.time, df_c['C_BOOST_BA.2'],
         color=flupredict2color[pango2flupredict['BA.2']], linewidth=2.)
plt.plot(df_c.time, df_c['C_BOOST_BA.4/5'],
         color=flupredict2color[pango2flupredict['BA.5']], linewidth=2.)
plt.plot(df_c.time, df_c['C_BOOST_BQ.1'],
         color=flupredict2color[pango2flupredict['BQ.1']], linewidth=2.)

# Bivalent types
for c in countries:
    df_c = df.loc[(df.country == c)
                  & (df.time > Time.dateToCoordinate("2022-09-01"))]
    plt.plot(df_c.time, df_c['C_BIVALENT_BA.2'],
             color=flupredict2color[pango2flupredict['BA.2']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_BIVALENT_BA.4/5'],
             color=flupredict2color[pango2flupredict['BA.5']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_BIVALENT_BQ.1'],
             color=flupredict2color[pango2flupredict['BQ.1']],
             alpha=alpha_back)
df_c = df_R.loc[(df_R.time > Time.dateToCoordinate("2022-09-01"))]
plt.plot(df_c.time, df_c['C_BIVALENT_BA.2'],
         color=flupredict2color[pango2flupredict['BA.2']],
         linewidth=2.)
plt.plot(df_c.time, df_c['C_BIVALENT_BA.4/5'],
         color=flupredict2color[pango2flupredict['BA.5']],
         linewidth=2.)
plt.plot(df_c.time, df_c['C_BIVALENT_BQ.1'],
         color=flupredict2color[pango2flupredict['BQ.1']],
         linewidth=2.)

xtick_labels = ['2021-01-01', '2021-05-01', '2021-09-01', '2022-01-01',
                '2022-05-01', '2022-09-01', '2023-01-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Jan. $\'$21', 'May $\'$ 21', 'Sep. $\'$21',
                'Jan. $\'$22', 'May $\'$22', 'Sep. $\'$22',
                'Jan. $\'$23']
plt.xticks(xtick_pos, xtick_labels, rotation=0, ha='center', fontsize=fs)
plt.xlim([Time.dateToCoordinate("2021-01-01"),
          Time.dateToCoordinate("2023-03-01") - 1])
plt.ylim([-0.05, 0.8])
plt.ylabel("Fitness, $f_i(t)$", fontsize=fs)
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
plt.tick_params(direction='in', labelsize=ls)


ax = plt.subplot(212)
alpha_time = min(voc2era['ALPHA'])
ba2_time = min(voc2era['BA.2'])
# Alpha recovery
for c in countries:
    df_c = df.loc[(df.country == c) & (df.time > alpha_time)
                  & (df.time < ba2_time)]
    plt.plot(df_c.time, df_c['C_RECOV_ALPHA_ALPHA'],
             color=flupredict2color[pango2flupredict['ALPHA']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_ALPHA_DELTA'],
             color=flupredict2color[pango2flupredict['DELTA']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_ALPHA_OMICRON'],
             color=flupredict2color[pango2flupredict['BA.1']],
             alpha=alpha_back)
df_c = df_R.loc[(df_R.time > alpha_time) & (df_R.time < ba2_time)]
plt.plot(df_c.time, df_c['C_RECOV_ALPHA_ALPHA'],
         color=flupredict2color[pango2flupredict['ALPHA']],
         linewidth=2.)
plt.plot(df_c.time, df_c['C_RECOV_ALPHA_DELTA'],
         color=flupredict2color[pango2flupredict['DELTA']],
         linewidth=2.)
plt.plot(df_c.time, df_c['C_RECOV_ALPHA_OMICRON'],
         color=flupredict2color[pango2flupredict['BA.1']],
         linewidth=2.)


delta_time = min(voc2era['DELTA'])
ba45_time = min(voc2era['BA.4/5'])
# Delta recovery
for c in countries:
    df_c = df.loc[(df.country == c) & (df.time > delta_time)
                  & (df.time < ba45_time)]
    plt.plot(df_c.time, df_c['C_RECOV_DELTA_ALPHA'],
             color=flupredict2color[pango2flupredict['ALPHA']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_DELTA_DELTA'],
             color=flupredict2color[pango2flupredict['DELTA']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_DELTA_OMICRON'],
             color=flupredict2color[pango2flupredict['BA.1']],
             alpha=alpha_back)
df_c = df_R.loc[(df_R.time > delta_time) & (df_R.time < ba45_time)]
plt.plot(df_c.time, df_c['C_RECOV_DELTA_ALPHA'],
         color=flupredict2color[pango2flupredict['ALPHA']],
         linewidth=2.)
plt.plot(df_c.time, df_c['C_RECOV_DELTA_DELTA'],
         color=flupredict2color[pango2flupredict['DELTA']],
         linewidth=2.)
plt.plot(df_c.time, df_c['C_RECOV_DELTA_OMICRON'],
         color=flupredict2color[pango2flupredict['BA.1']],
         linewidth=2.)

# BA.1 recovery
ba1_time = min(voc2era['BA.1'])
bq1_time = min(voc2era['BQ.1'])
for c in countries:
    df_c = df.loc[(df.country == c) & (df.time > ba1_time)
                  & (df.time < bq1_time)]
    plt.plot(df_c.time, df_c['C_RECOV_BA1_DELTA'],
             color=flupredict2color[pango2flupredict['DELTA']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_BA1_BA.1'],
             color=flupredict2color[pango2flupredict['BA.1']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_BA1_BA.2'],
             color=flupredict2color[pango2flupredict['BA.2']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_BA1_BA.4/5'],
             color=flupredict2color[pango2flupredict['BA.5']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_BA1_BQ.1'],
             color=flupredict2color[pango2flupredict['BQ.1']],
             alpha=alpha_back)
df_c = df_R.loc[(df_R.time > ba1_time) & (df_R.time < bq1_time)]
plt.plot(df_c.time, df_c['C_RECOV_BA1_DELTA'],
         color=flupredict2color[pango2flupredict['DELTA']],
         linewidth=2.0)
plt.plot(df_c.time, df_c['C_RECOV_BA1_BA.1'],
         color=flupredict2color[pango2flupredict['BA.1']],
         linewidth=2.0)
plt.plot(df_c.time, df_c['C_RECOV_BA1_BA.2'],
         color=flupredict2color[pango2flupredict['BA.2']],
         linewidth=2.0)
plt.plot(df_c.time, df_c['C_RECOV_BA1_BA.4/5'],
         color=flupredict2color[pango2flupredict['BA.5']],
         linewidth=2.0)
plt.plot(df_c.time, df_c['C_RECOV_BA1_BQ.1'],
         color=flupredict2color[pango2flupredict['BQ.1']],
         linewidth=2.0)

# BA.2 recovery
ba2_time = min(voc2era['BA.1']) + 50
bq1_time = min(voc2era['BQ.1']) + 50
for c in countries:
    df_c = df.loc[(df.country == c) & (df.time > ba2_time)
                  & (df.time < bq1_time)]
    plt.plot(df_c.time, df_c['C_RECOV_BA2_BA.1'],
             color=flupredict2color[pango2flupredict['BA.1']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_BA2_BA.2'],
             color=flupredict2color[pango2flupredict['BA.2']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_BA2_BA.4/5'],
             color=flupredict2color[pango2flupredict['BA.5']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_BA2_BQ.1'],
             color=flupredict2color[pango2flupredict['BQ.1']],
             alpha=alpha_back)
df_c = df_R.loc[(df_R.time > ba2_time) & (df_R.time < bq1_time)]
plt.plot(df_c.time, df_c['C_RECOV_BA2_BA.1'],
         color=flupredict2color[pango2flupredict['BA.1']], linewidth=2.0)
plt.plot(df_c.time, df_c['C_RECOV_BA2_BA.2'],
         color=flupredict2color[pango2flupredict['BA.2']], linewidth=2.0)
plt.plot(df_c.time, df_c['C_RECOV_BA2_BA.4/5'],
         color=flupredict2color[pango2flupredict['BA.5']], linewidth=2.0)
plt.plot(df_c.time, df_c['C_RECOV_BA2_BQ.1'],
         color=flupredict2color[pango2flupredict['BQ.1']], linewidth=2.0)

# BA.4/5 recovery
ba45_time = min(voc2era['BA.4/5'])
for c in countries:
    df_c = df.loc[(df.country == c) & (df.time > ba45_time)]
    plt.plot(df_c['time'], df_c['C_RECOV_BA45_BA.2'],
             color=flupredict2color[pango2flupredict['BA.2']],
             alpha=alpha_back)
    plt.plot(df_c['time'], df_c['C_RECOV_BA45_BA.4/5'],
             color=flupredict2color[pango2flupredict['BA.5']],
             alpha=alpha_back)
    plt.plot(df_c['time'], df_c['C_RECOV_BA45_BQ.1'],
             color=flupredict2color[pango2flupredict['BQ.1']],
             alpha=alpha_back)
df_c = df_R.loc[(df_R.time > ba45_time)]
plt.plot(df_c.time, df_c['C_RECOV_BA45_BA.2'],
         color=flupredict2color[pango2flupredict['BA.2']], linewidth=2.0)
plt.plot(df_c.time, df_c['C_RECOV_BA45_BA.4/5'],
         color=flupredict2color[pango2flupredict['BA.5']], linewidth=2.0)
plt.plot(df_c.time, df_c['C_RECOV_BA45_BQ.1'],
         color=flupredict2color[pango2flupredict['BQ.1']], linewidth=2.0)

bq1_time = min(voc2era['BQ.1'])
# BQ.1 recovery
for c in countries:
    df_c = df.loc[(df.country == c) & (df.time > bq1_time)]
    plt.plot(df_c.time, df_c['C_RECOV_BQ1_BA.4/5'],
             color=flupredict2color[pango2flupredict['BA.5']],
             alpha=alpha_back)
    plt.plot(df_c.time, df_c['C_RECOV_BQ1_BQ.1'],
             color=flupredict2color[pango2flupredict['BQ.1']],
             alpha=alpha_back)
df_c = df_R.loc[df_R.time > bq1_time]
plt.plot(df_c.time, df_c['C_RECOV_BQ1_BA.4/5'],
         color=flupredict2color[pango2flupredict['BA.5']], linewidth=2.0)
plt.plot(df_c.time, df_c['C_RECOV_BQ1_BQ.1'],
         color=flupredict2color[pango2flupredict['BQ.1']], linewidth=2.0)


xtick_labels = ['2021-01-01', '2021-05-01', '2021-09-01', '2022-01-01',
                '2022-05-01', '2022-09-01', '2023-01-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Jan. $\'$21', 'May $\'$ 21', 'Sep. $\'$21', 'Jan. $\'$22',
                'May $\'$22', 'Sep. $\'$22', 'Jan. $\'$23']
plt.xticks(xtick_pos, xtick_labels, rotation=0, ha='center', fontsize=fs)
plt.xlim([Time.dateToCoordinate("2021-01-01"),
          Time.dateToCoordinate("2023-03-01") - 1])
plt.ylim([-0.02, 0.2])
plt.ylabel("Fitness, $f_i(t)$", fontsize=fs)
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
plt.tick_params(direction='in', labelsize=ls)
plt.savefig("Fig2.pdf")
plt.close()
