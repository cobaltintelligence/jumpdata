## explore.py
## Yuan Wang

from pandas import Series, DataFrame
import pandas
import matplotlib
from sklearn.preprocessing import MinMaxScaler
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from normalize import normalize, scale_series, norm_series
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import sys
# load in data
from scipy.constants import g as __g__
df = pandas.read_csv('data/jump180905.csv', header=0, index_col=0)
df.columns = ['load', 'center_of_pressure']
from scipy.interpolate import spline
# test/train split
__SPLIT__ = float(sys.argv[1])
print("Splitting data at %f" % __SPLIT__)

test = df[__SPLIT__:]
df = df[:__SPLIT__]

# swap
# df, test = test, df

##########################################
# INDEX FIXING
##########################################
df['index'] = df.index - min(df.index)
df.set_index('index', drop=False)
# print(df)

# taking rolling average load
print("Taking rolling window calculations")

df['avload'] = df['load'].rolling(window=1800, win_type="triang").mean()
df['absload'] = df['load'].rolling(window=1800).sum()
# df['var'] = df['load'].rolling(window=1500).var()
# print(np.sum(df['jumping']), len(df['jumping']))

##########################################
# JUMP COUNTS
##########################################
df['loadvar'] = df['load'].rolling(window=1200).var()
df['centervar'] = df['center_of_pressure'].rolling(window=100).var()
df['centervarlong'] = df['center_of_pressure'].rolling(window=4000).var()
# drop NaNs
df = df.dropna()

# Normalize



# df = normalize(df)


# model = ARIMA(df, order=(1,0,10))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())
# # calculate rolling average load variance

# df['loadvarnorm'] = scale_series(df['load'].rolling(window=900).var()
# Jump labeling
__JUMP_THRESHOLD__ = 60
df['jumping'] = [ el > __JUMP_THRESHOLD__ for el in df['loadvar'] ]

##########################################
# JUMP COUNTS
##########################################
print("Indexing jumps")

df['jumpindex'] = ( (df['jumping'] != df['jumping'].shift()) & (df['jumping'] != None ) & (df['jumping'].shift() != None ) ).cumsum()
max_jump_index = df['jumpindex'].tail().iloc[0] 
jump_count = int((max_jump_index - 1) / 2)
jumps = DataFrame(index=range(jump_count))

##########################################
# JUMP STAGE AND DURATIONS
##########################################
print("Calculating jump durations")

__ROUND_RANGE__ = 5
load_mode = float(( round(df['load'] / __ROUND_RANGE__) * __ROUND_RANGE__ ).mode())

for index, row in df.iterrows():
	print("index and row", index, row)
	break;


# TODO: check in jump
df['air'] = [ ( ( el['jumping'] == True ) &  (el['load'] < 250 / 2) ) for index, el in df.iterrows()  ]
print(sum(df['air']))

durations = [ 0 ] * jump_count
airtimes = [ 0 ] * jump_count
maxforces = [ 0 ] * jump_count
ntiles = [ 0 ] * jump_count

__N__ = 30

for i in range(jump_count):
	jump_index = 2 + i * 2
	# check when load is low
	duration_segment = df[((df.jumpindex == jump_index) & (df.load < load_mode) )]
	durations[i] = max(duration_segment.index) - min(duration_segment.index)

	# When in air
	air_segment = df[((df.jumpindex == jump_index) & (df.air == True) )]
	if len(air_segment > 0):
		airtimes[i] = max(air_segment.index) - min(air_segment.index)

	maxforces[i] = max(duration_segment.load)
	ntiles[i] = duration_segment.quantile(q = 1.0 / __N__ ).mean()


jumps['duration'] = durations
jumps['airtime'] = airtimes
jumps['height'] = pow(jumps['airtime'], 2)  * __g__ / 2 / 2
jumps['maxforce'] = maxforces
jumps['ntile'] = ntiles

print(jumps)
print(jumps['ntile'])

# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show()
# residuals.plot(kind='kde')
# print(residuals.describe())
# pyplot.show()

##########################################
# PLOT SETUP
##########################################

fig, axes = plt.subplots(3, 3, sharex='none')

# fig, ax = plt.subplots()
axes[0, 0].set_title('Load Variance')
axes[0, 0].plot(df.index, df['loadvar'], color='black')

trans = mtransforms.blended_transform_factory(axes[0, 0].transData, axes[0, 0].transAxes)
theta = 0.2
# ax.axhline(theta, color='green', lw=2, alpha=0.5)
# ax.axhline(-theta, color='red', lw=2, alpha=0.5)
axes[0, 0].fill_between(df.index, -1, 1, where=df['loadvar'] > __JUMP_THRESHOLD__, facecolor='green', alpha=0.5, transform=trans)
axes[0, 0].fill_between(df.index, -1, 1, where=df['loadvar'] < __JUMP_THRESHOLD__, facecolor='red', alpha=0.5, transform=trans)
axes[0, 0].fill_between(df.index, -1, 1, where=df['air'] == True, facecolor='blue', alpha=0.5, transform=trans)


axes[1, 0].plot(df.index, np.log(df['load']), color='blue')
axes[1, 0].set_title('Load (logarithmic)')
# trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
theta = 0.2
# ax1.axhline(theta, color='green', lw=2, alpha=0.5)
# ax1.axhline(-theta, color='red', lw=2, alpha=0.5)
axes[1, 0].fill_between(df.index, 0, 1, where=df['air'] == True, facecolor='green', alpha=0.5, transform=trans)
axes[1, 0].fill_between(df.index, 0, 1, where=df['air'] == True, facecolor='red', alpha=0.5, transform=trans)



axes[1, 1].plot(df.index, np.log(df['centervar']), color='red', alpha = 0.5)
axes[1, 1].plot(df.index, np.log(df['centervarlong']), color='orange')

axes[1, 1].set_title('Stability (logarithmic)')

# trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
theta = 0.2
# ax1.axhline(theta, color='green', lw=2, alpha=0.5)
# ax1.axhline(-theta, color='red', lw=2, alpha=0.5)
axes[1, 1].fill_between(df.index, 0, 1, where=df['air'] == True, facecolor='green', alpha=0.5, transform=trans)
axes[1, 1].fill_between(df.index, 0, 1, where=df['air'] == True, facecolor='red', alpha=0.5, transform=trans)



axes[2, 1].set_title('Fatigue')

__SPLINE_SMOOTH__ = False
## Smoothing
if (__SPLINE_SMOOTH__):
	xnew = np.linspace(
		norm_series(jumps.index).min(),
		norm_series(jumps.index).max(),
		300
	)
	# print(xnew)
	power_smooth = spline(jumps.index, jumps['ntile'], xnew)

# axes[2, 1].plot(xnew, power_smooth, color='blue', alpha = 1)
# axes[2, 1].plot(xnew, power_smooth, color='violet', alpha = 1)

axes[2, 1].plot(jumps.index, jumps['ntile'], color='blue', alpha = 1)
axes[2, 1].plot(jumps.index, jumps['maxforce'], color='violet', alpha = 1)

axes[0, 1].plot(df.index, df['center_of_pressure'], color='orange')
axes[0, 1].set_title('Center of Pressure (cm)')
# trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
theta = 0.2
# ax2.axhline(theta, color='green', lw=2, alpha=0.5)
# ax2.axhline(-theta, color='red', lw=2, alpha=0.5)
axes[0, 1].fill_between(df.index, 0, 1, where=df['loadvar'] > theta, facecolor='green', alpha=0.5, transform=trans)
axes[0, 1].fill_between(df.index, 0, 1, where=df['loadvar'] < theta, facecolor='red', alpha=0.5, transform=trans)


axes[0, 2].set_title('Airtime and jump duration (s)')
axes[0, 2].bar(jumps.index, jumps['airtime'], alpha=1, color='b')
axes[0, 2].bar(jumps.index, jumps['duration'], alpha=0.5, color='g')

axes[1, 2].set_title('Jump height (approximate)')
axes[1, 2].bar(jumps.index, jumps['height'], alpha=1, color='orange')

axes[2, 2].set_title('Peak power (quantile)')
axes[2, 2].bar(jumps.index, jumps['ntile'], alpha=1, color='blue')


# axes[1, 2].xlabel("Jumps")
# axes[1, 2].xticks(jumps.index, [ "Jump %d" % idx for idx in jumps.index ])
# ax4.fill_between(df.index, -1, 1, where=df['loadvar'] < __JUMP_THRESHOLD__, facecolor='red', alpha=0.5, transform=trans)


##########################################
# REPORTING
##########################################

print("Jump Count: %d" % jump_count )
print("Resting GRF: %d" % load_mode)
print("Jumps")
print(jumps)

plt.show()
######
# df.plot()
# plt.show()