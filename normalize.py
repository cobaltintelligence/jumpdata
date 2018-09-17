## normalize.py
## Yuan Wang

from pandas import DataFrame
import pandas
from sklearn.preprocessing import MinMaxScaler
matplotlib.use('TkAgg')

__DEBUG__ = True

def normalize(df):
	"""
	Normalizes all feature columns in a given dataframe
	"""
	values = series.values
	scaler = MinMaxScaler(feature_range=(-1, 1))
	print('Min: %f, Max: %f' % (scaler.data_min_[0], scaler.data_max_[0]))
	print('Min: %f, Max: %f' % (scaler.data_min_[1], scaler.data_max_[1]))

	normalized = scaler.transform(values)
	if __DEBUG__:
	for i in range(5):
		print(normalized[i])


	df = DataFrame(
	data = normalized, 
	index = series.index, 
	columns = df.columns.values )


	