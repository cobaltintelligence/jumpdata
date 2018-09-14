from pandas import Series
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
series = Series.from_csv('data/jump180905.csv', header=0)
values = series.values
values = values.reshape((len(values), 1))

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)

print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
normalized = scaler.transform(values)

series.plot(style='k.')
pyplot.show()

for i in range(5):
	print(normalized[i])

inversed = scaler.inverse_transform(normalized)

for i in range(5):
	print(inversed[i])