from pandas import Series, DataFrame
import pandas
import matplotlib
from sklearn.preprocessing import MinMaxScaler
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from normalize import normalize, scale_series

__DEBUG2__ = False

series = pandas.read_csv('data/jump180905.csv', header=0, index_col=0)
print(type(series))
# for row in series:
#   print(row.iloc(2))
#   break

values = series.values
print("Values[0]:", values[0])
print("Values shape:", values.shape)
# print(values[:,0])

for i in range(2):
    continue
    data = values[:,i]
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaler = scaler.fit(data.reshape(-1, 1))
    print('Min: %f, Max: %f' % (scaler.data_min_[0], scaler.data_max_[0]))
    normalized = scaler.transform(values)
    for i in range(5):
        print(normalized[i])

data = values
scaler = MinMaxScaler(feature_range=(-1, 1))

scaler = scaler.fit(data)
print('Min: %f, Max: %f' % (scaler.data_min_[0], scaler.data_max_[0]))
print('Min: %f, Max: %f' % (scaler.data_min_[1], scaler.data_max_[1]))


normalized = scaler.transform(values)
if __DEBUG2__:
    for i in range(5):
        print(normalized[i])

print(series.shape)
print(normalized.shape)

# series.values = normalized

df = DataFrame(
    data = normalized, 
    index = series.index, 
    columns = ["load", "center_of_pressure"])


df['var'] = df['load'].rolling(window=100).var()
df['avload'] = scale_series(
    df['load'].rolling(window=1500, win_type="triang").mean().reshape(-1, 1), 
    feature_range=(0, 4)
)


# 
data = values
scaler = MinMaxScaler(feature_range=(-1, 1))

scaler = scaler.fit(data)
print('Min: %f, Max: %f' % (scaler.data_min_[0], scaler.data_max_[0]))
print('Min: %f, Max: %f' % (scaler.data_min_[1], scaler.data_max_[1]))


normalized = scaler.transform(values)
if __DEBUG2__:
    for i in range(5):
        print(normalized[i])

# for i in range(5):
#   print(inversed[i])

df.plot()
plt.show()