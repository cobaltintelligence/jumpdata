## normalize.py
## Yuan Wang

from pandas import DataFrame
import pandas
from sklearn.preprocessing import MinMaxScaler, StandardScaler

__DEBUG__ = True

def normalize(df):
    """
    Normalizes all feature columns in a given dataframe. Returns a new dataframe.
    """
    data = df.values

    normalized = scale_series(data)

    # print beginning
    if __DEBUG__:
        for i in range(5):
            print(normalized[i])

    # create new df, retain old col names
    out = DataFrame(
        data = normalized, 
        index = df.index, 
        columns = df.columns.values
    )

    # todo: return scaler as well.
    return out

# scaler = MinMaxScaler(feature_range=(-1, 1))

def scale_series(data, scaler=StandardScaler()):
    """
    Scales a series using a min-max scaler with a specified feature range
    """
    # scaler = MinMaxScaler(feature_range=feature_range)
    scaler = scaler.fit(data)

    # print stats on scaler
    if len(scaler.mean_ > 0):
        for i in range(len(scaler.mean_)):
            print('Mean: %f, Var: %f' % (scaler.mean_[i], scaler.var_[i]))

    normalized = scaler.transform(data)
    return normalized

def norm_series(x, scaler=StandardScaler()):
    """
    normalize data of df column
    """
    data = x.values.reshape(-1, 1) #returns a numpy array
    x_scaled = scaler.fit_transform(data)

    return x_scaled

    