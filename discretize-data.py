import pandas as pd
import numpy as np
import math


# Discretize synthetic data
colnames=['X', 'Y', 'Label']
df = pd.read_csv('data/synthetic-1.csv', names=colnames, header=None)
# Cut was used to find 4 equally distant bins
# df['X'] = pd.cut(df['X'], 4)
# df['Y'] = pd.cut(df['Y'], 4)
# print(df)
data = df.to_numpy()
for i in range(len(data)):
    if data[i, 0] > -2.984 and data[i, 0] <= 1.023:
        data[i, 0] = 1
    elif data[i, 0] > 1.023 and data[i, 0] <= 5.014:
        data[i, 0] = 2
    elif data[i, 0] > 5.014 and data[i, 0] <= 9.005:
        data[i, 0] = 3
    elif data[i, 0] > 9.005 and data[i, 0] <= 13:
        data[i, 0] = 4
for i in range(len(data)):
    if data[i, 1] > -2.403 and data[i, 1] <= 0.259:
        data[i, 1] = 1
    elif data[i, 1] > 0.259 and data[i, 1] <= 2.91:
        data[i, 1] = 2
    elif data[i, 1] > 2.91 and data[i, 1] <= 5.562:
        data[i, 1] = 3
    elif data[i, 1] > 5.562 and data[i, 1] <= 8.3:
        data[i, 1] = 4
df = pd.DataFrame(data)
df.to_csv('data/discrete-synthetic-1.csv', header=None, index=None)


df = pd.read_csv('data/synthetic-2.csv', names=colnames, header=None)
# Cut was used to find 4 equally distant bins
# df['X'] = pd.cut(df['X'], 4)
# df['Y'] = pd.cut(df['Y'], 4)
# print(df)
data = df.to_numpy()
for i in range(len(data)):
    if data[i, 0] > -2.984 and data[i, 0] <= -1.256:
        data[i, 0] = 1
    elif data[i, 0] > -1.256 and data[i, 0] <= 0.0878:
        data[i, 0] = 2
    elif data[i, 0] > 0.0878 and data[i, 0] <= 1.431:
        data[i, 0] = 3
    elif data[i, 0] > 1.431 and data[i, 0] <= 2.8:
        data[i, 0] = 4
for i in range(len(data)):
    if data[i, 1] > -1.292 and data[i, 1] <= 0.667:
        data[i, 1] = 1
    elif data[i, 1] > 0.667 and data[i, 1] <= 2.618:
        data[i, 1] = 2
    elif data[i, 1] > 2.618 and data[i, 1] <= 4.570:
        data[i, 1] = 3
    elif data[i, 1] > 4.570 and data[i, 1] <= 6.6:
        data[i, 1] = 4
df = pd.DataFrame(data)
df.to_csv('data/discrete-synthetic-2.csv', header=None, index=None)


df = pd.read_csv('data/synthetic-3.csv', names=colnames, header=None)
# Cut was used to find 4 equally distant bins
# df['X'] = pd.cut(df['X'], 4)
# df['Y'] = pd.cut(df['Y'], 4)
# print(df)
data = df.to_numpy()
for i in range(len(data)):
    if data[i, 0] > -1.554 and data[i, 0] <= 0.268:
        data[i, 0] = 1
    elif data[i, 0] > 0.268 and data[i, 0] <= 2.083:
        data[i, 0] = 2
    elif data[i, 0] > 2.083 and data[i, 0] <= 3.898:
        data[i, 0] = 3
    elif data[i, 0] > 3.898 and data[i, 0] <= 5.8:
        data[i, 0] = 4
for i in range(len(data)):
    if data[i, 1] > -3.464 and data[i, 1] <= -1.665:
        data[i, 1] = 1
    elif data[i, 1] > -1.665 and data[i, 1] <= 0.127:
        data[i, 1] = 2
    elif data[i, 1] > 0.127 and data[i, 1] <= 1.918:
        data[i, 1] = 3
    elif data[i, 1] > 1.918 and data[i, 1] <= 3.8:
        data[i, 1] = 4
df = pd.DataFrame(data)
df.to_csv('data/discrete-synthetic-3.csv', header=None, index=None)


df = pd.read_csv('data/synthetic-4.csv', names=colnames, header=None)
# Cut was used to find 4 equally distant bins
# df['X'] = pd.cut(df['X'], 4)
# df['Y'] = pd.cut(df['Y'], 4)
# print(df)
data = df.to_numpy()
for i in range(len(data)):
    if data[i, 0] > -6.323 and data[i, 0] <= -1.124:
        data[i, 0] = 1
    elif data[i, 0] > -1.124 and data[i, 0] <= 4.055:
        data[i, 0] = 2
    elif data[i, 0] > 4.055 and data[i, 0] <= 9.233:
        data[i, 0] = 3
    elif data[i, 0] > 9.233 and data[i, 0] <= 14.5:
        data[i, 0] = 4
for i in range(len(data)):
    if data[i, 1] > -10.639 and data[i, 1] <= -4.989:
        data[i, 1] = 1
    elif data[i, 1] > -4.989 and data[i, 1] <= 0.639:
        data[i, 1] = 2
    elif data[i, 1] > 0.639 and data[i, 1] <= 6.266:
        data[i, 1] = 3
    elif data[i, 1] > 6.266 and data[i, 1] <= 11.9:
        data[i, 1] = 4
df = pd.DataFrame(data)
df.to_csv('data/discrete-synthetic-4.csv', header=None, index=None)


# Discretize pokemon data
bins = 6
df = pd.read_csv('data/pokemonStats.csv')
data = df.to_numpy()
df = pd.read_csv('data/pokemonLegendary.csv')
temp_labels = df.to_numpy()
labels = []
for i in range(len(temp_labels)):
    if temp_labels[i, 0] == True:
        labels.append(1)
    else:
        labels.append(0)
labels = np.vstack(labels)
for i in range(7):
    feat_min = np.amin(data[:, i])
    feat_max = np.amax(data[:, i])
    for j in range(len(data[:, i])):
        example = data[j, i]
        bin_interval = (feat_max - feat_min)/bins
        bin = math.floor(example/bin_interval)
        data[j, i] = bin
data = np.append(data, labels, 1)
df = pd.DataFrame(data)
df.to_csv('data/discrete-pokemonStats.csv', header=None, index=None)
