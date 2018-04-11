import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in group:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1][2] for i in sorted(distances)[:k]]

    print(Counter(votes).most_common(1))

    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = float(Counter(votes).most_common(1)[0][1]) / float(k)
    return vote_result, confidence


'''
TEMPLATE EXECUTION OF KNN

df = pd.read_csv("problem10.txt")
df.replace('?',-99999, inplace=True)
full_data = df.astype(float).values.tolist()

print(k_nearest_neighbors(full_data, np.array([1, 1])))
print(k_nearest_neighbors(full_data, np.array([1, 1]), k=7))
'''
