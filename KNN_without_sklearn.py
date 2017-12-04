import numpy as np
import pandas as pd
import warnings
from collections import Counter
import random
from math import sqrt

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

# for i in dataset:
# 	for ii in dataset[i]:
# 		plt.scatter(ii[0], ii[1], s=100, color=i)

# plt.show()

def kNN(data, predict, k=3):
	if len(data)>=k:
		warnings.warn('K is set to a value less than voting groups')
	distances = []
	for group in data:
		for instances in data[group]:
			d = np.linalg.norm(np.array(instances) - np.array(predict))
			distances.append([d, group])

	votes=[i[1] for i in sorted(distances)[:k]]
	vote_result=Counter(votes).most_common(1)[0][0]
	confidence=Counter(votes).most_common(1)[0][1]/k

	return vote_result,confidence

df=pd.read_csv("x.txt")
df.replace('?',-999999, inplace=True)
df.drop(['id'], 1, inplace=True)

print(df.head())
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size=.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]

for i in train_data:
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct,total=0,0

for group in train_set:
	for data in train_set[group]:
		vote,confidence = kNN(train_set, data, k=5)
		if group==vote:
			correct+=1
		total+=1
print('accuracy => ', correct/total)
