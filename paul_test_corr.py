import numpy as np
import csv

with open("my_train.csv", "r") as f:
    reader = csv.reader(f)
    train = np.array(list(reader)[1:])

with open("node_information.csv", "r") as f:
	reader = csv.reader(f)
	node_information = np.array(list(reader)[1:])

start = 10000
j = 0
for i, a in enumerate(train[start:]):
	if int(a[2]) == 1:
		j = i + start
		break

for i, a in enumerate(node_information):
	if int(a[0]) == int(train[j][0]):
		node1 = a

for i, a in enumerate(node_information):
	if int(a[0]) == int(train[j][1]):
		node2 = a

print(j)
print(train[j])
print(node1[-1])
print(" ")
print(node2[-1])