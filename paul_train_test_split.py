import numpy as np
import csv

train = []
test = []

with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

split = int(len(training_set) * 0.8)

for i,a in enumerate(training_set):
	line = training_set[i][0]
	line = line.split(" ")
	line = [int(line[0]), int(line[1]), int(line[2])]
	if i < split:
		train.append(line)
	else:
		test.append(line)

train = np.array(train)
test = np.array(test)

with open("my_train.csv","w") as my_train:
    csv_out = csv.writer(my_train)
    csv_out.writerow(["ID1", "ID2", "category"])
    for row in train:
        csv_out.writerow(row)

with open("my_test.csv","w") as my_train:
    csv_out = csv.writer(my_train)
    csv_out.writerow(["ID1", "ID2", "category"])
    for row in test:
        csv_out.writerow(row)