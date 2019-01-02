import random
import numpy as np
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import csv

with open("reduced_processed_train.csv", "r") as f:
    reader = csv.reader(f)
    train = np.array(list(reader))
    training_features = train[1:,:3]
    training_features = np.array([[float(t) for t in a] for a in training_features])
    labels_array = train[1:,3:].T
    labels_array = np.array([[int(t) for t in a] for a in labels_array])[0]
    print(training_features)
    print(labels_array)

with open("processed_test.csv", "r") as f:
    test = np.array(list(csv.reader(f)))
    testing_features = test[1:,:3]
    testing_features = np.array([[float(t) for t in a] for a in testing_features])
    print(testing_features)
    

# initialize basic SVM
classifier = svm.SVC(kernel='rbf', C=1.0)

# train
classifier.fit(training_features, labels_array)

# issue predictions
predictions_SVM = list(classifier.predict(testing_features))

# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
predictions_SVM = zip(range(len(testing_features)), predictions_SVM)

with open("improved_predictions.csv","w") as pred1:
    csv_out = csv.writer(pred1)
    csv_out.writerow(["ID","category"])
    for row in predictions_SVM:
        csv_out.writerow(row)