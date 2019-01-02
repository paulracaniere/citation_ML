import csv

with open("node_information.csv","r") as f:
    csv_read = csv.reader(f)
    array = list(csv_read)

with open("node_information.csv", "w") as f:
    csv_write = csv.writer(f)
    csv_write.writerow(["ID","year","title","authors","journal","abstract"])
    for a in array:
        csv_write.writerow(a)