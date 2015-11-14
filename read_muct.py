import csv

with open('muct-master/muct-landmarks/muct76.csv', "rb") as muctFile:
	reader = csv.DictReader(muctFile)
	for line in reader:
		print line
