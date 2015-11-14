import csv
import sqlite3 as sql
import sys
import os

#Generates the SQL query string to create a new Landmarks table
def landmarks_create_table_query():
	query = "CREATE TABLE Landmarks(Id INT, name TEXT, tag TEXT, "
	for i in range(0, 76):
		query += 'x{:02d} REAL, '.format(i)
		query += 'y{:02d} REAL'.format(i)
		if i < 75:
			query += ", "
	query += ")"
	return query

#Generates the SQL query string to add a CSV line to the Landmarks table
def landmarks_insert_line_query(line, id_num):
	query = "INSERT INTO Landmarks VALUES("
	query += "{}".format(id_num) + ", "
	query += '"'+line['name'] + '", '
	query += '"'+line['tag'] + '", '

	for i in range (0, 76):
		query += line['x{:02d}'.format(i)] + ", "
		query += line['y{:02d}'.format(i)]
		if i < 75:
			query += ", "
	query += ")"
	return query



def create_db(db_filename):
	#TODO handle pre-existing file instead of deleting
	if os.path.exists(db_filename):
		os.remove(db_filename)

	con = sql.connect(db_filename)
	cur = con.cursor()
	cur.execute(landmarks_create_table_query())
	return con

#Returns an sqlite3 connection to a database with the data from csv_path
def read_csv(csv_path, db_filename):
	with open(csv_path, "rb") as muctFile:
		con = create_db(db_filename)
		cur = con.cursor()

		reader = csv.DictReader(muctFile)
		id_num = 0
		for line in reader:
				cur.execute(landmarks_insert_line_query(line, id_num))
				id_num += 1

		con.commit()
		return con

