import csv
import sqlite3 as sql
import sys
import os
import random

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


#Returns a muct object containing a database from the MUCT csv file
class Muct:
	def __init__(self, csv_path, db_filename):
		if os.path.exists(db_filename):
			self.con = sql.connect(db_filename, check_same_thread=False)

			cur = self.con.cursor()
			cur.execute("SELECT COUNT(*) From Landmarks")
			rows = cur.fetchall()
			self.n = rows[0][0] + 1
			print self.n
		else:
			self.create_db(db_filename, csv_path)

	def create_db(self, db_filename, csv_path):
		self.con = sql.connect(db_filename, check_same_thread=False)
		cur = self.con.cursor()
		cur.execute(landmarks_create_table_query())
		cur.execute("CREATE TABLE Ratings(username TEXT, img_name TEXT, rating INT)")

		with open(csv_path, "rb") as muctFile:
			reader = csv.DictReader(muctFile)
			id_num = 0
			for line in reader:
					#We only want shots from camera A
					if line['name'][5] == 'a':
						cur.execute(landmarks_insert_line_query(line, id_num))
						id_num += 1
			self.n = id_num
			self.con.commit()

	def rand_img_name(self):
		i = random.randint(0, self.n-1)
		cur = self.con.cursor()
		cur.execute("SELECT name FROM Landmarks WHERE Id={}".format(i))
		rows = cur.fetchall()
		return rows[0][0]

	def add_rating(self, username, img_name, rating):
		cur = self.con.cursor()
		cur.execute('INSERT INTO Ratings VALUES("'+username + '","'+img_name+'", {})'.format(rating))
		self.con.commit()


