import csv
import sqlite3 as sql
import sys
import os
import random
from os.path import isfile, join

#Returns a facedb object containing a database from the specified collection of images
class FaceDB:
	def __init__(self, img_dir_path, db_filename):
		if os.path.exists(db_filename):
			self.con = sql.connect(db_filename, check_same_thread=False)

			cur = self.con.cursor()
			cur.execute("SELECT COUNT(*) From Images")
			rows = cur.fetchall()
			self.n = rows[0][0]
		else:
			self.create_db(db_filename, img_dir_path)

	def create_db(self, db_filename, img_dir_path):
		self.con = sql.connect(db_filename, check_same_thread=False)
		cur = self.con.cursor()
		cur.execute("CREATE TABLE Images(Id INT, name TEXT)")
		cur.execute("CREATE TABLE Ratings(username TEXT, img_name TEXT, rating INT)")

		id_num = 0
		for f in os.listdir(img_dir_path):
			if isfile(join(img_dir_path, f)) and f.endswith('.jpg'):
				cur.execute('INSERT INTO Images VALUES({}'.format(id_num) + 
						',"' + f + '")')
				id_num += 1
		self.n = id_num
		print 'n: ',self.n
		self.con.commit()

	def rand_img_name(self):
		i = random.randint(0, self.n-1)
		cur = self.con.cursor()
		cur.execute("SELECT name FROM Images WHERE Id={}".format(i))
		rows = cur.fetchall()
		return rows[0][0]

	#Returns the faces that a user rated as attractive
	def pos_faces_for_user(self, username):
		cur = self.con.cursor()
		cur.execute('SELECT img_name FROM Ratings where username="' + username
				+ '" and rating=1')
		rows = cur.fetchall()
		strings = []
		for s in rows:
			strings.append(s[0])
		return strings;

	#returns a list of (filename, rating) tuples
	def ratings_for_user(self, username):
		cur = self.con.cursor()
		cur.execute('SELECT * FROM Ratings WHERE username="' + username
				+ '"')
		rows = cur.fetchall()
		result = []
		for row in rows:
			result.append((str(row[1]), row[2]))
		return result


	def add_rating(self, username, img_name, rating):
		cur = self.con.cursor()
		if self.rating_exists(username, img_name):
			cur.execute('UPDATE Ratings SET rating={} '.format(rating) + 
					'WHERE username="'+username+'" AND img_name = "' + img_name + '"')
		else:
			cur.execute('INSERT INTO Ratings VALUES("'+username + '","'+img_name+'", {})'.format(rating))
		self.con.commit()
	
	def rating_exists(self, username, img_name):
		cur = self.con.cursor()
		cur.execute('SELECT COUNT(*) FROM Ratings WHERE username="'+username + '" AND img_name = "' + img_name + '"')
		rows = cur.fetchall()
		count = rows[0][0]
		return count > 0

