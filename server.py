import flask
from flask import render_template
import os
import facedb

app = flask.Flask(__name__)

here = os.getcwd()

db = None

@app.route("/")
def index():
	global db 
	return render_template('index.html', image_name=db.rand_img_name())

@app.route("/img/<path>")
def images(path):
	fullpath = here + "/10kfaces/" + path
	resp = flask.make_response(open(fullpath).read())
	resp.content_type = "image/jpeg"
	return resp

@app.route("/rating", methods=['POST'])
def rating():
	global db
	r = flask.request
	if r.method=='POST':
		#print r.form['img_name'], r.form['rating']
		img_name = r.form['img_name']
		rating = 0
		if r.form['rating'] == 'yes':
			rating = 1

		username = r.form['username']
		if username:
			db.add_rating(username, img_name, rating)
	return "okay"


if __name__ == "__main__":
		db = facedb.FaceDB('10kfaces/', 'test.db')
		app.run(debug=True)
