import flask
from flask import render_template
import os
import muct

app = flask.Flask(__name__)

here = os.getcwd()

m = None

@app.route("/")
def index():
	global m
	return render_template('index.html', image_name=m.rand_img_name())

@app.route("/img/<path>")
def images(path):
	fullpath = here + "/muct-master/jpg/" + path
	resp = flask.make_response(open(fullpath).read())
	resp.content_type = "image/jpeg"
	return resp

@app.route("/rating", methods=['POST'])
def rating():
	global m
	r = flask.request
	if r.method=='POST':
		#print r.form['img_name'], r.form['rating']
		img_name = r.form['img_name']
		rating = 0
		if r.form['rating'] == 'yes':
			rating = 1
		m.add_rating(img_name, rating)
	return "okay"


if __name__ == "__main__":
		m = muct.Muct('muct-master/muct-landmarks/muct76.csv', 'test.db')
		app.run(debug=True)
