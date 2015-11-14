import flask
from flask import render_template
import os
import muct

app = flask.Flask(__name__)

here = os.getcwd()

@app.route("/")
def index():
	return render_template('index.html');

@app.route("/img/<path>")
def images(path):
	fullpath = here + "/muct-master/jpg/" + path
	resp = flask.make_response(open(fullpath).read())
	resp.content_type = "image/jpeg"
	return resp


if __name__ == "__main__":
		muct.read_csv('muct-master/muct-landmarks/muct76.csv', 'test.db')
		app.run(debug=True)
