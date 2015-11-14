import flask
from flask import render_template
import os

app = flask.Flask(__name__)

here = os.getcwd()

@app.route("/")
def index():
	return render_template('index.html');

@app.route("/img/<path>")
def images(path):
	print 'images(path), path=', path
	fullpath = here + "/muct-master/jpg/" + path
	print fullpath
	print open(fullpath).read()
	resp = flask.make_response(open(fullpath).read())
	resp.content_type = "image/jpeg"
	return resp


if __name__ == "__main__":
		app.run(debug=True)
