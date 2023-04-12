
from flask import Flask, request, render_template
from sklearn import datasets
import pickle
app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
  return render_template('home.html', query="")


@app.route("/predict", methods=['POST'])
def predict():
  inputQuery1 = request.form['query1']
  inputQuery2 = request.form['query2']
  inputQuery3 = request.form['query3']
  inputQuery4 = request.form['query4']
  iris = datasets.load_iris()
  ml_model = pickle.load(open('model.pkl', 'rb'))
  data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4]]
  pred = ml_model.predict(data)
  target_names = iris.target_names[pred]
  return render_template('home.html', output1=target_names, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'])


if __name__ == "__main__":
    app.run(host='0.0.0.0')
