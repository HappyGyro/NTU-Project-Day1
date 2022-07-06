from json import load

from torch import rand_like
from flask import Flask
app = Flask(__name__)

from flask import request, render_template
import joblib

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        rates = float(request.form.get("rates"))
        print(rates)
        model = joblib.load("regression")
        r_l = model.predict([[rates]])
        model = joblib.load("decision_tree")
        r_t = model.predict([[rates]])
        return(render_template("index.html", result_linear = r_l,result_tree = r_t))
    else:
        return(render_template("index.html", result_linear = "waiting",result_tree = "waiting"))

if __name__=="__main__":
    app.run()