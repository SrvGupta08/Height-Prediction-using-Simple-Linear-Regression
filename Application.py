import pickle
from flask import Flask, request, render_template

application = Flask(__name__)
app = application

scaler = pickle.load(open("standard_scaler.pkl", "rb"))
model = pickle.load(open("simple_linear_regression.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictheight", methods = ["GET", "POST"])
def predict_height():
    if request.method == "POST":
        weight = float(request.form.get("weight"))
        new_weight = scaler.transform([[weight]])
        height = model.predict(new_weight)

        return render_template("prediction.html", result = height)
    else:
        return render_template("prediction.html")

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 3000, debug = True)