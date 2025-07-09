from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import random

app = Flask(__name__)


model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
training_columns = pd.read_csv("training_columns.csv")
columns = training_columns["columns"].tolist()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        holiday = request.form["holiday"]
        temp = float(request.form["temp"])
        rain = float(request.form["rain"])
        snow = float(request.form["snow"])
        weather = request.form["weather"]
        year = int(request.form["year"])
        month = int(request.form["month"])
        day = int(request.form["day"])
        hour = int(request.form["hour"])
        minute = int(request.form["minute"])
        second = int(request.form["second"])
        df = pd.DataFrame([{
            "holiday": holiday,
            "temp": temp,
            "rain": rain,
            "snow": snow,
            "weather": weather,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second
        }])

        # One-hot encode
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        # Scale input
        scaled = scaler.transform(df)

        # Predict traffic volume
        prediction = model.predict(scaled)
        volume = round(prediction[0], 2)

        lat = random.uniform(12.90, 13.10)   
        lon = random.uniform(77.50, 77.70)

        return render_template("result.html", volume=volume, lat=lat, lon=lon)

    except Exception as e:
        return f"❌ Error occurred: {e}"

@app.route("/map")
def map_page():
    return render_template("map.html")

@app.route("/traffic_data")
def traffic_data():
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))

    nearby = []
    for _ in range(10):
        offset_lat = lat + random.uniform(-0.01, 0.01)
        offset_lon = lon + random.uniform(-0.01, 0.01)
        volume = random.randint(500, 5000)
        nearby.append({"lat": offset_lat, "lon": offset_lon, "volume": volume})

    return jsonify(nearby)

if __name__ == "__main__":
    app.run(debug=True)
