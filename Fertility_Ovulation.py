import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mysql.connector
import pandas as pd
from flask import Flask, jsonify, request, render_template, redirect, url_for, session, flash
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = "your_secret_key"

# ---------- MySQL Connection ----------
def fetch_data(query, params=()):
    conn = mysql.connector.connect(
        host="localhost",
        user="your_username",
        password="your_password",
        database="period_tracker"
    )
    cursor = conn.cursor()
    cursor.execute(query, params)
    data = cursor.fetchall()
    conn.close()
    return data

# ---------- 1️⃣ Period Prediction Model ----------
class PeriodPredictor(nn.Module):
    def __init__(self):
        super(PeriodPredictor, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

@app.route('/period_tracker')
def period_tracker():
    if "user_id" in session:
        return render_template('period_tracker.html')
    flash("Please log in first!", "warning")
    return redirect(url_for("login"))

# ---------- 2️⃣ AI Fertility & Ovulation Insights ----------
def analyze_fertility(user_id):
    data = fetch_data("SELECT start_date, cycle_length, ovulation_day FROM period_logs WHERE user_id = %s ORDER BY start_date", (user_id,))
    if len(data) < 3:
        return {"error": "Not enough data"}
    
    dates = [datetime.strptime(row[0], "%Y-%m-%d") for row in data]
    cycle_lengths = [row[1] for row in data]
    ovulation_days = [row[2] for row in data]
    
    # Predict next ovulation and fertility window
    last_period_date = dates[-1]
    avg_cycle_length = int(np.mean(cycle_lengths))
    avg_ovulation_day = int(np.mean(ovulation_days))
    
    next_ovulation_date = last_period_date + timedelta(days=avg_ovulation_day)
    fertility_window_start = next_ovulation_date - timedelta(days=4)
    fertility_window_end = next_ovulation_date + timedelta(days=1)
    
    return {
        "predicted_ovulation_day": next_ovulation_date.strftime("%Y-%m-%d"),
        "fertility_window": [fertility_window_start.strftime("%Y-%m-%d"), fertility_window_end.strftime("%Y-%m-%d")]
    }

@app.route('/fertility_insights')
def fertility_insights():
    if "user_id" in session:
        user_id = session["user_id"]
        insights = analyze_fertility(user_id)
        irregularity = detect_cycle_irregularity(user_id)
        recommendations = provide_health_recommendations(user_id)
        insights.update(irregularity)
        insights.update(recommendations)
        return jsonify(insights)
    flash("Please log in first!", "warning")
    return redirect(url_for("login"))

# ---------- 3️⃣ Cycle Irregularity Detection ----------
def detect_cycle_irregularity(user_id):
    data = fetch_data("SELECT cycle_length FROM period_logs WHERE user_id = %s ORDER BY start_date", (user_id,))
    if len(data) < 5:
        return {"cycle_irregularity": "Insufficient data"}
    
    cycle_lengths = [row[0] for row in data]
    std_dev = np.std(cycle_lengths)
    mean_cycle = np.mean(cycle_lengths)
    
    if std_dev > 5:
        return {"cycle_irregularity": "Possible hormonal imbalance (PCOS, stress, or other factors)"}
    if mean_cycle < 21 or mean_cycle > 35:
        return {"cycle_irregularity": "Possible cycle disorder (PCOS, endometriosis, hormonal imbalance)"}
    return {"cycle_irregularity": "Cycle appears regular"}

# ---------- 4️⃣ Personalized Lifestyle & Health Recommendations ----------
def provide_health_recommendations(user_id):
    wearable_data = fetch_data("SELECT stress_level, heart_rate, sleep_hours FROM wearable_logs WHERE user_id = %s ORDER BY log_date DESC LIMIT 1", (user_id,))
    if not wearable_data:
        return {"health_recommendations": "No wearable data available"}
    
    stress_level, heart_rate, sleep_hours = wearable_data[0]
    recommendations = []
    
    if stress_level > 7:
        recommendations.append("Reduce stress through meditation or yoga")
    if heart_rate > 90:
        recommendations.append("Monitor heart health and consider relaxation exercises")
    if sleep_hours < 6:
        recommendations.append("Improve sleep quality with better sleep hygiene")
    
    if not recommendations:
        recommendations.append("Your health parameters look good! Keep up the healthy habits.")
    
    return {"health_recommendations": recommendations}

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5500, debug=True)

