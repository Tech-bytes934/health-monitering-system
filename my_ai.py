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

def fetch_period_data(user_id):
    data = fetch_data("SELECT start_date, cycle_length FROM period_logs WHERE user_id = %s ORDER BY start_date", (user_id,))
    if len(data) < 3:
        return None
    dates = [datetime.strptime(row[0], "%Y-%m-%d") for row in data]
    cycle_lengths = [row[1] for row in data]
    return dates, cycle_lengths

def train_period_model(user_id):
    data = fetch_period_data(user_id)
    if data is None:
        return None
    dates, cycle_lengths = data
    X_train = np.array(range(len(cycle_lengths)), dtype=np.float32).reshape(-1, 1)
    y_train = np.array(cycle_lengths, dtype=np.float32).reshape(-1, 1)
    model = PeriodPredictor()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    X_train_tensor, y_train_tensor = torch.tensor(X_train), torch.tensor(y_train)
    
    for _ in range(1000):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_tensor), y_train_tensor)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f"model_user_{user_id}.pth")
    return model

def predict_next_period(user_id):
    data = fetch_period_data(user_id)
    if data is None:
        return {"error": "Not enough data"}
    dates, cycle_lengths = data
    last_period_date = dates[-1]
    model = PeriodPredictor()
    model.load_state_dict(torch.load(f"model_user_{user_id}.pth"))
    model.eval()
    next_cycle_length = model(torch.tensor([[len(cycle_lengths)]], dtype=torch.float32)).item()
    next_period_date = last_period_date + timedelta(days=int(next_cycle_length))
    return {"predicted_period_start": next_period_date.strftime("%Y-%m-%d")}

if __name__ == '__main__':
    app.run(debug=True)