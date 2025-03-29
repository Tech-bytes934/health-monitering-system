import logging
from flask import session, flash, redirect, url_for, request, jsonify, Flask, render_template
from flask_cors import CORS
from werkzeug.security import check_password_hash
from database import get_user_by_email, insert_user
import torch
from datetime import datetime, timedelta
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mysql.connector
import pandas as pd
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier




# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend to access API
app.secret_key = "your_secret_key"  # Required for session management

# ------------------- HOME PAGE -------------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------------- LOGIN PAGE -------------------
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = get_user_by_email(email)

        if user and check_password_hash(user["password"], password):  # Check hashed password
            session["user_id"] = user["id"]
            session["username"] = user["name"]
            flash("Login successful!", "success")
            # print("login")
            return render_template('index.html')
        else:
            flash("Invalid email or password!", "danger")

    return render_template("login.html")

# ------------------- SIGNUP PAGE -------------------
@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        if insert_user(name, email, password):
            flash("Signup successful! Please log in.", "success")
            return redirect(url_for("login"))
        else:
            flash("Email already registered!", "danger")

    return render_template("signup.html")

# ------------------- DASHBOARD PAGE -------------------
@app.route('/dashboard')
def dashboard():
    if "user_id" in session:
        return f"Welcome, {session['username']}! <a href='/logout'>Logout</a>"
    flash("Please log in first!", "warning")
    return redirect(url_for("login"))

# ------------------- PERIOD TRACKER PAGE -------------------


# ---------- 1️⃣ Period Prediction Model ----------
class PeriodPredictor(nn.Module):
    def __init__(self):
        super(PeriodPredictor, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ---------- Routes ----------
# ✅ Fetch Data Helper Function
def fetch_data(query, params=()):
    with get_db_connection() as conn:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

# ✅ Features Page
@app.route('/features')
def features():
    if "user_id" in session:
        return render_template('features.html')  # Ensure the correct file name is used
    flash("Please log in first!", "warning")
    return redirect(url_for("login"))

# ✅ Route to render menstrual.html
@app.route('/menstrual')
def menstrual():
    if "user_id" in session:
        return render_template("menstrual.html")
    flash("Please log in first!", "warning")
    return redirect(url_for("login"))

# ✅ API to predict the next period
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        last_period_date = datetime.strptime(data["lastPeriod"], "%Y-%m-%d")
        cycle_length = int(data["cycleLength"])
        stress_level = int(data.get("stressLevel", 0))  # Default to 0 if not provided

        # Adjust cycle length based on stress level
        if stress_level == 1:  # High Stress
            cycle_length += 2
        elif stress_level == 2:  # Poor Sleep
            cycle_length += 1
        elif stress_level == 3:  # Unhealthy Diet
            cycle_length += 1

        # Predict next period start date
        next_period_date = last_period_date + timedelta(days=cycle_length)

        return jsonify({
            "predicted_date": next_period_date.strftime("%Y-%m-%d"),
            "message": "Prediction based on your cycle and lifestyle."
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Define analyze_fertility function separately
def analyze_fertility(user_id):
    data = fetch_data(
        "SELECT start_date, cycle_length, ovulation_day FROM period_logs WHERE user_id = %s ORDER BY start_date",
        (user_id,)
    )

    if len(data) < 3:
        return {"error": "Not enough data"}

    dates = [datetime.strptime(row["start_date"], "%Y-%m-%d") for row in data]
    cycle_lengths = [row["cycle_length"] for row in data]
    ovulation_days = [row["ovulation_day"] for row in data]

    last_period_date = dates[-1]
    avg_cycle_length = int(np.mean(cycle_lengths))
    avg_ovulation_day = int(np.mean(ovulation_days))

    next_ovulation_date = last_period_date + timedelta(days=avg_ovulation_day)
    fertility_window_start = next_ovulation_date - timedelta(days=4)
    fertility_window_end = next_ovulation_date + timedelta(days=1)

    return {
        "predicted_ovulation_day": next_ovulation_date.strftime("%Y-%m-%d"),
        "fertility_window": [
            fertility_window_start.strftime("%Y-%m-%d"),
            fertility_window_end.strftime("%Y-%m-%d")
        ]
    }

@app.route('/fertility_insights')
def fertility_insights():
    if "user_id" in session:
        user_id = session["user_id"]
        insights = analyze_fertility(user_id)
        irregularity = detect_cycle_irregularity(user_id)
        recommendations = health_recommendations(user_id)

        # Ensure recommendations is a dictionary before updating insights
        if not isinstance(recommendations, dict):
            try:
                recommendations = dict(recommendations)
            except (ValueError, TypeError):
                recommendations = {}  # Fallback to an empty dictionary

        insights.update(irregularity)  # Add cycle irregularity insights
        insights.update(recommendations)  # Add health recommendations

        return jsonify(insights)

    flash("Please log in first!", "warning")
    return redirect(url_for("login"))  # Fix: Redirect should point to "login" or another valid route

'''
@app.route('/health_recommendations')
def health_recommendations():
    return render_template("health_recommendations.html")

@app.route('/menstrual', methods=["POST"])
def menstrual():
    try:
        data = request.form

        # Validate required fields
        if not all(key in data for key in ["lastPeriod", "cycleLength", "stressLevel"]):
            flash("Missing required fields", "danger")
            return redirect(url_for("features"))

        # Convert input values
        last_period_date = datetime.strptime(data["lastPeriod"], "%Y-%m-%d")
        cycle_length = int(data["cycleLength"])
        stress_level = int(data["stressLevel"])

        # Validate stress level
        stress_adjustment = [0, 1, 2, 3]  # Adjustments for stress levels 0 to 3
        if stress_level not in range(len(stress_adjustment)):
            flash("Invalid stress level", "danger")
            return redirect(url_for("features.html"))

        # Calculate adjusted cycle
        adjusted_cycle_length = cycle_length + stress_adjustment[stress_level]
        next_period_date = last_period_date + timedelta(days=adjusted_cycle_length)
        ovulation_date = last_period_date + timedelta(days=14)

        return render_template("health_recommendations.html", 
                               nextPeriodDate=next_period_date.strftime("%Y-%m-%d"), 
                               ovulationDate=ovulation_date.strftime("%Y-%m-%d"))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for("features.html")) 

# ✅ AI Fertility & Ovulation Insights
@app.route('/fertility_insights')
def fertility_insights():
    return render_template("fertility_insights.html")

    if "user_id" in session:
        user_id = session["user_id"]
        insights = analyze_fertility(user_id)
        irregularity = detect_cycle_irregularity(user_id)
        recommendations = health_recommendations(user_id)
        insights.update(irregularity)
        insights.update(recommendations)
        return jsonify(insights)
    flash("Please log in first!", "warning")
    return redirect(url_for("lfertility_insights.html")) 
    

# ---------- 2️⃣ AI Fertility & Ovulation Insights ----------

@app.route('/fertility_insights')
def analyze_fertility(user_id):

    data = fetch_data("SELECT start_date, cycle_length, ovulation_day FROM period_logs WHERE user_id = %s ORDER BY start_date", (user_id,))
    if len(data) < 3:
        return {"error": "Not enough data"}
    
    dates = [datetime.strptime(row[0], "%Y-%m-%d") for row in data]
    cycle_lengths = [row[1] for row in data]
    ovulation_days = [row[2] for row in data]
    
    last_period_date = dates[-1]
    avg_cycle_length = int(np.mean(cycle_lengths))
    avg_ovulation_day = int(np.mean(ovulation_days))
    
    next_ovulation_date = last_period_date + timedelta(days=avg_ovulation_day)
    fertility_window_start = next_ovulation_date - timedelta(days=4)
    fertility_window_end = next_ovulation_date + timedelta(days=1)
    
    return {
        "predicted_ovulation_day": next_ovulation_date.strftime("%Y-%m-%d"),
        "fertility_window": [
            fertility_window_start.strftime("%Y-%m-%d"),
            fertility_window_end.strftime("%Y-%m-%d")
        ]
    }
'''
def detect_cycle_irregularity(user_id):
    data = fetch_data("SELECT cycle_length FROM period_logs WHERE user_id = %s ORDER BY start_date", (user_id,))
    if len(data) < 5:
        return {"cycle_irregularity": "Insufficient data"}
    
    cycle_lengths = [row[0] for row in data]
    
    if not cycle_lengths:
        return {"cycle_irregularity": "No cycle data available"}
    
    std_dev = np.std(cycle_lengths)
    mean_cycle = np.mean(cycle_lengths)
    
    if std_dev > 5:
        return {"cycle_irregularity": "Possible hormonal imbalance (PCOS, stress, or other factors)"}
    if mean_cycle < 21 or mean_cycle > 35:
        return {"cycle_irregularity": "Possible cycle disorder (PCOS, endometriosis, hormonal imbalance)"}
    
    return {"cycle_irregularity": "Cycle appears regular"}

# ---------- 4️⃣ Personalized Health Recommendations ----------
@app.route("/health_recommendations/<int:user_id>", methods=["GET"])
def health_recommendations(user_id):
    try:
        # Fetch the latest wearable data
        query = """
        SELECT stress_level, heart_rate, sleep_hours 
        FROM wearable_logs 
        WHERE user_id = %s 
        ORDER BY log_date DESC 
        LIMIT 1
        """
        wearable_data = fetch_data(query, (user_id,))

        if not wearable_data:
            return {"health_recommendations": ["No wearable data available"]}

        stress_level, heart_rate, sleep_hours = wearable_data[0]
        recommendations = []

        # Define health thresholds
        STRESS_THRESHOLD = 7
        HEART_RATE_THRESHOLD = 90
        SLEEP_HOURS_THRESHOLD = 6

        # Provide recommendations based on health data
        if stress_level > STRESS_THRESHOLD:
            recommendations.append("Reduce stress through meditation, deep breathing, or yoga.")
        if heart_rate > HEART_RATE_THRESHOLD:
            recommendations.append("Monitor heart health and practice relaxation techniques.")
        if sleep_hours < SLEEP_HOURS_THRESHOLD:
            recommendations.append("Improve sleep quality by maintaining a regular sleep schedule and avoiding screens before bed.")

        # If no recommendations, encourage good habits
        if not recommendations:
            recommendations.append("Your health parameters look good! Keep up the healthy habits.")

        logging.info(f"User {user_id} recommendations: {recommendations}")
        return {"health_recommendations": recommendations}

    except Exception as e:
        logging.error(f"Error fetching health recommendations for user {user_id}: {str(e)}")
        return {"error": "An unexpected error occurred. Please try again later."}, 500


# ------------------- SOS PAGE -------------------#
@app.route('/sos')
def sos():
    if "user_id" in session:
        return render_template('sos.html')
    flash("Please log in first!", "warning")
    return redirect(url_for("login"))

# ------------------- LOGOUT -------------------
@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully!", "info")
    return redirect(url_for("login"))


import mysql.connector
from werkzeug.security import generate_password_hash





# Database connection settings
DB_HOST = "localhost"
DB_USER = "root"  # Change to your MySQL username
DB_PASSWORD = ""  # Change to your MySQL password
DB_NAME = "shecare"

# Connect to MySQL database
def get_db_connection():
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    return conn

# Get user by email
def get_user_by_email(email):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()

    cursor.close()
    conn.close()
    return user

# Insert new user into the database
def insert_user(name, email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        hashed_password = generate_password_hash(password)  # Secure password
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, hashed_password))
        conn.commit()
        return True  # Signup successful
    except mysql.connector.IntegrityError:
        return False  # Email already exists
    finally:
        cursor.close()
        conn.close()


# ✅ Run Flask App
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5500, debug=True)

