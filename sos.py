
from flask import Flask, request, jsonify
import mysql.connector
import smtplib
from twilio.rest import Client
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import json
import threading
import time
import os                                               

# Disable TensorFlow oneDNN optimizations warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

# Load AI model
model = load_model("sos_model.h5")
scaler = StandardScaler()

# MySQL Database Connection
try:
    db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="your_correct_password",
    database="health_db"
)
    cursor = db.cursor()
except mysql.connector.Error as err:
    print("Error: Could not connect to MySQL:", err)
    exit()

# Twilio API for SMS Alerts
TWILIO_SID = "your_twilio_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_PHONE_NUMBER = "your_twilio_phone_number"
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# SMTP Email Configuration
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_email_password"

def send_sms(phone_number, message):
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=phone_number
    )

def send_email(to_email, subject, message):
    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        msg = f"Subject: {subject}\n\n{message}"
        server.sendmail(EMAIL_ADDRESS, to_email, msg)

# AI-based Health Monitoring
@app.route('/monitor', methods=['POST'])
def monitor_health():
    data = request.json
    vitals = np.array([data['heart_rate'], data['bp'], data['oxygen']]).reshape(1, -1)
    vitals_scaled = scaler.transform(vitals)  # Use transform instead of fit_transform
    prediction = model.predict(vitals_scaled)
    
    if prediction[0][0] > 0.8:  # Threshold for emergency
        send_sos_alert(data['user_id'])
        return jsonify({"status": "Emergency detected, SOS triggered!"})
    return jsonify({"status": "Vitals normal"})

# Trigger SOS Manually
@app.route('/trigger_sos', methods=['POST'])
def trigger_sos():
    user_id = request.json['user_id']
    send_sos_alert(user_id)
    return jsonify({"status": "Manual SOS triggered!"})

# Send SOS Alert
def send_sos_alert(user_id):
    cursor.execute("SELECT name, phone, email FROM users WHERE id=%s", (user_id,))
    user = cursor.fetchone()
    if user:
        message = f"Emergency Alert! {user[0]} needs immediate help. Contact immediately."
        send_sms(user[1], message)
        send_email(user[2], "Emergency Alert", message)

# AI Chatbot for Emergency Assistance
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['message'].lower()
    responses = {
        "help": "Stay calm. What symptoms are you experiencing? (e.g., chest pain, dizziness, difficulty breathing)",
        "chest pain": "Take deep breaths. Have you taken any medication?",
        "dizziness": "Lie down and raise your legs. Are you alone?",
        "difficulty breathing": "Try to remain calm and take slow breaths. Help is on the way.",
        "alone": "Help is on the way. Try to unlock your door for responders."
    }
    response = responses.get(user_input, "I am here to help. Can you describe your symptoms?")
    return jsonify({"response": response})

# Automated Health Monitoring (Runs in Background)
def health_monitoring_service():
    while True:
        cursor.execute("SELECT user_id, heart_rate, bp, oxygen FROM health_vitals")
        vitals = cursor.fetchall()
        for user_id, heart_rate, bp, oxygen in vitals:
            vitals_array = np.array([[heart_rate, bp, oxygen]])
            vitals_scaled = scaler.transform(vitals_array)  # Use transform instead of fit_transform
            prediction = model.predict(vitals_scaled)
            if prediction[0][0] > 0.8:
                send_sos_alert(user_id)
        time.sleep(30)  # Check every 30 seconds

# Run Health Monitoring in Background
monitoring_thread = threading.Thread(target=health_monitoring_service)
monitoring_thread.daemon = True
monitoring_thread.start()

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5500, debug=True)

    