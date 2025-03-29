import pymysql
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def get_db_connection():
    try:
        return pymysql.connect(host='localhost', user='your_user', password='your_password', database='your_db', cursorclass=pymysql.cursors.DictCursor)
    except pymysql.MySQLError as e:
        print(f"Database connection failed: {e}")
        return None

# AI Model for Digital Twin - Predicting Women's Health Parameters
class DigitalTwinModel(nn.Module):
    def __init__(self):
        super(DigitalTwinModel, self).__init__()
        self.fc1 = nn.Linear(7, 15)
        self.fc2 = nn.Linear(15, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

digital_twin_model = DigitalTwinModel()
scaler = StandardScaler()

@app.route('/digital_twin', methods=['POST'])
def digital_twin():
    data = request.json
    user_id = data['user_id']
    
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed."}), 500
    
    with connection.cursor() as cursor:
        cursor.execute("SELECT stress_level, sleep_hours, heart_rate, bp, menstrual_cycle, hormone_levels, pregnancy_status FROM health_logs WHERE user_id=%s ORDER BY timestamp DESC LIMIT 1", (user_id,))
        health_data = cursor.fetchone()
    connection.close()
    
    if not health_data:
        return jsonify({"error": "No health data found."}), 404
    
    input_data = np.array([[
        health_data['stress_level'],
        health_data['sleep_hours'],
        health_data['heart_rate'],
        health_data['bp'],
        health_data['menstrual_cycle'],
        health_data['hormone_levels'],
        health_data['pregnancy_status']
    ]])
    input_data = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    with torch.no_grad():
        prediction = digital_twin_model(input_tensor).item()
    
    response = {"prediction": prediction}
    
    if prediction > 0.5:
        response["message"] = "High risk of hormonal imbalance, fertility issues, or pregnancy complications."
    else:
        response["message"] = "Health status is stable."
    
    return jsonify(response)

@app.route('/fertility_insights', methods=['POST'])
def fertility_insights():
    data = request.json
    user_id = data['user_id']
    
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed."}), 500
    
    with connection.cursor() as cursor:
        cursor.execute("SELECT menstrual_cycle, hormone_levels FROM health_logs WHERE user_id=%s ORDER BY timestamp DESC LIMIT 1", (user_id,))
        fertility_data = cursor.fetchone()
    connection.close()
    
    if not fertility_data:
        return jsonify({"error": "No fertility data found."}), 404
    
    cycle_length = fertility_data['menstrual_cycle']
    hormone_levels = fertility_data['hormone_levels']
    
    if cycle_length < 25 or cycle_length > 35:
        return jsonify({"message": "Irregular cycle detected. Fertility may be affected."})
    
    if hormone_levels < 0.3:
        return jsonify({"message": "Low hormone levels detected. Consider medical advice."})
    
    return jsonify({"message": "Fertility cycle is within a healthy range."})

@app.route('/disease_prediction', methods=['POST'])
def disease_prediction():
    data = request.json
    user_id = data['user_id']
    
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed."}), 500
    
    with connection.cursor() as cursor:
        cursor.execute("SELECT stress_level, menstrual_cycle, hormone_levels, bone_density FROM health_logs WHERE user_id=%s ORDER BY timestamp DESC LIMIT 1", (user_id,))
        disease_data = cursor.fetchone()
    connection.close()
    
    if not disease_data:
        return jsonify({"error": "No health data found."}), 404
    
    risk_factors = disease_data['stress_level'] + disease_data['menstrual_cycle'] + disease_data['hormone_levels'] + disease_data['bone_density']
    
    if risk_factors > 5:
        return jsonify({"message": "High risk of PCOS, endometriosis, or osteoporosis. Consult a specialist."})
    return jsonify({"message": "No significant disease risk detected."})

@app.route('/nutrition_recommendation', methods=['POST'])
def nutrition_recommendation():
    data = request.json
    user_id = data['user_id']
    
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed."}), 500
    
    with connection.cursor() as cursor:
        cursor.execute("SELECT hormone_levels, stress_level FROM health_logs WHERE user_id=%s ORDER BY timestamp DESC LIMIT 1", (user_id,))
        nutrition_data = cursor.fetchone()
    connection.close()
    
    if not nutrition_data:
        return jsonify({"error": "No nutrition data found."}), 404
    
    if nutrition_data['hormone_levels'] < 0.3:
        return jsonify({"message": "Increase intake of omega-3, leafy greens, and whole grains."})
    
    if nutrition_data['stress_level'] > 5:
        return jsonify({"message": "Reduce caffeine and increase magnesium-rich foods."})
    
    return jsonify({"message": "Maintain a balanced diet."})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5500, debug=True)