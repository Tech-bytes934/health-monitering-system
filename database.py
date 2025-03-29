import torch
import torch.nn as nn
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

class PeriodPredictor(nn.Module):
    def __init__(self):
        super(PeriodPredictor, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)