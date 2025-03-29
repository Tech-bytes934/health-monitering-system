from flask import Flask, jsonify
import pytest

# Flask Application
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "Welcome to Flask!"})

@app.route('/add/<int:a>/<int:b>')
def add(a, b):
    return jsonify({"result": a + b})

# Pytest Configuration & Tests
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test index route"""
    response = client.get('/')
    assert response.status_code == 200
    assert response.get_json() == {"message": "Welcome to Flask!"}

def test_add(client):
    """Test add route"""
    response = client.get('/add/2/3')
    assert response.status_code == 200
    assert response.get_json() == {"result": 5}

# Run the Flask app if executed directly
if __name__ == "__main__":
    app.run(debug=True)

