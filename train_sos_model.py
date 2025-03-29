from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Sample data for training (heart_rate, bp, oxygen, emergency label)
X_train = np.array([
    [90, 120, 98],   # Normal
    [110, 140, 90],  # Risky
    [150, 180, 85],  # Emergency
    [85, 115, 99],   # Normal
    [140, 175, 80]   # Emergency
])
y_train = np.array([[0], [0], [1], [0], [1]])  # 1 = Emergency, 0 = Normal

# Standardizing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "scaler.pkl")  # Save scaler

# Create model
model = Sequential([
    Dense(10, activation='relu', input_shape=(3,)),
    Dense(5, activation='relu'),
    Dense(1, activation='sigmoid')  # Output: 0 (Normal) or 1 (Emergency)
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_scaled, y_train, epochs=50, verbose=1)

# Save model
model.save("sos_model.h5")
print("✅ Model trained and saved as sos_model.h5")
