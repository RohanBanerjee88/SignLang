import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
with open("hand_gesture_data.pkl", "rb") as f:
    data, labels = pickle.load(f)

# Flatten the data
data = data.reshape(data.shape[0], -1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
with open("hand_gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)
