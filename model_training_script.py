import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ---------------------------
# Step 1: Load the dataset
# ---------------------------
df = pd.read_csv("cleaned_weather_forecast_dataset.csv")

# Select features and target
features = [
    'Rain', 'Temp Max', 'Temp Min', 'Rain_lag1', 'TempMax_lag1',
    'TempMin_lag1', 'Rain_avg3', 'TempMax_avg3', 'TempMin_avg3',
    'Day', 'Month', 'DayOfYear'
]
X = df[features]
y = df['Target_Rain']

# ---------------------------
# Step 2: Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Step 3: Hyperparameter Tuning with GridSearchCV
# ---------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    rf, param_grid, cv=3, n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("✅ Best Parameters Found:", grid_search.best_params_)

# ---------------------------
# Step 4: Predict and Evaluate
# ---------------------------
y_pred_best = best_model.predict(X_test)

print("\n✅ Improved Classification Report:")
print(classification_report(y_test, y_pred_best))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Rain', 'Rain'],
            yticklabels=['No Rain', 'Rain'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Improved Rain Prediction Confusion Matrix")
plt.show()

# ---------------------------
# Step 5: Feature Importance
# ---------------------------
importances = best_model.feature_importances_
feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

feature_df.plot.bar(x='Feature', y='Importance', legend=False)
plt.title("Feature Importance in Rain Prediction")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# ---------------------------
# Step 6: Save the trained model
# ---------------------------
joblib.dump(best_model, "rain_prediction_model.pkl")
print("✅ Model saved as 'rain_prediction_model.pkl'")