import pandas as pd  # Data manipulation library
import matplotlib.pyplot as plt  # Visualization library
from sklearn.model_selection import train_test_split  # Function to split dataset
from sklearn.ensemble import RandomForestClassifier  # ML model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Load local dataset
df = pd.read_csv("data/raw_inspections.csv")

# 2. Filter to valid inspection outcomes only
valid_results = ['Pass', 'Fail', 'Pass w/ Conditions']
df_clean = df[df['results'].isin(valid_results)].copy()

# 3. Clean missing values
df_clean['facility_type'] = df_clean['facility_type'].fillna('Unknown')
df_clean['risk'] = df_clean['risk'].fillna('Unknown')
df_clean['violations'] = df_clean['violations'].fillna('No Violations')

# 4. Feature Engineering: Extract violation count
df_clean['violation_count'] = df_clean['violations'].apply(
    lambda text: 0 if text == 'No Violations' else text.count('|') + 1
)

# 5. Define features (X) and target (y)
features = ['risk', 'facility_type', 'violation_count']
X = pd.get_dummies(df_clean[features], drop_first=True)  # Convert text categories to numbers
y = df_clean['results']  # Target column we want to predict

# 6. Split into Training set (80%) and Test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)  # Train model on training data

# 8. Test model predictions on unseen test data
y_pred = model.predict(X_test)

# 9. Print Model Performance Evaluation
print("--- MODEL PERFORMANCE EVALUATION ---")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)  # <--- THIS LINE DRAWS THE MATRIX

plt.title(
    "Food Inspection Outcome Predictions\n(Model Accuracy vs. Actual City Data)",
    fontsize=12,
    fontweight="bold",
    pad=35
)
plt.xlabel('Model Prediction (What ML Model Guessed)', fontsize=10, fontweight='bold')
plt.ylabel('Actual Outcome (Real Inspection Result)', fontsize=10, fontweight='bold')
fig.subplots_adjust(top=0.82, bottom=0.25)
fig.text(
    0.5,
    0.04,
    "How to read this chart:\n"
    "• Diagonal squares (top-left to bottom-right) = CORRECT predictions\n"
    "• Off-diagonal squares = MISCLASSIFICATIONS\n"
    "• Color intensity shows total volume of inspection test cases",
    ha="center",
    fontsize=8,
    bbox={"facecolor": "orange", "alpha": 0.1, "pad": 5}
)
plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
print("\nSaved 'confusion_matrix.png'!")
plt.show()
# ==========================================
# VISUALIZATION 2: FEATURE IMPORTANCE
# ==========================================
plt.figure(figsize=(8, 5))
importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(5)
importances.plot(kind='barh', color='skyblue')
plt.title('Top 5 Predictive Features in Machine Learning Model', fontsize=12, fontweight='bold')
plt.xlabel('Importance Score (Predictive Weight)', fontsize=10, fontweight='bold')
plt.gca().invert_yaxis()  # Highest importance on top
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300) # Saves chart as an image
print("Saved 'feature_importance.png'!")
plt.show()