#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib   # <-- for saving/loading models

# Load metrics CSV produced by your edge cluster script
df = pd.read_csv("output/edge_density_scatter_matplotlib/edge_density_metrics.csv")

# Features and labels
X = df[["edge_density_corners", "edge_density_global"]].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
  # probability=True gives confidence scores
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- Save model and label encoder ---
model_path = Path("output/svm_edge_classifier.pkl")
encoder_path = Path("output/label_encoder.pkl")
joblib.dump(clf, model_path)
joblib.dump(label_encoder, encoder_path)
print(f"Saved model to {model_path}")
print(f"Saved label encoder to {encoder_path}")

# --- Reload later ---
loaded_clf = joblib.load(model_path)
loaded_encoder = joblib.load(encoder_path)

# Example inference
sample = np.array([[0.12, 0.08]])  # [edge_density_corners, edge_density_global]
pred = loaded_clf.predict(sample)
label = loaded_encoder.inverse_transform(pred)
print("Predicted class:", label[0])

# Plot decision boundaries
plt.figure(figsize=(9,7))
colors = ["tab:red","tab:blue","tab:green"]
labels = df["label"].unique()

x_min, x_max = X[:,0].min()-0.01, X[:,0].max()+0.01
y_min, y_max = X[:,1].min()-0.01, X[:,1].max()+0.01
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = loaded_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
for i, label in enumerate(labels):
    sub = df[df["label"] == label]
    plt.scatter(sub["edge_density_corners"], sub["edge_density_global"],
                c=colors[i], label=label, edgecolor="k", s=40)

plt.xlabel("edge_density_corners")
plt.ylabel("edge_density_global")
plt.title("Linear SVM Decision Boundary")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.show()
