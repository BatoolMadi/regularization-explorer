import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# ======================
# 1. Load Data
# ======================

df = pd.read_csv("data/telecom_churn.csv")

print("Columns:", df.columns)

# ======================
# 2. Define target & features
# ======================

y = df['churned']   # حسب ما شفنا من الأعمدة
X = df.drop(columns=['churned', 'customer_id'])  # نحذف ID كمان

# تحويل categorical
X = pd.get_dummies(X, drop_first=True)

# ======================
# 3. Train-Test Split
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 4. Scaling
# ======================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# نخزن أسماء الفيتشرز (رح نحتاجهم بعدين)
feature_names = X.columns

print("\n/////////////////////////////////////////////")
print("///Data loaded and preprocessed successfully!///")
print("/////////////////////////////////////////////")
print("\nData ready ✅")

# ======================
# 5. Generate C values
# ======================

print("\n/////////////////////////////////////////////")
print("///C values generated///")
print("/////////////////////////////////////////////")

C_values = np.logspace(-3, 2, 20)

# ======================
# 6. Train models & store coefficients
# ======================

coeffs_l1 = []
coeffs_l2 = []

for C in C_values:

    # L1
    model_l1 = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=C,
        max_iter=5000
    )
    model_l1.fit(X_train_scaled, y_train)
    coeffs_l1.append(model_l1.coef_[0])

    # L2
    model_l2 = LogisticRegression(
        penalty='l2',
        C=C,
        max_iter=5000
    )
    model_l2.fit(X_train_scaled, y_train)
    coeffs_l2.append(model_l2.coef_[0])

print("\nC values:")
print(len(coeffs_l1))        # لازم 20

print("\nFeatures:")
print(len(coeffs_l1[0]))     # عدد الفيتشرز
print("\nModels trained ✅")

# ======================
# 7. Plot
# ======================

print("\n/////////////////////////////////////////////")
print("///Draw Regularization Path///")
print("/////////////////////////////////////////////")

print("\nPlotting regularization paths for L1 and L2...")

coeffs_l1 = np.array(coeffs_l1)
coeffs_l2 = np.array(coeffs_l2)

plt.figure(figsize=(14, 6))

# ----- L1 -----
plt.subplot(1, 2, 1)

for i in range(coeffs_l1.shape[1]):
    plt.plot(C_values, coeffs_l1[:, i])

plt.xscale('log')
plt.title("L1 Regularization (Lasso)")
plt.xlabel("C (log scale)")
plt.ylabel("Coefficient Value")

# ----- L2 -----
plt.subplot(1, 2, 2)

for i in range(coeffs_l2.shape[1]):
    plt.plot(C_values, coeffs_l2[:, i])

plt.xscale('log')
plt.title("L2 Regularization (Ridge)")
plt.xlabel("C (log scale)")
plt.ylabel("Coefficient Value")

plt.tight_layout()
plt.savefig("output/regularization_plot.png")
plt.show()

print("\nImage saved to output/regularization_plot.png")

# ======================
# Interpretation
# ======================

"""
The regularization path plot clearly illustrates how coefficients change as regularization strength varies. As C decreases,
coefficients shrink significantly in both models. In the L1 plot,
several coefficients are driven exactly to zero, showing that L1 performs feature selection by eliminating less important variables.
In contrast, the L2 plot shows that coefficients are reduced gradually but remain non-zero, meaning all features are retained.
Some features remain stable across all values of C, indicating strong predictive power, while others shrink rapidly, suggesting weaker influence.
Based on this behavior, L1 is more suitable when model simplicity and feature selection are desired,
whereas L2 is preferable when preserving all features while controlling overfitting is important.
"""