# 9. (coding) Gradient Boosting

# 1) Implement the following Gradient Boosting program: Regression using squared error 

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text

# you can change no of classifiers, if necessary
# show the first 3 steps of boosting only.
T = 3

# ===========================
# 9.1)-1: Create/Load Data
# ===========================
# Create synthetic regression data
np.random.seed(42)
n_samples = 50
X_train = np.linspace(0, 10, n_samples).reshape(-1, 1)
# y = 2*x + noise
Y_train = 2 * X_train.ravel() + np.random.randn(n_samples) * 2

print("=" * 60)
print("9.1)-1: Dataset Created")
print("=" * 60)
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_train (first 5): {Y_train[:5]}")
print()

# ===========================
# 9.1)-2: Initialize h_0 (mean)
# ===========================
h_0 = np.mean(Y_train)
print("=" * 60)
print("9.1)-2: Initial Weak Classifier h_0")
print("=" * 60)
print(f"h_0 (mean of Y_train) = {h_0:.4f}")
print()

# Initialize ensemble: list of trees
H = [h_0]  # H^0 = {h_0}
ensemble_predictions = np.full(n_samples, h_0)

# ===========================
# Add weak classifiers h_t
# ===========================
for t in range(1, T):
    print("=" * 60)
    print(f"Iteration t = {t}")
    print("=" * 60)
    
    # ===========================
    # 9.1)-3: Compute residuals
    # ===========================
    # Predict using current ensemble
    residuals = Y_train - ensemble_predictions
    
    print(f"9.1)-3: Residuals")
    print(f"Predicted values (first 5): {ensemble_predictions[:5]}")
    print(f"Residuals (first 5):        {residuals[:5]}")
    print()
    
    # ===========================
    # 9.1)-4: Fit new tree h_t
    # ===========================
    # Create and fit decision tree to residuals
    h_t = DecisionTreeRegressor(max_depth=2, random_state=42)
    h_t.fit(X_train, residuals)
    
    # Predictions from new tree
    h_t_pred = h_t.predict(X_train)
    
    print(f"9.1)-4: New Weak Classifier h_{t}")
    print("Tree structure:")
    tree_rules = export_text(h_t, feature_names=['X'])
    print(tree_rules)
    
    # Updated ensemble predictions
    ensemble_predictions = ensemble_predictions + h_t_pred
    
    print(f"Updated predictions (first 5): {ensemble_predictions[:5]}")
    print()
    
    # ===========================
    # 9.1)-5: Check stopping criterion
    # ===========================
    # Compute sum of residuals * predictions from new tree
    stopping_criterion = np.sum(residuals * h_t_pred)
    
    print(f"9.1)-5: Stopping Criterion")
    print(f"Sum of (residuals * h_{t}): {stopping_criterion:.4f}")
    
    if stopping_criterion < 0:
        # Add tree to ensemble and continue
        H.append(h_t)
        print(f"Criterion < 0: Adding h_{t} to ensemble and continuing.")
    else:
        # Stop boosting
        print(f"Criterion >= 0: Stopping boosting.")
        print(f"Final ensemble: H^({t-1}) = {t} classifiers")
        break
    print()

# ===========================
# Final Summary
# ===========================
print("=" * 60)
print("FINAL ENSEMBLE")
print("=" * 60)
print(f"Total classifiers in ensemble: {len(H)}")
print(f"  h_0 (mean): {h_0:.4f}")
for i in range(1, len(H)):
    print(f"  h_{i}: DecisionTreeRegressor(max_depth=2)")
print()

# Final predictions
final_predictions = np.full(n_samples, h_0)
for i in range(1, len(H)):
    final_predictions += H[i].predict(X_train)

final_error = np.mean((Y_train - final_predictions) ** 2)
print(f"Final MSE on training data: {final_error:.4f}")
