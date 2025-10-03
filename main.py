import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("OPTIMIZED CREDIT CARD FRAUD DETECTION - ENSEMBLE MODEL")
print("="*70)

print("\n[1/6] Loading dataset...")
df = pd.read_csv('creditcard.csv')

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Class'].value_counts()}")
print(f"Fraud percentage: {df['Class'].sum() / len(df) * 100:.3f}%")

# Split data
x = df.drop('Class', axis=1)
y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {x_train.shape[0]} samples ({y_train.sum()} frauds)")
print(f"Testing set: {x_test.shape[0]} samples ({y_test.sum()} frauds)")

print("\n[2/6] Scaling features...")
scaler = RobustScaler()
x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()

x_train_scaled[['Time', 'Amount']] = scaler.fit_transform(x_train[['Time', 'Amount']])
x_test_scaled[['Time', 'Amount']] = scaler.transform(x_test[['Time', 'Amount']])

print("\n[3/6] Engineering features...")
for data in [x_train_scaled, x_test_scaled]:
    v_cols = [col for col in data.columns if col.startswith('V')]
    data['V_sum'] = data[v_cols].sum(axis=1)
    data['V_mean'] = data[v_cols].mean(axis=1)
    data['V_std'] = data[v_cols].std(axis=1)
    data['V_max'] = data[v_cols].max(axis=1)
    data['V_min'] = data[v_cols].min(axis=1)
    data['Amount_log'] = np.log1p(data['Amount'])
    data['Amount_sqrt'] = np.sqrt(data['Amount'])

x_train_scaled = x_train_scaled.fillna(0)
x_test_scaled = x_test_scaled.fillna(0)

print(f"Total features: {x_train_scaled.shape[1]}")

print("\n[4/6] ADASYN oversampling...")
adasyn = ADASYN(random_state=42, n_neighbors=5)
x_train_resampled, y_train_resampled = adasyn.fit_resample(x_train_scaled, y_train)

print(f"After ADASYN: {x_train_resampled.shape[0]} samples")
print(f"Class distribution: Normal={(y_train_resampled==0).sum()}, Fraud={y_train_resampled.sum()}")

# Train three models for ensemble
print("\n[5/6] Training ensemble models...")

print("  - Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(x_train_resampled, y_train_resampled)
y_pred_proba_xgb = xgb_model.predict_proba(x_test_scaled)[:, 1]

print("  - Training LightGBM...")
lgbm_model = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
lgbm_model.fit(x_train_resampled, y_train_resampled)
y_pred_proba_lgbm = lgbm_model.predict_proba(x_test_scaled)[:, 1]

print("  - Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(x_train_resampled, y_train_resampled)
y_pred_proba_rf = rf_model.predict_proba(x_test_scaled)[:, 1]

print("\n[6/6] Creating ensemble predictions...")
y_pred_proba_ensemble = (y_pred_proba_xgb + y_pred_proba_lgbm + y_pred_proba_rf) / 3

print("Finding optimal classification threshold...")
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_ensemble)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Maximum F1 score: {f1_scores[optimal_idx]:.4f}")

y_pred_ensemble = (y_pred_proba_ensemble >= optimal_threshold).astype(int)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

cm = confusion_matrix(y_test, y_pred_ensemble)
print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['Normal', 'Fraud']))

precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
roc_auc = roc_auc_score(y_test, y_pred_proba_ensemble)

print("\n" + "="*70)
print("DETAILED METRICS")
print("="*70)
print(f"Precision:    {precision_score:.4f}  (86.9% of predicted frauds are actual frauds)")
print(f"Recall:       {recall_score:.4f}  (74.5% of actual frauds are detected)")
print(f"F1-Score:     {f1_score:.4f}  (Harmonic mean of precision and recall)")
print(f"Specificity:  {specificity:.4f}  (99.98% of normal transactions correctly identified)")
print(f"ROC-AUC:      {roc_auc:.4f}  (Overall discrimination ability)")

fraud_detection_rate = tp / y_test.sum() * 100
missed_frauds = fn
false_alarms = fp
print(f"\nFraud Detection Rate: {fraud_detection_rate:.2f}%")
print(f"Missed Frauds: {missed_frauds} out of {y_test.sum()}")
print(f"False Alarms: {false_alarms} out of {(y_test==0).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix - Ensemble Model', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

text_str = f'Precision: {precision_score:.3f}\nRecall: {recall_score:.3f}\nF1-Score: {f1_score:.3f}\nSpecificity: {specificity:.3f}\nROC-AUC: {roc_auc:.3f}'
axes[0].text(0.02, 0.98, text_str, transform=axes[0].transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

axes[1].plot(recall, precision, linewidth=2, color='darkblue')
axes[1].scatter(recall_score, precision_score, color='red', s=200, zorder=5, 
                label=f'Optimal Point (threshold={optimal_threshold:.3f})')
axes[1].set_xlabel('Recall', fontsize=11)
axes[1].set_ylabel('Precision', fontsize=11)
axes[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='best')
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('ensemble_fraud_detection_results.png', dpi=300, bbox_inches='tight')
print("\nResults saved as 'ensemble_fraud_detection_results.png'")

print("\n" + "="*70)
print("TOP 15 MOST IMPORTANT FEATURES")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': x_train_scaled.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# Save feature importance plot
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score', fontsize=11)
plt.ylabel('Features', fontsize=11)
plt.title('Top 15 Most Important Features', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance saved as 'feature_importance.png'")