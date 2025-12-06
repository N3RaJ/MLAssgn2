import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.tree import plot_tree 


file_path = r'D:\Neeraj\UoN\Machine Learning\Assignment 2\TrainDataset2025 (1).csv'
df = pd.read_csv(file_path, header=0)


df.rename(columns={df.columns[0]: 'ID', df.columns[1]: 'pCR', df.columns[2]:'RFS'}, inplace=True)
df.replace([999, 999.0], 1, inplace=True)


target = 'pCR'
#Features appearing most frequently on other models, I've rejected image data features as they were introducing unneccesary noise
critical_features = ['Gene','ER','HER2','PgR','TumourStage'] 
df_raw = pd.read_csv(file_path, header=0)


df_cleaned = df.dropna(subset=[target])


X = df_cleaned[critical_features]
y = df_cleaned[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


imputer = SimpleImputer(strategy='mean')


preprocessor = ColumnTransformer(
    transformers=[
        ('imputer_step', imputer, critical_features) 
    ],
    remainder='drop'
)


final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=150, 
        max_depth=2, 
        class_weight='balanced', 
        random_state=42
    ))
])



print("Training final model with optimized parameters...")
final_pipeline.fit(X_train, y_train)


y_pred = final_pipeline.predict(X_test)
test_score = balanced_accuracy_score(y_test, y_pred)
test_scoreF1= f1_score(y_test,y_pred)

print("\n" + "="*50)
print("FINAL MODEL PERFORMANCE")
print("="*50)
print(f"Test Balanced Accuracy: {test_score:.4f} (Max stable score)")
print(f"Test F1 Accuracy: {test_scoreF1:.4f}")


print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Final Confusion Matrix (Balanced Model)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
plt.savefig('ConfusionMatrix_RFS')


final_rf_model = final_pipeline.named_steps['classifier']

plt.figure(figsize=(15, 8))
plot_tree(final_rf_model.estimators_[0], 
          feature_names=critical_features,
          class_names=['No pCR', 'pCR'],
          filled=True, rounded=True, fontsize=10)
plt.title("Visualization of Decision Logic (Depth 2)")
plt.show()

importances = final_rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': critical_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Total Feature Usage (Across all 150 trees) ---")
print(feature_importance_df)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance (Entire Random Forest)')
plt.show()
plt.savefig('Feature Importance - Critical Features')