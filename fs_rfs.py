import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix

file_path = r'D:\Neeraj\UoN\Machine Learning\Assignment 2\TrainDataset2025 (1).csv'
df = pd.read_csv(file_path, header=0)

df.rename(columns={df.columns[0]: 'ID', df.columns[1]: 'pCR', df.columns[2]:'RFS'}, inplace=True)

target = 'pCR'
exclude_cols = ['ID', 'RFS', '(outcome)']
critical_features = ['ER', 'HER2', 'Gene']

features_all = [c for c in df.columns if c not in [target] + exclude_cols]
selection_from = [c for c in features_all if c not in critical_features]

df.replace([999, 999.0, np.nan], 1, inplace=True)

df_cleaned = df.dropna(subset=[target])

X = df_cleaned[features_all]
y = df_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


selection_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', fill_value=0)),
    ('selector', SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        threshold='mean' 
    ))
])

critical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('select_step', selection_pipeline, selection_from),
        ('keep_step', critical_pipeline, critical_features) 
    ],
    remainder='drop'
)

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
])


param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [2, 3, 5],
    'classifier__class_weight':['balanced']
}

print("Starting Random Forest")
grid_search = GridSearchCV(
    estimator=full_pipeline, 
    param_grid=param_grid, 
    cv=5,                 
    scoring='f1', 
    verbose=1,            
    n_jobs=-1           
)

grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

bal_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test Balanced Accuracy: {bal_acc:.4f}")
print(f"Test F1 Score: {f1:.4f}")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

try:
    selector_step = best_model.named_steps['preprocessor'].named_transformers_['select_step'].named_steps['selector']
    mask = selector_step.get_support()
    selected_feats = np.array(selection_from)[mask]
    
    
    print(f"Extra Features Selected: {len(selected_feats)}")
    if len(selected_feats) > 0:
        print(selected_feats)
    else:
        print("None (All extra features were dropped).")
except Exception as e:
    print(f"Could not extract features: {e}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Feature Selection RFS')
plt.savefig('ConfusionMatrix_RFS_FS')
plt.show()