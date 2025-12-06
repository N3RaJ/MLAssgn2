import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score,classification_report, confusion_matrix


file_path = r'D:\Neeraj\UoN\Machine Learning\Assignment 2\TrainDataset2025 (1).csv'
df = pd.read_csv(file_path, header=0)

df.rename(columns={df.columns[0]: 'ID', df.columns[1]: 'pCR', df.columns[2]:'RFS'}, inplace=True)

target = 'pCR'
exclude_cols = ['ID', 'RFS', '(outcome)']
critical_features = ['ER', 'HER2', 'Gene']

features_all = [c for c in df.columns if c not in [target] + exclude_cols]
selection_pool = [c for c in features_all if c not in critical_features]


df.replace([999, 999.0, np.nan], np.nan, inplace=True)

df_cleaned = df.dropna(subset=[target])
X = df_cleaned[features_all]
y = df_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



imputer = SimpleImputer(strategy='mean', fill_value=0)
X_train_pool = X_train[selection_pool]
X_train_pool_imputed = pd.DataFrame(imputer.fit_transform(X_train_pool), columns=selection_pool)


rfecv = RFECV(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    step=1,
    cv=5,
    scoring='f1',
    min_features_to_select=1,
    n_jobs=-1,
    importance_getter='auto'
)

rfecv.fit(X_train_pool_imputed, y_train)

mask = rfecv.support_
selected_extra_features = np.array(selection_pool)[mask]

print(f"Original Pool Size: {len(selection_pool)}")
print(f"Features Selected by RFECV: {len(selected_extra_features)}")
if len(selected_extra_features) > 0:
    print(f"Names: {selected_extra_features}")
else:
    print("RFECV dropped all extra features.")

final_features = critical_features + list(selected_extra_features)
print(f"Total Features for Modeling: {len(final_features)}")

X_train_reduced = X_train[final_features].copy()
X_test_reduced = X_test[final_features].copy()




final_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', fill_value=0)),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__class_weight': ['balanced'], 
    'classifier__max_depth': [2, 3, 5, 7, 9, 11], 
    'classifier__n_estimators': [50, 100, 150, 200, 250, 300]
}

grid_search = GridSearchCV(
    estimator=final_pipeline, 
    param_grid=param_grid, 
    cv=5,                 
    scoring='f1', 
    verbose=1,            
    n_jobs=-1             
)

grid_search.fit(X_train_reduced, y_train)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_reduced) 

test_score = balanced_accuracy_score(y_test, y_pred)
F1_score=f1_score(y_test,y_pred)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"\nTest Balanced Accuracy: {test_score:.4f}")
print(f"\nF1 Score: {F1_score:.4f}")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix RFECV RFS')
plt.show()
plt.savefig('ConfusionMatrix_RFECV_RFS')