import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler     
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score


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
    
    
    ('scaler', StandardScaler()), 
    
    ('selector', RFECV(
        estimator=LogisticRegression(
            class_weight='balanced', 
            penalty='l1',       
            solver='liblinear', 
            random_state=42,
            max_iter=100000
        ),
        step=1,
        cv=5,
        scoring='f1',
        min_features_to_select=2 
    ))
])


critical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', fill_value=0))
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
    ('classifier', RandomForestClassifier(random_state=42))
])



param_grid = {
    'classifier__class_weight': ['balanced'], 
    'classifier__max_depth': [2,3,5,7],        
    'classifier__n_estimators': [50,100,150,200],
}

print("Starting training: Logistic Regression Selection Random Forest Classification")
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
test_score = balanced_accuracy_score(y_test, y_pred)
F1_score=f1_score(y_test,y_pred)
print(grid_search.best_params_)
print(f"\nTest Balanced Accuracy: {test_score:.4f}")
print(f"\nF1 Score: {F1_score:4f}")

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix RFECV LR')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
plt.savefig('ConfusionMatrix_RFECV_LR')


try:
    selector_step = best_model.named_steps['preprocessor'].named_transformers_['select_step'].named_steps['selector']
    selected_mask = selector_step.support_
    selected_extra_features = np.array(selection_from)[selected_mask]
    
    print(f"\nLogistic Regression Selected {len(selected_extra_features)} extra features:")
    print(selected_extra_features)
except:
    print("Could not extract feature names.")