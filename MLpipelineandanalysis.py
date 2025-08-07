import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/Dataset.csv')

#1:Data Splitting

X = df.drop(['PerformanceLabel', 'EmployeeId', 'FirstName', 'LastName', 'HireDate','AverageRevenuePerCustomer'], axis=1)  
y = df['PerformanceLabel']  

#Train:80%, Test:10%, Holdout:10% 
X_temp, X_holdout, y_temp, y_holdout = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y, shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp, shuffle=True) 

print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
print("Holdout set size:", len(X_holdout))

#2:Spot-Check Algorythms--Initial Evaluation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_holdout_scaled = scaler.transform(X_holdout)

#classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "SVM": SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

#Cross-Validation(F1 score)
print("Cross-Validation Results:")
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='f1_weighted')  
    print(f"{name}: Mean F1 Score = {scores.mean():.3f}, Std Dev = {scores.std():.3f}")




print("\nTest Set Results:")
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"{name}: Test F1-score = {f1:.3f}")
    print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}\n")


#3:K-Ford Evaluation(F1-Score)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nK- Fold Cross-Validation Results (F1-Score):")

for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=kf, scoring='f1_weighted')
    print(f"{name}: Mean F1-Score = {scores.mean():.4f}, Std Dev = {scores.std():.4f}")


#4:Hyperparameter tuning

#A:Random Forest Hyperparameter tuning
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

rf_model = RandomForestClassifier(random_state=42)

grid_search_rf = GridSearchCV(rf_model, param_grid=param_grid_rf,cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)

grid_search_rf.fit(X_train_scaled, y_train)

print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
print("Best F1-Score for Random Forest:", grid_search_rf.best_score_)

#B:SVM Hyperparameter tuning
param_grid_svm = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
}

svm_model = SVC(random_state=42)

grid_search_svm = GridSearchCV(svm_model, param_grid=param_grid_svm,cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)
grid_search_svm.fit(X_train_scaled, y_train)

print("Best Parameters for SVM:", grid_search_svm.best_params_)
print("Best F1-Score for SVM:", grid_search_svm.best_score_)

#5:Training

#A:Descision Tree

rf_best_model = RandomForestClassifier(
    bootstrap=True,
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

rf_best_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_best_model.predict(X_test_scaled)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print("Random Forest F1-Score on Test Set:", f1_rf)

#B:SVM
best_svm_model = SVC(C=100, kernel='linear', degree=2, gamma='scale', random_state=42)

best_svm_model.fit(X_train_scaled, y_train)

y_pred = best_svm_model.predict(X_test_scaled)

test_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"SVM Test F1-Score: {test_f1:.4f}")

#6:Model Evaluation

#A:Descision Tree

X_holdout_scaled = scaler.transform(X_holdout)

y_pred_rf_holdout = rf_best_model.predict(X_holdout_scaled)

print("Random Forest Evaluation on Holdout Data")
print("F1-Score:", f1_score(y_holdout, y_pred_rf_holdout, average='weighted'))
print("Classification Report:\n", classification_report(y_holdout, y_pred_rf_holdout))
print("Confusion Matrix:\n", confusion_matrix(y_holdout, y_pred_rf_holdout))
print("-" * 50)

#B:SVM

y_holdout_pred = best_svm_model.predict(X_holdout_scaled)
holdout_f1 = f1_score(y_holdout, y_holdout_pred, average='weighted')

print(f"SVM Holdout F1-Score: {holdout_f1:.4f}")
print("Classification Report for SVM (Holdout Data):")
print(classification_report(y_holdout, y_holdout_pred))


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f'Confusion Matrix for {model_name} on Holdout Set')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

plot_confusion_matrix(y_holdout, y_pred_rf_holdout, "Random Forest")

y_holdout_pred = best_svm_model.predict(X_holdout_scaled)

cm = confusion_matrix(y_holdout, y_holdout_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Average Performer', 'High Performer', 'Low Performer'], yticklabels=['Average Performer', 'High Performer', 'Low Performer'])
plt.title("Confusion Matrix for SVM on Holdout Set")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

#Analyze the results
rf_feature_importance = rf_best_model.feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_feature_importance
}).sort_values(by='Importance', ascending=False)

def plot_feature_importance(importances_df, model_name):
    plt.figure(figsize=(8, 6))
    plt.barh(importances_df['Feature'], importances_df['Importance'], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title(f'Feature Importance for {model_name}')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

print("\nRandom Forest Feature Importance:")
print(rf_importance_df)

# Plot for Random Forest
plot_feature_importance(rf_importance_df, "Random Forest")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_holdout_scaled = scaler.transform(X_holdout)

best_svm_model = SVC(C=100, kernel='linear', degree=2, gamma='scale', random_state=42)

best_svm_model.fit(X_train_scaled, y_train)

#coefficients
coefficients = best_svm_model.coef_[0]  
feature_importance = pd.DataFrame({'Feature': X_train.columns,  'Coefficient': coefficients,'Importance': abs(coefficients)}).sort_values(by='Importance', ascending=False)

print("Feature Importance for SVM:")
print(feature_importance)
