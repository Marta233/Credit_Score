from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import pandas as pd
import joblib
import os

class ModelTrainer:
    def __init__(self, data, target_column, drop_columns=None, test_size=0.2, random_state=42):
        self.data = data
        self.target_column = target_column
        self.drop_columns = drop_columns if drop_columns is not None else []
        self.test_size = test_size
        self.random_state = random_state

        # Drop specified columns
        self.data = self.data.drop(columns=self.drop_columns, errors='ignore')

        # Splitting features and target variable
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]

        # Initialize models
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=200, multi_class='ovr'),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def split_data(self):
        """Split the data into training and testing sets."""
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def evaluate_models(self, models_dict):
        """Evaluate trained models on the test data and return results in a DataFrame."""
        results = []
        for name, model in models_dict.items():
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]  # Probability for the positive class

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_proba)

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            })

        results_df = pd.DataFrame(results)
        return results_df

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for all models."""
        # Define parameter grids for all models
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'Decision Tree': {
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }

        # Perform tuning for each model
        self.tuned_models = {}
        for name, param_grid in param_grids.items():
            model = self.models[name]
            search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

            search.fit(self.X_train, self.y_train)
            print(f"Best parameters for {name}: {search.best_params_}")

            # Update the model with the best estimator
            self.tuned_models[name] = search.best_estimator_
    def Model_comparision(self):
        print("\nPerforming hyperparameter tuning...")
        self.hyperparameter_tuning()

        print("\nEvaluating models after hyperparameter tuning...")
        # Store the evaluation results in a DataFrame
        evaluation_results = self.evaluate_models(self.tuned_models)
        print(evaluation_results)  # Print the results DataFrame

    def save_best_model(self, save_path):
        """Save the best model based on F1 Score or any other metric."""
        tuned_results = self.evaluate_models(self.tuned_models)
        best_model_name = tuned_results.loc[tuned_results['F1 Score'].idxmax()]['Model']
        best_model = self.tuned_models[best_model_name]

        # Save the best model
        model_filename = f"{best_model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(best_model, os.path.join(save_path, model_filename))
        print(f"Best model '{best_model_name}' saved as {model_filename}")
        return best_model_name, model_filename

 


