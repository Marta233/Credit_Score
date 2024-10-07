from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import pandas as pd

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
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        }
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def split_data(self):
        """Split the data into training and testing sets."""
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def train_models(self):
        """Train all selected models on the training data."""
        self.trained_models = {}
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
            print(f"{name} trained.")

    def evaluate_models(self):
        """Evaluate trained models on the test data and return results in a DataFrame."""
        results = []
        probabilities = {}

        for name, model in self.trained_models.items():
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

            # Store probabilities for the positive class
            probabilities[name] = y_proba

        results_df = pd.DataFrame(results)
        return results_df  # Return both results and probabilities

    def hyperparameter_tuning(self, model_name, param_grid):
        """Perform hyperparameter tuning for a specified model."""
        if model_name not in self.models:
            raise ValueError(f"{model_name} is not a valid model.")

        model = self.models[model_name]
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

        search.fit(self.X_train, self.y_train)
        print(f"Best parameters for {model_name}: {search.best_params_}")

        # Update the trained model with the best estimator
        self.trained_models[model_name] = search.best_estimator_
    def train_and_evaluate(self, model_name, param_grid):
        """Train, tune, and evaluate a specified model, returning risk probabilities."""
        # Perform hyperparameter tuning
        self.hyperparameter_tuning(model_name, param_grid)

        # Train the model with the best parameters
        model = self.trained_models[model_name]
        model.fit(self.X_train, self.y_train)

        # Evaluate the model
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]  # Probability for the positive class

        # Calculate evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

        # Return risk probabilities
        return pd.DataFrame({
            'CustomerID': self.X_test.index,  # Assuming your test set has an index or identifier
            'RiskProbability': y_proba
        })