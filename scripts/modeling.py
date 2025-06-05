import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

def model_pipeline(data_path):
    df = pd.read_csv(data_path)

    # Encode categorical
    le = LabelEncoder()
    df['Country'] = le.fit_transform(df['Country'])
    df['Status'] = le.fit_transform(df['Status'])

    X = df.drop(columns='Life expectancy ')
    y = df['Life expectancy ']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Extra Trees': ExtraTreesRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor()
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({'Model': name, 'RMSE': rmse, 'R2 Score': r2})

    results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)

    # Plot R2
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x='Model', y='R2 Score')
    plt.title('R2 Score Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/r2_comparison.png')

    # Plot RMSE
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x='Model', y='RMSE')
    plt.title('RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/rmse_comparison.png')

    # Cross-validation on best model
    best_model = XGBRegressor()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(best_model, X_scaled, y, cv=kf, scoring='r2')

    plt.figure(figsize=(8, 5))
    plt.plot(scores, marker='o')
    plt.title('Cross-Validation R2 Scores')
    plt.xlabel('Fold')
    plt.ylabel('R2 Score')
    plt.tight_layout()
    plt.savefig('outputs/cross_val_scores.png')

    print("✅ Modeling complete. Graphs saved to outputs/ folder.")

if __name__ == "__main__":
    model_pipeline('data/cleaned_data.csv')
