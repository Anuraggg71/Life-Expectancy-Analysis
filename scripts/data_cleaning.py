import pandas as pd
from sklearn.impute import SimpleImputer 

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    imputer = SimpleImputer(strategy='mean')
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = imputer.fit_transform(df[[col]])

    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved at: {output_path}")

if __name__ == "__main__":
    clean_data('data/Life_Expectancy_Data.csv', 'data/cleaned_data.csv')
