from scripts.data_cleaning import clean_data
from scripts.modeling import model_pipeline

if __name__ == "__main__":
    print("🚀 Starting Life Expectancy Analysis Project...")

    # Step 1: Clean data
    clean_data('data/Life_Expectancy_Data.csv', 'data/cleaned_data.csv')

    # Step 2: Run ML pipeline
    model_pipeline('data/cleaned_data.csv')

    print("🎉 Project pipeline completed successfully!")
