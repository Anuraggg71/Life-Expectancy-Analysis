# 🧠 Life Expectancy Analysis (Internship Project)

### 👤 Submitted By: **Anurag Dewangan**   
📅 Submission Date: **25/06/2025**

---

## 📌 Project Overview

**Life Expectancy Analysis** is an advanced-level data analytics internship project that aims to predict 
life expectancy using historical health, economic, and social indicators collected from the World Health Organization
(WHO) and the United Nations.
We analyze data from **193 countries between 2000–2015**, explore important trends, and build regression models to 
identify key factors that impact human lifespan.

---

## 🎯 Project Objectives

- Understand how different factors affect life expectancy
- Compare life expectancy across developed vs developing countries
- Predict life expectancy using machine learning models
- Visualize key health and economic indicators interactively

---

## 🧰 Tools & Technologies Used

| Tool             | Purpose                           |
|------------------|------------------------------------|
| Python           | Main language                     |
| Pandas, NumPy    | Data cleaning and transformation  |
| Matplotlib, Seaborn, Plotly | Data visualization         |
| Scikit-learn     | Machine learning models           |
| XGBoost          | High-performance regression model |
| Jupyter Notebook | Analysis and visual development   |
| VS Code          | Script-based development          |
| Git & GitHub     | Version control and sharing       |

---

## 📁 Project Folder Structure

```
Life_Expectancy_Analysis/
├── data/                    # Raw & cleaned dataset
├── scripts/                 # Python scripts for cleaning & modeling
├── notebooks/               # Jupyter notebooks for analysis
├── outputs/                 # Saved charts & graphs
├── main.py                  # Project entry point
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation (this file)
```

---

## 🔍 Exploratory Data Analysis (EDA)

- Correlation Matrix of all features
- Trend: Life Expectancy over Years
- Comparison: Developed vs Developing countries
- Relationship: Life Expectancy vs GDP, Alcohol, Schooling
- Interactive visualizations using Plotly

---

## 🤖 Machine Learning Models

| Model                  | R² Score | RMSE |
|------------------------|----------|------|
| XGBoost Regressor      | ~0.96    | ~1.9 |
| Extra Trees Regressor  | ~0.96    | ~2.0 |
| Random Forest Regressor| ~0.95    | ~2.0 |
| Gradient Boosting      | ~0.94    | ~2.4 |

✅ Best Model: **XGBoost**

---

## 📈 Output Graphs

Saved in the `outputs/` folder:
- `correlation_matrix.png`
- `r2_comparison.png`
- `rmse_comparison.png`
- `cross_val_scores.png`
- `Average Life Expectancy By country Status.png`
- `Average Life Expectancy over the years.png`
- `Life Expectancy vs Alcohol Consumption Over the year.png`
- `Life Expectancy vs GDP.png`
- `Life Expectancy vs Years of schooling.png`
---

## 💡 Key Insights

- Countries with higher GDP and schooling tend to have higher life expectancy
- Alcohol consumption has a varying impact depending on region
- Mortality rates and immunization are strongly correlated with lifespan
- Developed countries consistently show higher life expectancy

---

## Conclusion

- This analysis shows that factors like education, healthcare spending, immunization, and GDP strongly improve life expectancy, while high mortality rates and undernutrition reduce it. Our machine learning model (XGBoost) predicted life expectancy with high accuracy (R² = 0.96), helping identify key areas for public health improvement and policy focus.

---

## 🚀 How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the project

```bash
python main.py
```

It will:
- Clean the data
- Train ML models
- Save charts in the `outputs/` folder

---

## 📤 Submission & Author

**Name**: Anurag Dewangan  
**Project Title**: Life Expectancy Analysis  
**Submission Date**: 26 June 2025  
**Internship Role**: Data Analytics Intern  
**Tools Used**: Python, Excel, SQL, VS Code, Jupyter Notebook

---

## 🔗 GitHub Link

👉 [GitHub Repository](https://github.com/Anuraggg71/Life-Expectancy-Analysis)

---

## 🏁 Final Note

This project demonstrates real-world data analytics skills, EDA techniques, and regression modeling, tailored for 
internship-level learning and showcasing. It helped me strengthen my foundation in Python and machine learning as 
a Data Analytics student passionate about data.

---
