<div align="center">

<img src="logo.png" alt="CAR logo" width="110"/>

# CAR - Credit Assessment & Risk

### Loan Default Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Level-Beginner%20Friendly-brightgreen)

</div>

---

## 📌 Overview

Banks lose money when borrowers fail to repay loans (a **loan default**). **CAR (Credit Assessment & Risk)** is a beginner-friendly machine learning project that learns from **255,347 past loans** and predicts whether a new borrower will **repay (0)** or **default (1)**.

The project covers the full workflow: data cleaning, visualization, model training with 3 algorithms, and an interactive **Streamlit dashboard** with KPIs, graphs, model comparison and live prediction.

## ✨ Features

- 🧹 **Data cleaning** notebook, cell by cell, with explanations
- 📊 **Data visualization** notebook to find which factors affect default
- 🤖 **3 ML algorithms:** Logistic Regression, Decision Tree, Random Forest
- 📏 **4 metrics:** Accuracy, R2 Score, MAE, RMSE (accuracy kept between 80% and 90%)
- 🏆 **Automatic best-model selection**
- 🖥️ **Streamlit dashboard** with 4 tabs, 6 KPI cards, 6 graphs, filters and a prediction form

## 🗂️ Project Structure

```
Loan_Default_Project/
│
├── 1_Data_Cleaning.ipynb        # Step 1: clean the raw data
├── 2_Data_Visualization.ipynb   # Step 2: charts and insights
├── 3_ML_Model.ipynb             # Step 3: train and compare 3 models
├── CAR.py                       # Step 4: Streamlit dashboard app
├── logo.png                     # App logo
├── requirements.txt             # Python libraries
├── Loan_default.csv             # Raw dataset (add this file)
└── README.md
```

> `Loan_default_cleaned.csv` and `loan_model.pkl` are created automatically when you run the notebooks.

## 📁 Dataset

- **Rows:** 255,347 loans  **Columns:** 18 (17 after dropping `LoanID`)
- **Target:** `Default` (0 = repaid, 1 = defaulted)
- **Class balance:** 88.4% repaid, 11.6% defaulted

| Type | Columns |
|---|---|
| Numeric | Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio |
| Categorical | Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner |

The data has no missing values, no duplicates and no outliers.

## 🔄 Workflow

| Step | File | What happens |
|---|---|---|
| 1 | `1_Data_Cleaning.ipynb` | Check missing values, duplicates and outliers (IQR), drop `LoanID`, save `Loan_default_cleaned.csv` |
| 2 | `2_Data_Visualization.ipynb` | Distributions, boxplots, default rate by group, correlation heatmap |
| 3 | `3_ML_Model.ipynb` | Encode text columns, 80/20 split, scale, train 3 models, compare metrics, save best model |
| 4 | `CAR.py` | Interactive dashboard and prediction app |

## 🤖 Models and Results

| Algorithm | Settings | Accuracy | R2 Score | MAE | RMSE |
|---|---|---|---|---|---|
| **Logistic Regression** 🏆 | `max_iter=1000` | **88.52%** | -0.1180 | 0.1148 | 0.3388 |
| Decision Tree | `max_depth=5` | 88.49% | -0.1215 | 0.1151 | 0.3393 |
| Random Forest | `n_estimators=100, max_depth=8` | 88.40% | -0.1304 | 0.1160 | 0.3406 |

**Best model:** Logistic Regression (highest accuracy; lowest RMSE breaks ties).

**Notes**
- All three models score very close to each other. Only about 12% of loans default, so a model that always predicts "repay" would already reach about 88.4%. The accuracy range of 80-90% is natural for this dataset.
- R2, MAE and RMSE are regression metrics. Here they are calculated on the 0/1 predictions, so a slightly negative R2 is normal.

## 🔍 Key Insights

- 👶 Younger borrowers default more (20.8% for ages 18-25 vs 5.5% for ages 56-69)
- 📈 Higher interest rate means higher default rate (6.0% up to 17.8%)
- 💼 Unemployed borrowers default most (13.6%); full-time workers least (9.5%)
- 🤝 Loans without a co-signer default more (12.9% vs 10.4%)

## 🖥️ Dashboard (CAR.py)

| Tab | Contents |
|---|---|
| 📊 **Dashboard** | 6 KPIs (Total Loans, Default Rate, Avg Loan Amount, Avg Income, Avg Credit Score, Avg Interest Rate) and 6 graphs, with sidebar filters |
| 🏆 **Model Comparison** | Best-algorithm banner, KPIs, score table and 2 comparison charts |
| 🔮 **Predict Default** | Enter borrower details, pick an algorithm, get risk level (Low / Medium / High) and default probability |
| ℹ️ **About** | Project information and how to use the app |

> 📸 **Add screenshots:** save images in a `screenshots/` folder and show them here, for example:
> `![Dashboard](screenshots/dashboard.png)`

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install the requirements

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks in order

```bash
jupyter notebook
```

Open and **Run All** in this order:
1. `1_Data_Cleaning.ipynb`
2. `2_Data_Visualization.ipynb`
3. `3_ML_Model.ipynb`

### 5. Start the app

```bash
streamlit run CAR.py
```

The app opens at **http://localhost:8501**. On the first start it trains the 3 models (about 30 seconds); after that it is cached and fast.

## 🛠️ Tech Stack

| Purpose | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Web app | Streamlit |
| Notebooks | Jupyter |

## ⚠️ Limitations

- The dataset is imbalanced (about 12% defaults), so the models rarely predict "default" and accuracy alone can be misleading.
- Predictions are for learning and demonstration only, not for real lending decisions.

## 🔮 Future Improvements

- Handle class imbalance (class weights or SMOTE)
- Evaluate with precision, recall, F1 and ROC-AUC
- Try Gradient Boosting or XGBoost and tune hyperparameters
- Add a CSV upload to predict many borrowers at once
- Deploy the app on Streamlit Community Cloud

## 📄 License

This project is for educational purposes. Add a license of your choice (for example MIT) by creating a `LICENSE` file.

## 🙋 Author

**Your Name**
GitHub: [@your-username](https://github.com/your-username)

⭐ If you found this project helpful, please give it a star!
