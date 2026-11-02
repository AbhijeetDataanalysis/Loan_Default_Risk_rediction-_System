"""
Loan Risk Default Prediction Dashboard

Tabs:
    1. Dashboard        -> 6 KPIs + 6 graphs
    2. Model Comparison -> which of the 3 algorithms is the best
    3. Predict          -> predict default risk for a new borrower
    4. About            -> project information

How to run:
    streamlit run app.py

Keep these files in the same folder:
    app.py, Loan_default_cleaned.csv (created by notebook 1)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# =====================================================================
# 1. BASIC SETTINGS
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

# Colors used in all charts
TEAL, CORAL, BLUE, GOLD, PURPLE = "#2ec4b6", "#ff6b6b", "#4f8cff", "#f5b942", "#9b6bff"
TEXT_COLOR = "#9aa0a6"   # grey text: readable in both dark and light mode

EDUCATION_MAP = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}


def make_logo(path):
    """Draws a simple bank logo and saves it as logo.png (only if missing)."""
    if os.path.exists(path):
        return
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, 248, 248), radius=56, fill=(31, 78, 121, 255))
    d.polygon([(128, 48), (52, 98), (204, 98)], fill="white")            # roof
    d.ellipse((118, 66, 138, 86), fill=(31, 78, 121, 255))                # roof dot
    for x in (66, 102, 138, 174):                                         # pillars
        d.rectangle((x, 110, x + 16, 172), fill="white")
    d.rectangle((52, 182, 204, 196), fill="white")                        # step 1
    d.rectangle((40, 204, 216, 218), fill=(245, 185, 66, 255))            # step 2 (gold)
    img.save(path)


make_logo(LOGO_PATH)

# Page settings (must be the first Streamlit command)
st.set_page_config(
    page_title="CAR - Loan Default Prediction",
    page_icon=Image.open(LOGO_PATH),
    layout="wide",
)

# Small CSS for the KPI cards
st.markdown(
    """
<style>
.kpi {display:flex; align-items:center; gap:14px; padding:14px 16px; border-radius:12px;
      background:rgba(128,128,128,0.12); margin-bottom:10px;}
.kpi-icon {font-size:32px;}
.kpi-label {font-size:13px; opacity:0.7;}
.kpi-value {font-size:24px; font-weight:700;}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 2. LOAD DATA + TRAIN THE 3 MODELS  (cached, so it runs only once)
# =====================================================================
@st.cache_data
def load_data():
    for name in ("Loan_default_cleaned.csv", "Loan_default.csv"):
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            data = pd.read_csv(path)
            if "LoanID" in data.columns:
                data = data.drop("LoanID", axis=1)
            return data
    return None


def encode(data):
    """Same encoding that we used in 3_ML_Model.ipynb"""
    d = data.copy()
    for col in ["HasMortgage", "HasDependents", "HasCoSigner"]:
        d[col] = d[col].map({"Yes": 1, "No": 0})
    d["Education"] = d["Education"].map(EDUCATION_MAP)
    d = pd.get_dummies(d, columns=["EmploymentType", "MaritalStatus", "LoanPurpose"],
                       drop_first=True, dtype=int)
    return d


@st.cache_resource(show_spinner=False)
def train_models(_df):
    """Trains Logistic Regression, Decision Tree and Random Forest."""
    d = encode(_df)
    X = d.drop("Default", axis=1)
    y = d["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8,
                                                random_state=42, n_jobs=-1),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        rows.append({
            "Model": name,
            "Accuracy (%)": round(accuracy_score(y_test, pred) * 100, 2),
            "R2 Score": round(r2_score(y_test, pred), 4),
            "MAE": round(mean_absolute_error(y_test, pred), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        })

    return {"models": models, "scaler": scaler,
            "columns": X.columns.tolist(), "results": pd.DataFrame(rows)}


def pick_best(results):
    """Best = highest accuracy (inside 80-90%). If tie, lowest RMSE wins."""
    ok = results[(results["Accuracy (%)"] >= 80) & (results["Accuracy (%)"] <= 90)]
    if ok.empty:
        ok = results
    ok = ok.sort_values(["Accuracy (%)", "RMSE"], ascending=[False, True])
    return ok.iloc[0]["Model"]


# =====================================================================
# 3. HELPER FUNCTIONS (KPI cards + charts)
# =====================================================================
def kpi(col, icon, label, value, color):
    html = (f'<div class="kpi" style="border-left:6px solid {color}">'
            f'<div class="kpi-icon">{icon}</div>'
            f'<div><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div></div>')
    col.markdown(html, unsafe_allow_html=True)


def new_fig(width=5.5, height=3.5):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_alpha(0)          # transparent background
    ax.set_facecolor("none")
    ax.tick_params(colors=TEXT_COLOR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    return fig, ax


def show(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def donut_chart(data):
    counts = data["Default"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = new_fig()
    wedges, labels, percents = ax.pie(
        counts, labels=["Repaid", "Default"], colors=[TEAL, CORAL], autopct="%1.1f%%",
        startangle=90, pctdistance=0.78, wedgeprops=dict(width=0.45),
        textprops=dict(color=TEXT_COLOR))
    for p in percents:
        p.set_color("white")
        p.set_fontweight("bold")
    ax.set_title("Loan Status", color=TEXT_COLOR, fontweight="bold")
    return fig


def bar_chart(rate, title, color, horizontal=False):
    """rate = pandas Series of default rate (%)"""
    fig, ax = new_fig()
    labels = [str(i) for i in rate.index]
    if horizontal:
        ax.barh(labels, rate.values, color=color)
        for i, v in enumerate(rate.values):
            ax.text(v + 0.1, i, f"{v:.1f}%", va="center", color=TEXT_COLOR, fontsize=9)
        ax.set_xlabel("Default Rate (%)")
    else:
        ax.bar(labels, rate.values, color=color)
        for i, v in enumerate(rate.values):
            ax.text(i, v + 0.1, f"{v:.1f}%", ha="center", color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel("Default Rate (%)")
    ax.set_title(title, color=TEXT_COLOR, fontweight="bold")
    return fig


def line_chart(rate, title, color):
    fig, ax = new_fig()
    labels = [str(i) for i in rate.index]
    ax.plot(labels, rate.values, marker="o", linewidth=2.5, color=color)
    ax.fill_between(labels, rate.values, alpha=0.15, color=color)
    ax.margins(x=0.1)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Default Rate (%)")
    ax.set_title(title, color=TEXT_COLOR, fontweight="bold")
    return fig


def default_rate(data, column, bins=None, labels=None):
    """Default rate (%) for each group of a column."""
    if bins is not None:
        group = pd.cut(data[column], bins=bins, labels=labels, include_lowest=True)
    else:
        group = data[column]
    return data.groupby(group, observed=True)["Default"].mean() * 100


# =====================================================================
# 4. LOAD EVERYTHING
# =====================================================================
df = load_data()
if df is None:
    st.error("Data file not found. Please keep Loan_default_cleaned.csv in the same folder as app.py "
             "(run 1_Data_Cleaning.ipynb first).")
    st.stop()

with st.spinner("Training the 3 models for the first time... (about 20-40 seconds, only once)"):
    bundle = train_models(df)

results = bundle["results"]
best_name = pick_best(results)
best_row = results[results["Model"] == best_name].iloc[0]


# =====================================================================
# 5. SIDEBAR
# =====================================================================
if hasattr(st, "logo"):
    st.logo(LOGO_PATH)
st.sidebar.image(LOGO_PATH, width=90)
st.sidebar.markdown("## 🏦 CAR")
st.sidebar.caption("Credit Assessment & Risk")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Dashboard Filters")

purpose_list = sorted(df["LoanPurpose"].unique())
emp_list = sorted(df["EmploymentType"].unique())
sel_purpose = st.sidebar.multiselect("💼 Loan Purpose", purpose_list, default=purpose_list)
sel_emp = st.sidebar.multiselect("👔 Employment Type", emp_list, default=emp_list)

st.sidebar.markdown("---")
st.sidebar.success(f"🏆 Best model: **{best_name}**")
st.sidebar.caption("Filters apply to the Dashboard tab only.")

df_f = df[df["LoanPurpose"].isin(sel_purpose) & df["EmploymentType"].isin(sel_emp)]


# =====================================================================
# 6. HEADER
# =====================================================================
c_logo, c_title = st.columns([1, 9])
c_logo.image(LOGO_PATH, width=90)
c_title.title("LOAN DEFAULT RISK ANALYTICS & PREDICTION ")
c_title.caption("🏦 Loan Default Prediction Dashboard   |   📊 Analytics   •   🤖 Machine Learning   •   🔮 Risk Prediction")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🏆 Model Comparison", "🔮 Predict Default", "ℹ️ About"])


# =====================================================================
# TAB 1 : DASHBOARD  (6 KPIs + 6 graphs)
# =====================================================================
with tab1:
    st.subheader("📊 Loan Portfolio Overview")

    if df_f.empty:
        st.warning("No data for the selected filters. Please select at least one option in the sidebar.")
    else:
        # ---- 6 KPIs ----
        k = st.columns(3)
        kpi(k[0], "📄", "Total Loans", f"{len(df_f):,}", BLUE)
        kpi(k[1], "⚠️", "Default Rate", f"{df_f['Default'].mean() * 100:.2f}%", CORAL)
        kpi(k[2], "💰", "Avg Loan Amount", f"${df_f['LoanAmount'].mean():,.0f}", TEAL)
        k = st.columns(3)
        kpi(k[0], "💵", "Avg Income", f"${df_f['Income'].mean():,.0f}", GOLD)
        kpi(k[1], "⭐", "Avg Credit Score", f"{df_f['CreditScore'].mean():.0f}", PURPLE)
        kpi(k[2], "📈", "Avg Interest Rate", f"{df_f['InterestRate'].mean():.2f}%", CORAL)

        st.markdown("---")
        st.subheader("📉 Visual Analysis")

        # ---- 6 graphs (2 per row) ----
        g1, g2 = st.columns(2)
        with g1:
            show(donut_chart(df_f))
        with g2:
            age_rate = default_rate(df_f, "Age", [17, 25, 35, 45, 55, 70],
                                    ["18-25", "26-35", "36-45", "46-55", "56-69"])
            show(bar_chart(age_rate, "Default Rate by Age Group", BLUE))

        g3, g4 = st.columns(2)
        with g3:
            emp_rate = default_rate(df_f, "EmploymentType").sort_values()
            show(bar_chart(emp_rate, "Default Rate by Employment Type", CORAL, horizontal=True))
        with g4:
            cs_rate = default_rate(df_f, "CreditScore", [299, 499, 599, 699, 849],
                                   ["300-499", "500-599", "600-699", "700-850"])
            show(line_chart(cs_rate, "Default Rate by Credit Score", PURPLE))

        g5, g6 = st.columns(2)
        with g5:
            ir_rate = default_rate(df_f, "InterestRate", [0, 5, 10, 15, 20, 25],
                                   ["0-5%", "5-10%", "10-15%", "15-20%", "20-25%"])
            show(line_chart(ir_rate, "Default Rate by Interest Rate", GOLD))
        with g6:
            lp_rate = default_rate(df_f, "LoanPurpose").sort_values()
            show(bar_chart(lp_rate, "Default Rate by Loan Purpose", TEAL))

        st.info("💡 Younger borrowers, low credit scores, high interest rates and unemployment show the highest default rates.")


# =====================================================================
# TAB 2 : MODEL COMPARISON
# =====================================================================
with tab2:
    st.subheader("🏆 Which Algorithm is Best?")
    st.success(f"🏆 Best Algorithm: **{best_name}**  with  **{best_row['Accuracy (%)']}%** accuracy")

    # ---- KPIs of the best model ----
    k = st.columns(5)
    kpi(k[0], "🤖", "Best Algorithm", best_name, GOLD)
    kpi(k[1], "🎯", "Accuracy", f"{best_row['Accuracy (%)']}%", TEAL)
    kpi(k[2], "📐", "R2 Score", f"{best_row['R2 Score']}", PURPLE)
    kpi(k[3], "📏", "MAE", f"{best_row['MAE']}", BLUE)
    kpi(k[4], "📉", "RMSE", f"{best_row['RMSE']}", CORAL)

    st.markdown("### 📋 Score Table (all 3 algorithms)")
    table = results.copy()
    table.insert(0, "Rank", ["🥇" if m == best_name else "🥈" for m in table["Model"]])
    st.dataframe(table, hide_index=True)

    ch1, ch2 = st.columns(2)

    with ch1:
        fig, ax = new_fig()
        colors = [GOLD if m == best_name else BLUE for m in results["Model"]]
        ax.axhspan(80, 90, color=TEAL, alpha=0.15)
        ax.text(-0.4, 90.6, "Target zone: 80-90%", color=TEAL, ha="left", va="bottom", fontsize=9)
        ax.bar(results["Model"], results["Accuracy (%)"], color=colors)
        for i, v in enumerate(results["Accuracy (%)"]):
            ax.text(i, v + 0.3, f"{v}%", ha="center", color=TEXT_COLOR, fontsize=10)
        ax.set_ylim(70, 94)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy Comparison", color=TEXT_COLOR, fontweight="bold")
        show(fig)

    with ch2:
        fig, ax = new_fig()
        x = np.arange(len(results))
        ax.bar(x - 0.2, results["MAE"], width=0.4, label="MAE", color=BLUE)
        ax.bar(x + 0.2, results["RMSE"], width=0.4, label="RMSE", color=CORAL)
        ax.set_xticks(x)
        ax.set_xticklabels(results["Model"])
        ax.set_ylim(0, results["RMSE"].max() * 1.35)
        ax.set_ylabel("Error (lower is better)")
        ax.set_title("MAE and RMSE Comparison", color=TEXT_COLOR, fontweight="bold")
        ax.legend(frameon=False, labelcolor=TEXT_COLOR)
        show(fig)

    with st.expander("How is the best algorithm selected?"):
        st.write(
            "1. We keep only the models whose accuracy is between **80% and 90%**.\n"
            "2. The model with the **highest accuracy** wins.\n"
            "3. If accuracies are equal, the model with the **lowest RMSE** wins.\n\n"
            "Note: all three models score very close to each other, because only about 12% of "
            "loans default. R2, MAE and RMSE are calculated on the 0/1 predictions, so a small "
            "negative R2 is normal."
        )


# =====================================================================
# TAB 3 : PREDICT
# =====================================================================
with tab3:
    st.subheader("🔮 Predict Loan Default")
    st.write("Enter the borrower details and click **Predict**.")

    model_names = list(bundle["models"].keys())
    chosen = st.selectbox("🤖 Algorithm", model_names, index=model_names.index(best_name))

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("🎂 Age", 18, 69, 35)
            income = st.number_input("💵 Yearly Income", 15000, 150000, 60000, step=1000)
            loan_amount = st.number_input("💰 Loan Amount", 5000, 250000, 100000, step=1000)
            credit_score = st.number_input("⭐ Credit Score", 300, 850, 600)
            months_employed = st.number_input("🗓️ Months Employed", 0, 119, 40)
            num_credit_lines = st.number_input("💳 Number of Credit Lines", 1, 4, 2)
            interest_rate = st.number_input("📈 Interest Rate (%)", 2.0, 25.0, 12.0, step=0.1)
            loan_term = st.selectbox("⏳ Loan Term (months)", [12, 24, 36, 48, 60], index=2)
            dti_ratio = st.slider("⚖️ DTI Ratio (debt-to-income)", 0.10, 0.90, 0.50)
        with col2:
            education = st.selectbox("🎓 Education", list(EDUCATION_MAP.keys()))
            employment = st.selectbox("👔 Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
            marital = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"])
            purpose = st.selectbox("🎯 Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
            has_mortgage = st.selectbox("🏠 Has Mortgage?", ["Yes", "No"])
            has_dependents = st.selectbox("👨‍👩‍👧 Has Dependents?", ["Yes", "No"])
            has_cosigner = st.selectbox("🤝 Has Co-Signer?", ["Yes", "No"])

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        # 1. Put the inputs in a DataFrame (same style as the training data)
        row = pd.DataFrame([{
            "Age": age, "Income": income, "LoanAmount": loan_amount,
            "CreditScore": credit_score, "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
            "LoanTerm": loan_term, "DTIRatio": dti_ratio,
            "Education": EDUCATION_MAP[education],
            "EmploymentType": employment, "MaritalStatus": marital,
            "HasMortgage": 1 if has_mortgage == "Yes" else 0,
            "HasDependents": 1 if has_dependents == "Yes" else 0,
            "LoanPurpose": purpose,
            "HasCoSigner": 1 if has_cosigner == "Yes" else 0,
        }])

        # 2. One-hot encode and match the training columns
        row = pd.get_dummies(row, columns=["EmploymentType", "MaritalStatus", "LoanPurpose"], dtype=int)
        row = row.reindex(columns=bundle["columns"], fill_value=0)
        row_scaled = bundle["scaler"].transform(row)

        # 3. Predict with the chosen model
        model = bundle["models"][chosen]
        prediction = model.predict(row_scaled)[0]
        probability = model.predict_proba(row_scaled)[0][1] * 100

        # 4. Show the result
        st.markdown("---")
        st.subheader("📋 Result")
        if probability < 15:
            level, icon = "LOW RISK", "🟢"
        elif probability < 50:
            level, icon = "MEDIUM RISK", "🟡"
        else:
            level, icon = "HIGH RISK", "🔴"

        r1, r2, r3 = st.columns(3)
        kpi(r1, icon, "Risk Level", level, GOLD)
        kpi(r2, "📊", "Default Probability", f"{probability:.1f}%", CORAL)
        kpi(r3, "🤖", "Model Decision", "Will DEFAULT" if prediction == 1 else "Will REPAY", TEAL)
        st.progress(int(min(max(probability, 0), 100)))

        if prediction == 1:
            st.error("⚠️ High Risk: this borrower is likely to DEFAULT.")
        else:
            st.success("✅ Low Risk: this borrower is likely to REPAY.")

        # What do all 3 algorithms say?
        st.markdown("##### 🤖 What each algorithm says")
        compare = []
        for name, m in bundle["models"].items():
            p = m.predict_proba(row_scaled)[0][1] * 100
            compare.append({"Algorithm": name, "Default Probability (%)": round(p, 1),
                            "Decision": "DEFAULT" if m.predict(row_scaled)[0] == 1 else "REPAY"})
        st.dataframe(pd.DataFrame(compare), hide_index=True)


# =====================================================================
# TAB 4 : ABOUT
# =====================================================================
with tab4:
    st.subheader("ℹ️ About this Project")
    a1, a2 = st.columns([1, 4])
    a1.image(LOGO_PATH, width=120)
    a2.markdown(
        "**CAR (Credit Assessment & Risk)** predicts whether a borrower is likely to default on a loan.\n\n"
        f"- 📁 **Dataset:** {len(df):,} loans, {df.shape[1]} columns\n"
        "- 🤖 **Algorithms:** Logistic Regression, Decision Tree, Random Forest\n"
        "- 📏 **Metrics:** Accuracy, R2 Score, MAE, RMSE\n"
        "- 🛠️ **Tools:** Python, Pandas, Scikit-learn, Matplotlib, Streamlit"
    )
    st.markdown("### 🧭 How to use")
    st.markdown(
        "1. Open **📊 Dashboard** to explore the loan data (use the sidebar filters).\n"
        "2. Open **🏆 Model Comparison** to see which algorithm is the best.\n"
        "3. Open **🔮 Predict Default** to test a new borrower."
    )
    st.caption("Made with ❤️ using Streamlit  •  CAR © 2026")
