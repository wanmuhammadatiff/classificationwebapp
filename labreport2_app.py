import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Classification Web App (Decision Tree, Random Forest, Gradient Boosting)", layout="wide")

st.title("Classification Web App (Decision Tree, Random Forest, Gradient Boosting)")
st.write(
    "Load Online Payments Fraud Detection Dataset, select features & target, train a model classifier and inspect performance."
)

# Sidebar - logo / authors
st.sidebar.image(
    "https://brand.umpsa.edu.my/images/2024/02/29/umpsa-bangunan__1764x719.png",
    use_container_width=True,
)
st.sidebar.header("Developers:")
st.sidebar.write("- Wan Muhammad Atiff Wan Ahmad Rafidi")

# -----------------------------
# Dataset upload
# -----------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")
    return df

with st.spinner("Loading dataset... please wait ⚙️"):
    df = load_dataset()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Missing Values Check")

@st.cache_data(show_spinner=False)
def check_missing(df):
    return df.isnull().sum()

with st.spinner("Checking missing values... please wait ⚙️"):
    missing_summary = check_missing(df)

total_missing = missing_summary.sum()


if total_missing == 0:
    st.success("No missing values found in the dataset.")
else:
    st.warning(f"Detected {total_missing} missing values in the dataset.")
    st.write("Missing values by column:")
    st.write(missing_summary[missing_summary > 0])

# -----------------------------
# Features & target selection
# -----------------------------
st.sidebar.header("Features & Target")

all_columns = df.columns.tolist()
if len(all_columns) == 0:
    st.error("Uploaded file appears empty.")
    st.stop()

# ---- Set default target column ----
default_target = "isFraud"
if default_target in all_columns:
    target_index = all_columns.index(default_target)
else:
    target_index = len(all_columns) - 1   # fallback

target_col = st.sidebar.selectbox(
    "Select target column (y)",
    all_columns,
    index=target_index
)

# ---- Feature columns ----
feature_cols = st.sidebar.multiselect(
    "Select feature columns (X)",
    [c for c in all_columns if c != target_col],
    default=[c for c in all_columns if c != target_col],
)

if len(feature_cols) == 0:
    st.error("Please select at least one feature column.")
    st.stop()


# -----------------------------
# Encode categorical features & target
# -----------------------------
df_processed = df.copy()
label_encoders = {}

for col in feature_cols:
    if not np.issubdtype(df_processed[col].dtype, np.number):
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        st.sidebar.info(f"Encoded feature: `{col}`")

if not np.issubdtype(df_processed[target_col].dtype, np.number):
    le = LabelEncoder()
    df_processed[target_col] = le.fit_transform(df_processed[target_col].astype(str))
    label_encoders[target_col] = le
    st.sidebar.info(f"Encoded target: `{target_col}`")

st.sidebar.subheader("Data Sampling")
use_sampling = st.sidebar.checkbox("Use 1,000,000-row sample (recommended for speed)", value=True)

if use_sampling and len(df_processed) > 1_000_000:
    frac = 1_000_000 / len(df_processed)

    df_processed = df_processed.groupby(target_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=42),
        include_groups=True
    ).reset_index(drop=True)

    st.write("Dataset size after sampling:", len(df_processed))

else:
    st.write("No sampling applied. Using full dataset:", len(df_processed))


X = df_processed[feature_cols].values
y = df_processed[target_col].values

# -----------------------------
# Train/test split settings
# -----------------------------
st.sidebar.header("Train/Test Split")
test_size = st.sidebar.slider("Test size (proportion)", 0.1, 0.5, 0.3, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

# Optional scaling
scale_features = st.sidebar.checkbox("Apply StandardScaler to features", value=True)

if scale_features:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# -----------------------------
# Model Selection
# -----------------------------
st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose a classification model",
    ("Decision Tree", "Random Forest", "Gradient Boosting")
)

def get_model(choice, random_state):
    if choice == "Decision Tree":
        return DecisionTreeClassifier(random_state=random_state)
    elif choice == "Random Forest":
        return RandomForestClassifier(n_estimators=50,max_depth=10,n_jobs=-1,random_state=random_state)
    elif choice == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=50,learning_rate=0.1,max_depth=3,random_state=random_state)

model = get_model(model_choice, random_state)

# -----------------------------
# Train model
# -----------------------------

@st.cache_data(show_spinner=False)
def train_model_cached(model_choice, X_train, y_train, random_state):
    model = get_model(model_choice, random_state)
    model.fit(X_train, y_train)
    return model

with st.spinner("Training model... please wait ⚙️"):
    trained_model = train_model_cached(model_choice, X_train, y_train, random_state)

y_pred = trained_model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
st.subheader("Model Evaluation Results")

@st.cache_data(show_spinner=False)
def evaluate_model_cached(model_choice, X_test, y_test, _trained_model):
    y_pred = _trained_model.predict(X_test)
    y_prob = _trained_model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob

with st.spinner("Evaluating model results... 📊"):
    y_pred, y_prob = evaluate_model_cached(model_choice, X_test, y_test, trained_model)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# AUC
auc = roc_auc_score(y_test, y_prob)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# -------------------
# Display metrics
# -------------------
eval_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "RMSE", "AUC"],
    "Value": [accuracy, precision, recall, f1, rmse, auc]
})

st.subheader("Evaluation Summary Table")
st.dataframe(eval_table.style.format({"Value": "{:.4f}"}))

label_names = {0: "Not Fraud", 1: "Fraud"}

labels = sorted(np.unique(np.concatenate([y_test, y_pred])))

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {label_names[lbl]}" for lbl in labels],
    columns=[f"Pred {label_names[lbl]}" for lbl in labels]
)

st.markdown("### Confusion Matrix (interactive table)")
st.dataframe(cm_df)

# -------------------
# ROC Curve
# -------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve - {model_choice}")
ax.legend(loc="lower right")

st.markdown("### ROC Curve")
st.pyplot(fig)

# -------------------
# Debug Info
# -------------------
st.markdown("### Debug Info")

st.write("Total rows in FULL dataset:", len(df))
st.write("Rows in TRAIN set:", len(y_train))
st.write("Rows in TEST set:", len(y_test))

st.write("y_pred length:", len(y_pred))
st.write("Confusion matrix total:", cm.sum())



