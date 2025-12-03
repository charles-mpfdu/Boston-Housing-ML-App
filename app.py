import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO  # Used for robust string handling if needed


# --- Data Loading and Preprocessing ---
# We use st.cache_data so the function only runs once, saving computation time.
@st.cache_data
def load_and_preprocess_data(file_content):
    """
    Loads the Boston Housing dataset from the raw text content, handles the
    space-separated format, and preprocesses it.
    """
    st.info("Loading Boston Housing data and applying preprocessing...")

    column_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
    ]

    # Read the data content
    lines = file_content.splitlines()
    # Data lines start after the description (around line 22)
    data_lines = lines[22:]

    all_data = []
    for line in data_lines:
        # Split by one or more spaces, and filter out any empty strings
        parts = line.strip().split()
        all_data.extend(parts)

    try:
        # Reshape into 506 rows and 14 columns
        data_array = np.array(all_data).astype(float)
        df = pd.DataFrame(data_array.reshape(506, 14), columns=column_names)
    except ValueError as e:
        st.error(f"Error during data reshaping: {e}. Check the raw data format.")
        return pd.DataFrame()  # Return empty DataFrame on failure

    # Drop any potential NaN values (though the above logic should clean it)
    df = df.dropna()

    return df


# --- Model Training ---
# We use st.cache_resource to cache the trained model object itself.
@st.cache_resource
def train_linear_regression(X_train, y_train):
    """Trains the Multiple Linear Regression model."""
    st.info("Training Multiple Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.success("Model training complete!")
    return model


# ----------------------------------------------
# STREAMLIT APP LAYOUT
# ----------------------------------------------

st.set_page_config(
    page_title="Boston Housing ML Deployment",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏡 Boston Housing Price Predictor")
st.markdown(
    "A demonstration of Multiple Linear Regression analysis deployed via Streamlit."
)

# --- Load Data from the uploaded file ---
try:
    with open("boston_housing.data", "r") as f:
        boston_data_content = f.read()
except FileNotFoundError:
    st.error(
        "boston_housing.data not found. Please ensure the file is in the same directory."
    )
    st.stop()

df = load_and_preprocess_data(boston_data_content)

if df.empty:
    st.stop()

# --- SIDEBAR: Data Exploration ---
st.sidebar.header("Data Exploration")
if st.sidebar.checkbox("Show Raw Data", value=True):
    st.sidebar.subheader("First 5 Rows")
    st.sidebar.dataframe(df.head())
    st.sidebar.subheader("Data Statistics")
    st.sidebar.write(df.describe().T)

# --- MAIN PAGE: Model & Results ---

# 1. Split Data
X = df.drop("MEDV", axis=1)
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Train Model
model = train_linear_regression(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# 3. Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.subheader("Model Performance Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric(
        "R-squared ($R^2$)",
        f"{r2:.4f}",
        help="Proportion of the variance in the dependent variable that is predictable from the independent variables.",
    )
with col2:
    st.metric(
        "Mean Squared Error (MSE)",
        f"{mse:.2f}",
        help="Average squared difference between the estimated values and the actual value.",
    )

st.markdown("---")

# 4. Visualization (Actual vs Predicted Prices)
st.subheader("Actual vs Predicted Prices Scatter Plot")
fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, ax=ax_scatter, alpha=0.6, color="#1f77b4")
ax_scatter.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction",
)  # Perfect prediction line
ax_scatter.set_xlabel("Actual MEDV ($1000's)")
ax_scatter.set_ylabel("Predicted MEDV ($1000's)")
ax_scatter.set_title("Actual vs Predicted House Prices on Test Set")
ax_scatter.legend()
plt.tight_layout()
st.pyplot(fig_scatter)

st.markdown("---")

# 5. Visualization (Residual Analysis)
st.subheader("Residual Distribution (Prediction Errors)")
fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, ax=ax_hist, color="#2ca02c")
ax_hist.axvline(0, color="red", linestyle="--", label="Zero Residual")
ax_hist.set_title("Distribution of Residuals")
ax_hist.set_xlabel("Residual (Actual Price - Predicted Price)")
ax_hist.legend()
plt.tight_layout()
st.pyplot(fig_hist)

st.markdown("---")

# 6. Feature Importance (Coefficients)
st.subheader("Feature Coefficients (Influence on Price)")
coefficients_df = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": model.coef_}
).sort_values(by="Coefficient", ascending=False)

st.dataframe(coefficients_df, use_container_width=True)

st.markdown(
    """
    **Interpretation Note:**
    * A positive coefficient (e.g., `RM` - average rooms) means that increasing the feature value tends to increase the house price.
    * A negative coefficient (e.g., `LSTAT` - lower status population) means that increasing the feature value tends to decrease the house price.
    """
)
