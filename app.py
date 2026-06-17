
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Adoption Analytics", layout="wide")
st.title("AI Adoption Analytics Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("Dataset.csv").head(1000)
    
    # Text values (Increased, Decreased, Same) ko numbers mein convert karne ke liye
    if "Productivity_Change" in df.columns:
        df["Productivity_Change"] = df["Productivity_Change"].astype(str).str.replace("%", "").str.strip()
        
        # Text values ko numeric values se map kiya
        mapping = {"Increased": 15.0, "Same": 0.0, "Decreased": -10.0}
        df["Productivity_Change"] = df["Productivity_Change"].replace(mapping)
        
        # Column ko float/numeric banaya
        df["Productivity_Change"] = pd.to_numeric(df["Productivity_Change"], errors='coerce')
        df["Productivity_Change"] = df["Productivity_Change"].fillna(0.0)
        
    return df

df = load_data()

st.sidebar.header("Filters")
industry = st.sidebar.selectbox("Industry", ["All"] + sorted(df["Industry"].dropna().unique().tolist()))
if industry != "All":
    df = df[df["Industry"] == industry]

st.metric("Employees", len(df))
st.metric("Average AI Usage (hrs)", round(df["Daily_AI_Usage_Hours"].mean(),2))
# Column mein se agar koi extra spaces ya symbols hain unhe clean karne ke liye aur numbers mein badalne ke liye:
df["Productivity_Change"] = df["Productivity_Change"].astype(str).str.replace("%", "").str.strip()
df["Productivity_Change"] = pd.to_numeric(df["Productivity_Change"], errors='coerce')
st.metric("Average Productivity Change", f"{round(df['Productivity_Change'].mean(), 2)}%")

st.subheader("Industry Distribution")
st.plotly_chart(px.histogram(df, x="Industry"), use_container_width=True)

st.subheader("Primary AI Tool")
st.plotly_chart(px.pie(df, names="Primary_AI_Tool"), use_container_width=True)

st.subheader("AI Usage vs Productivity")
st.plotly_chart(
    px.scatter(df,
               x="Daily_AI_Usage_Hours",
               y="Productivity_Change",
               color="Job_Satisfaction"),
    use_container_width=True
)

data = df.copy()
le = LabelEncoder()
for c in data.select_dtypes(exclude="number"):
    data[c] = le.fit_transform(data[c].astype(str))

X = data.drop("Target", axis=1)
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

st.subheader("Feature Importance")
imp = (
    pd.Series(model.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
    .head(10)
)
st.bar_chart(imp)

st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)
