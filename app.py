# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import folium
from streamlit_folium import st_folium
import base64

st.set_page_config(layout="wide", page_title="Cognifyz — Restaurants Dashboard")

# ---------- Helpers ----------
@st.cache_data
def load_data(path="Dataset.csv"):
    df = pd.read_csv(path, low_memory=False)
    return df

def download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

def create_folium_map(df, lat_col="Latitude", lon_col="Longitude", popup_cols=None, start_coords=(20,0), zoom=2):
    m = folium.Map(location=start_coords, zoom_start=zoom, tiles="OpenStreetMap")
    for _, r in df.iterrows():
        try:
            lat = float(r[lat_col]); lon = float(r[lon_col])
        except Exception:
            continue
        popup_text = ""
        if popup_cols:
            popup_text = "<br>".join([f"<b>{c}:</b> {r.get(c,'')}" for c in popup_cols])
        folium.CircleMarker(location=(lat, lon), radius=4, popup=popup_text,
                            color="blue", fill=True, fill_opacity=0.7).add_to(m)
    return m

# ---------- Load ----------
st.title("Cognifyz Internship — Restaurants Data Dashboard")
st.markdown("**Project:** Convert Level 1 & Level 2 tasks into an interactive portfolio dashboard.")
st.markdown("Source: Cognifyz internship task list (used as spec).") 
st.markdown("See internship tasks summary (Data Science.pdf).") 
st.caption("Task list & instructions used as the project's spec. :contentReference[oaicite:2]{index=2}")

with st.sidebar:
    st.header("Controls")
    data_path = st.text_input("Dataset path", value="Dataset.csv")
    run_button = st.button("Reload data")
    page = st.radio("Page", ["Overview", "EDA", "Map", "Feature Engineering", "Modeling", "Download & Files"])

# reload / load
if run_button:
    st.experimental_rerun()

try:
    df = load_data(data_path)
except Exception as e:
    st.error(f"Couldn't load dataset at {data_path}: {e}")
    st.stop()

# ---------- Overview ----------
if page == "Overview":
    st.header("Dataset overview")
    st.write("Rows, columns, and a quick preview of the data.")
    st.write("**Shape:**", df.shape)
    st.dataframe(df.head(10))
    st.subheader("Column types")
    st.write(df.dtypes)
    st.subheader("Missing values (per column)")
    missing = df.isna().sum().sort_values(ascending=False)
    st.bar_chart(missing[missing>0])

# ---------- EDA ----------
elif page == "EDA":
    st.header("Exploratory Data Analysis (Level 1 & Level 2 tasks)")
    st.subheader("Basic statistics")
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] > 0:
        st.write(numeric.describe().T)
    else:
        st.write("No numeric columns detected.")

    st.subheader("Target distribution: Aggregate rating")
    if "Aggregate rating" in df.columns:
        fig, ax = plt.subplots()
        df["Aggregate rating"].hist(bins=20, ax=ax)
        ax.set_xlabel("Aggregate rating")
        st.pyplot(fig)
        st.write("Value counts:")
        st.write(df["Aggregate rating"].value_counts().head())
    else:
        st.info("No 'Aggregate rating' column present in dataset.")

    st.subheader("Top cuisines and top cities")
    if "Cuisines" in df.columns:
        cuisines = (df["Cuisines"].astype(str)
                    .str.split(",", expand=True)
                    .stack()
                    .str.strip()
                    .value_counts().head(20))
        st.bar_chart(cuisines)
    if "City" in df.columns:
        st.write("Top cities:")
        st.table(df["City"].value_counts().head(15))

    st.subheader("Table booking & Online delivery analysis (Level 1 Task 3)")
    for col in ["Has Table booking", "Has Online delivery", "Table booking", "Online delivery"]:
        if col in df.columns:
            st.write(f"### {col}")
            vc = df[col].value_counts(normalize=True)*100
            st.write(vc)
            if "Aggregate rating" in df.columns:
                st.write("Average rating grouped by the column:")
                st.write(df.groupby(col)["Aggregate rating"].mean().sort_values(ascending=False).head())

# ---------- Map ----------
elif page == "Map":
    st.header("Geospatial Analysis (Level 1 Task 3)")
    st.markdown("If you want to use the pre-generated map HTML, it was included in the project files. You can embed it or regenerate from dataset.")
    st.markdown("Pre-generated `restaurants_map.html` is included in uploads. :contentReference[oaicite:3]{index=3}")

    lat_cols = [c for c in df.columns if "lat" in c.lower() or "latitude" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower() or "longitude" in c.lower()]
    if lat_cols and lon_cols:
        lat_col = st.selectbox("Latitude column", lat_cols, index=0)
        lon_col = st.selectbox("Longitude column", lon_cols, index=0)
        popup_cols = st.multiselect("Popup columns", options=list(df.columns), default=["Restaurant Name"] if "Restaurant Name" in df.columns else [])
        sample = st.slider("Sample points (0 = all)", 0, 5000, 0)
        if sample == 0:
            to_plot = df
        else:
            to_plot = df.sample(min(sample, len(df)), random_state=42)
        # center map around median coords if present
        try:
            center = (float(df[lat_col].median()), float(df[lon_col].median()))
        except Exception:
            center = (20,0)
        m = create_folium_map(to_plot, lat_col=lat_col, lon_col=lon_col, popup_cols=popup_cols, start_coords=center, zoom=4)
        st_data = st_folium(m, width=900, height=600)
    else:
        st.info("No latitude/longitude columns found. If you have `restaurants_map.html` uploaded you can view it by downloading and opening locally. Pre-generated HTML is in project files. :contentReference[oaicite:4]{index=4}")

# ---------- Feature Engineering ----------
elif page == "Feature Engineering":
    st.header("Feature Engineering (Level 2 Task 2)")
    st.write("Create simple features and inspect them.")
    # Example features:
    if "Restaurant Name" in df.columns:
        df["name_len"] = df["Restaurant Name"].astype(str).str.len()
        st.write("Added `name_len` (length of restaurant name).")
    if "Address" in df.columns:
        df["addr_len"] = df["Address"].astype(str).str.len()
        st.write("Added `addr_len` (length of address).")
    # Convert table booking/online delivery into boolean
    for col in ["Has Table booking", "Has Online delivery", "Table booking", "Online delivery"]:
        if col in df.columns:
            df[f"{col}_flag"] = df[col].astype(str).str.contains("Yes|yes|1|True|true", regex=True)
            st.write(f"Encoded {col} -> {col}_flag (boolean)")
    st.write("Preview of engineered columns:")
    st.dataframe(df.head(10))

# ---------- Modeling ----------
elif page == "Modeling":
    st.header("Predictive Modeling (Level 2 Task 3)")
    st.write("We'll build a simple regression model to predict `Aggregate rating`.")
    if "Aggregate rating" not in df.columns:
        st.warning("Dataset doesn't contain `Aggregate rating`. Can't train a regression model.")
    else:
        # Simple features selection: numeric features + engineered ones
        exclude = ["Aggregate rating", "Latitude", "Longitude", "Cuisines", "Address", "Restaurant Name", "City"]
        X = df.select_dtypes(include=[np.number]).copy()
        # If numeric features contain nulls - fill
        X = X.fillna(-999)
        y = df["Aggregate rating"].fillna(df["Aggregate rating"].median())
        st.write("Feature matrix shape:", X.shape)
        test_size = st.slider("Test size (%)", 10, 50, 25)
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state))
        n_estimators = st.slider("RF n_estimators", 10, 500, 100)
        max_depth = st.slider("RF max_depth (0 = None)", 0, 50, 8)
        model = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=(None if max_depth==0 else int(max_depth)),
                                      random_state=int(random_state))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        st.metric("MSE", f"{mse:.4f}")
        st.metric("R2", f"{r2:.4f}")
        st.subheader("Feature importances (top 20)")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
        st.table(importances)

# ---------- Download & Files ----------
elif page == "Download & Files":
    st.header("Download cleaned dataset & project files")
    st.markdown("If you want to download the current dataframe with engineered columns:")
    st.markdown(download_link(df, filename="dataset_with_features.csv"), unsafe_allow_html=True)
    st.write("---")
    st.subheader("Project files included (from upload)")
    st.write("- Cognifyz internship spec (Data Science.pdf).") 
    st.caption("Spec used to plan features & tasks. :contentReference[oaicite:5]{index=5}")
    st.write("- Pre-generated map HTML (restaurants_map.html) — can be embedded or served as static file in portfolio. :contentReference[oaicite:6]{index=6}")
    st.write("To include the interactive map as a static asset in a portfolio, upload `restaurants_map.html` to your GitHub pages or host alongside your portfolio and embed it.")

st.sidebar.write("Project: Cognifyz Internship Dashboard")
st.sidebar.caption("Use this app to show EDA, maps, feature engineering and a simple regression model suitable for portfolio demos.")
