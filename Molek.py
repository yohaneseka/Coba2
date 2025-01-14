import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Title
st.title("Analisis Statistik dan Regresi")
st.write("Masukkan data sesuai dengan variabel yang diminta, lalu klik 'Proses Analisis' untuk mendapatkan hasil analisis deskriptif, korelasi, dan regresi.")

# Input Data
st.subheader("Input Data")
n = st.number_input("Jumlah Data (N)", min_value=1, value=36, step=1)

st.write("Masukkan nilai untuk setiap variabel:")
cols = ["Kualitas Layanan", "Citra Coffee Shop", "Kepercayaan Merek", "Loyalitas"]
data_input = {}
for col in cols:
    data_input[col] = st.text_area(f"Masukkan data untuk {col} (pisahkan dengan koma)", placeholder="Contoh: 10,20,30")

if st.button("Proses Analisis"):
    try:
        # Parse data input
        data = {}
        for col in cols:
            data[col] = list(map(float, data_input[col].split(',')))

        # Create DataFrame
        df = pd.DataFrame(data)

        # Descriptive Statistics
        st.subheader("Statistik Deskriptif")
        descriptive_stats = df.describe().T
        descriptive_stats = descriptive_stats.rename(columns={"mean": "Mean", "std": "Std. Deviation", "min": "Minimum", "max": "Maximum"})
        st.write(descriptive_stats)

        # Correlation Analysis
        st.subheader("Analisis Korelasi")
        correlation_matrix = df.corr()
        st.write("Matriks Korelasi:")
        st.write(correlation_matrix)

        # Regression Analysis
        st.subheader("Analisis Regresi")
        X = df[["Kualitas Layanan", "Citra Coffee Shop", "Kepercayaan Merek"]]
        Y = df["Loyalitas"]

        model = LinearRegression()
        model.fit(X, Y)
        predictions = model.predict(X)

        coefficients = model.coef_
        intercept = model.intercept_
        r2 = r2_score(Y, predictions)

        st.write("Persamaan Regresi:")
        st.write(f"Y = {intercept:.3f} + ({coefficients[0]:.3f})X1 + ({coefficients[1]:.3f})X2 + ({coefficients[2]:.3f})X3")
        st.write(f"R-squared: {r2:.3f}")

        st.write("Koefisien Regresi:")
        coef_table = pd.DataFrame({"Variable": X.columns, "Coefficient": coefficients})
        st.write(coef_table)

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam memproses data: {e}")