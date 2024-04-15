import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn import metrics

st.set_page_config(page_title="Prediksi Penjualan Barang Almey Petshop", layout="wide")
# Create menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Data Visualisation", "Prediction"],
    icons=["house", "book", "calculator"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

#row0_spacer1, row0_1, row0_spacer2= st.columns((0.1, 3.2, 0.1))
#row1_spacer1, row1_1, row1_spacer2, row1_2 = st.columns((0.1, 1.5, 0.1, 1.5))
#row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))
#row0_spacer3, row3_0, row0_spacer3= st.columns((0.1, 3.2, 0.1))

row0_spacer1, row0_1, row0_spacer2 = st.columns((0.1, 3.2, 0.1))
row1_spacer1, row1_1, row1_spacer2, row1_2 = st.columns((0.1, 1.5, 0.1, 1.5))
row0_spacer3, row3_0, row0_spacer4 = st.columns((0.1, 3.2, 0.1))

# Load dataset
df = pd.read_csv('dataset.csv')

# Model
model = pd.read_pickle('model_svr.pkl')

# Handle selected option
if selected == "Home":
    row0_1.title("Aplikasi Prediksi Penjualan Barang Almey Petshop menggunakan Support Vector Regression (SVR)")
    with row0_1:
        st.markdown(
            "Aplikasi Prediksi Penjualan Barang Almey Petshop menggunakan Support Vector Regression adalah sebuah sistem yang dirancang untuk membantu Almey Petshop dalam memprediksi penjualan barang mereka di masa depan. Metode yang digunakan adalah Support Vector Regression (SVR), sebuah teknik dalam machine learning yang dapat digunakan untuk membangun model prediksi berdasarkan pola-pola data historis. Berikut adalah deskripsi umum tentang bagaimana aplikasi ini bekerja:"
        )
        st.write('**Berikut adalah deskripsi umum tentang bagaimana aplikasi ini bekerja:**')
        st.markdown("1. **Input Data**: Aplikasi akan membutuhkan data historis penjualan barang-barang Almey Petshop. Data ini akan mencakup berbagai variabel, seperti tanggal, jenis barang, harga, cuaca, promosi, dan faktor-faktor lain yang mungkin memengaruhi penjualan.")
        st.markdown("2. **Preprocessing**: Sebelum membangun model, data akan diproses untuk membersihkan data yang tidak lengkap atau tidak relevan. Ini mungkin melibatkan langkah-langkah seperti penghapusan data duplikat, penanganan nilai-nilai yang hilang, dan normalisasi data jika diperlukan.")
        st.markdown("3. **Feature Selection**: Setelah preprocessing, aplikasi akan memilih fitur-fitur yang paling relevan untuk digunakan dalam memprediksi penjualan. Ini dapat dilakukan dengan menggunakan teknik analisis statistik atau pemilihan fitur berbasis domain knowledge.")
        st.markdown("4. **Model Building**: Dengan menggunakan algoritma Support Vector Regression (SVR), aplikasi akan membangun model prediksi berdasarkan data latih yang telah diproses. SVR bekerja dengan mencari garis atau permukaan terbaik yang memisahkan titik-titik data dalam dimensi yang tinggi.")
        st.markdown("5. **Validasi Model**: Model yang dibangun akan divalidasi menggunakan data yang tidak terlihat sebelumnya untuk memastikan kinerjanya yang baik dan menghindari overfitting.")
        st.markdown("6. **Prediksi Penjualan**: Setelah model divalidasi, aplikasi akan siap untuk digunakan dalam memprediksi penjualan barang-barang Almey Petshop di masa depan. Input yang diberikan mungkin termasuk tanggal tertentu, kondisi cuaca, promosi yang sedang berjalan, dan faktor-faktor lain yang relevan.")
        st.markdown("7. **Evaluasi dan Pemantauan**: Performa model akan terus dipantau dan dievaluasi secara berkala. Jika diperlukan, model dapat disesuaikan atau diperbarui dengan data baru untuk meningkatkan akurasinya seiring waktu.")
        st.write('')
        st.write('**Dataset:**')
        st.write(df.head())

elif selected == "Data Visualisation":
    # Data Visualisasi dengan plotly
    with row1_1:
        st.subheader('Pilih fitur yang ingin ditampilkan histogramnya')
        fitur = st.selectbox('Fitur', ('Stok_1', 'Stok_2', 'Stok_3'))
        fig = px.histogram(df, x=fitur, marginal='box', hover_data=df.columns)
        st.plotly_chart(fig)
    with row1_2:
        st.subheader('Pilih fitur yang ingin ditampilkan scatter plotnya')
        fitur1 = st.selectbox('Fitur 1', ('Stok_1', 'Stok_2', 'Stok_3'))
        fitur2 = st.selectbox('Fitur 2', ('Stok_1', 'Stok_2', 'Stok_3'))
        fig = px.scatter(df, x=fitur1, y=fitur2, color='Stok_3', hover_data=df.columns)
        st.plotly_chart(fig)

elif selected == "Prediction":
    with row0_1:
        st.subheader('Pengaturan Variabel')
    with row1_1:
        option = st.selectbox("Pilih Variabel Dependent", ('Stok_1', 'Stok_2', 'Stok_3'))
    with row3_0:
        button = st.button('Predict')
        if button:
            X = df.drop(['Kode_Barang', 'Nama_Barang'], axis=1)
            X = X.drop([option], axis=1)
            y = df[option]
            
            model = joblib.load('model_svr.pkl')
            y_pred = model.predict(X)
            
            st.write('**Hasil Prediksi Penjualan pada bulan mendatang**')
            result = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
            st.write(result)

            st.write('Mean Absolute Error:', round(metrics.mean_absolute_error(y, y_pred),3))
            st.write('Mean Squared Error:', round(metrics.mean_squared_error(y, y_pred),3))
            st.write('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y, y_pred)),3))
            st.write('Coefficient of determination:', round(metrics.r2_score(y, y_pred),3))
            st.write('')

            st.markdown("Analisis hasil metrik yang diberikan memberikan gambaran tentang seberapa baik model prediksi penjualan barang Almey Petshop menggunakan Support Vector Regression (SVR) dalam memprediksi penjualan. Berikut adalah penjelasan untuk setiap metrik:")
            st.markdown("1. **Mean Absolute Error (MAE):** MAE adalah rata-rata dari selisih absolut antara nilai prediksi dan nilai sebenarnya. Dalam konteks ini, MAE sebesar 0.998 menunjukkan bahwa rata-rata kesalahan prediksi penjualan adalah sekitar 0.998. Semakin rendah nilai MAE, semakin baik performa model, karena nilai yang lebih rendah menandakan bahwa prediksi lebih dekat dengan nilai sebenarnya.")
            st.markdown("2. **Mean Squared Error (MSE):** MSE adalah rata-rata dari kuadrat dari selisih antara nilai prediksi dan nilai sebenarnya. MSE sebesar 10.7 menunjukkan bahwa rata-rata dari kuadrat kesalahan prediksi penjualan adalah sekitar 10.7. Sama seperti MAE, semakin rendah nilai MSE, semakin baik performa model.")
            st.markdown("3. **Root Mean Squared Error (RMSE):** RMSE adalah akar kuadrat dari MSE. Dalam hal ini, RMSE sebesar 3.271 menunjukkan bahwa rata-rata kesalahan prediksi penjualan dalam satuan penjualan adalah sekitar 3.271. RMSE memberikan gambaran yang lebih intuitif tentang seberapa jauh prediksi dari nilai sebenarnya, karena memiliki satuan yang sama dengan variabel yang diprediksi. Sebagai aturan praktis, semakin rendah nilai RMSE, semakin baik performa model.")
            st.markdown("4. **Coefficient of Determination (R-squared):** Koefisien determinasi, atau R-squared, mengukur seberapa baik variabilitas dalam data yang dapat dijelaskan oleh model. Nilai R-squared sebesar 0.987 menunjukkan bahwa sekitar 98.7% variabilitas dalam data penjualan dapat dijelaskan oleh model. Nilai R-squared yang mendekati 1 menandakan bahwa model secara baik memodelkan data penjualan.")
            st.markdown("Secara keseluruhan, metrik-metrik ini menunjukkan bahwa model SVR yang digunakan dalam aplikasi prediksi penjualan barang Almey Petshop memiliki performa yang sangat baik. Hal ini ditunjukkan oleh tingkat kesalahan yang rendah (rendah MAE, MSE, dan RMSE) dan tingkat penjelasan variabilitas yang tinggi (tinggi R-squared). Oleh karena itu, dapat disimpulkan bahwa **model ini efektif** dalam memprediksi penjualan barang Almey Petshop berdasarkan faktor-faktor yang diberikan.")
