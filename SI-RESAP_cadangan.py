# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:53:51 2022

@author: siddhardhan
"""
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt


# loading the saved models
risiko_model = pickle.load(open("Risk_Modelling.sav", "rb")) 

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('PRESISI AKUN (Prediksi Klasifikasi Risiko Akun)',
                          
                          ['Ekpsloratory Data Analysis',
                          'Prediksi Risiko Akun'],
                          icons=['activity','play'],
                          menu_icon="cast",
                          default_index=0
                          )

# Mutasi Prediction Page
if (selected == 'Prediksi Risiko Akun'):
    
    # page title
    # st.title('Prediksi Mutasi Pegawai Dengan Machine Learning')
    st.markdown("<h3 style='text-align: center; color: Black;'>Prediksi Risiko Atas Akun dengan Algoritma XGBoost</h3>", unsafe_allow_html=True)


    Entitas = st.selectbox('Entitas',['BS','KB','S'])

    if Entitas == 'BS':
        Entitas = 0
    elif Entitas =="KB":
        Entitas = 1
    else :
        Entitas = 2

    Akun = st.selectbox('Akun Terkait',['Akum. Amortisasi Aset Tak Berwujud', 
                                'Akumulasi Penyusutan', 'Aset Lain-lain', 'Aset Lainnya Aset Lain-Lain', 
                                'Aset Tak Berwujud', 'Aset Tetap Akumulasi Penyusutan', 'Aset Tetap Aset Tetap Lainnya', 
                                'Aset Tetap Gedung dan Bangunan', 'Aset Tetap Jalan, Irigasi, dan Jaringan', 
                                'Aset Tetap Konstruksi dalam Pengerjaan', 'Aset Tetap Lainnya', 
                                'Aset Tetap Peralatan dan Mesin', 'Aset Tetap Tanah', 'Aset Tidak Berwujud', 
                                'Bagi Hasil Pajak', 'Bagi Hasil Pendapatan Lainnya', 'Bagi Hasil Retribusi', 
                                'Bantuan Keuangan', 'Bantuan Keuangan Lainnya', 'Bantuan Sosial', 
                                'Beban Bantuan Sosial', 'Beban Hibah', 'Beban Jasa', 'Beban Lain-Lain', 
                                'Beban Lain-lain', 'Beban Luar Biasa', 'Beban Pegawai', 'Beban Pemeliharaan', 
                                'Beban Penyisihan Piutang', 'Beban Penyusutan', 'Beban Penyusutan dan Amortisasi', 
                                'Beban Perjalanan Dinas', 'Beban Subsidi', 'Beban Transfer', 
                                'Belanja Aset Tetap Lainnya', 'Belanja Barang dan Jasa', 'Belanja Bunga', 
                                'Belanja Dibayar Dimuka', 'Belanja Gedung dan Bangunan', 'Belanja Hibah', 
                                'Belanja Jalan, Irigasi dan Jaringan', 'Belanja Pegawai', 'Belanja Peralatan dan Mesin', 
                                'Belanja Subsidi', 'Belanja Tak Terduga', 'Belanja Tanah', 'Dana Otonomi Khusus', 
                                'Dana Penyesuaian', 'Ekuitas Dana', 'Gedung dan Bangunan', 'Hibah', 
                                'Investasi Permanen (Penyertaan Modal Pemerintah Daerah)', 
                                'Investasi Permanen Penyertaan Modal Pemerintah Daerah', 'Jalan Irigasi dan jaringan', 
                                'Jalan, Irigasi dan Jaringan', 'Kas Di Bendahara JKN Puskesmas', 'Kas Lainnya', 
                                'Kas di BLUD', 'Kas di Bendahara BOS', 'Kas di Bendahara Dana BOS', 
                                'Kas di Bendahara FKTP', 'Kas di Bendahara Penerimaan', 'Kas di Bendahara Pengeluaran', 
                                'Kas di Kas Daerah', 'Kemitraan dengan Pihak Ketiga', 'Konstruksi Dalam Pengerjaan', 
                                'Konstruksi dalam Pengerjaan', 'Lain-Lain Pendapatan yang Sah', 
                                'Lain-lain PAD yang Sah', 'Lain-lain Pendapatan yang Sah Pendapatan Hibah', 
                                'PAD - Hasil Kekayaan Daerah yang Dipisahkan', 'PAD - Pendapatan Pajak Daerah', 
                                'PAD - Pendapatan Retribusi Daerah', 'Pendapatan Bagi Hasil Lainnya', 
                                'Pendapatan Diterima Dimuka', 'Pendapatan Hibah', 'Pendapatan Luar Biasa', 
                                'Pendapatan Transfer', 'Penerimaan Pembiayaan Penggunaan SiLPA', 
                                'Pengeluran Pembiayaan Penyertaan Modal Pemerintah Daerah', 'Penggunaan Silpa', 
                                'Pengunaan SILPA', 'Penyertaan Modal Pemerintah Daerah', 'Penyisihan Piutang', 
                                'Penyisihan Piutang Tak Tertagih', 'Peralatan dan Mesin', 'Persediaan', 
                                'Piutang Lainnya', 'Piutang PAD Lainnya', 'Piutang Pajak', 'Piutang Pajak Daerah', 
                                'Piutang Retribusi', 'Piutang Retribusi Daerah', 'Piutang Transfer Antar Daerah', 
                                'Surplus/Defisit dari Kegiatan Non Operasional Lainnya', 'Tanah', 
                                'Transfer Pemerintah Provinsi BHP Pendapatan Bagi Hasil Pajak Daerah **)', 
                                'Transfer Pusat Dana Perimbangan Dana Bagi Hasil Pajak', 
                                'Transfer ke Kab/Kota Bantuan Keuangan **)', 'Utang Jangka Pendek Lainnya', 
                                'Utang Pegawai', 'Utang Perhitungan Fihak Ketiga (PFK)'])
    
    if Akun == 'Akum. Amortisasi Aset Tak Berwujud':
        Akun = 0
    elif Akun == 'Akumulasi Penyusutan':
        Akun = 1
    elif Akun == 'Aset Lain-lain':
         Akun = 2
    elif Akun == 'Aset Lainnya Aset Lain-Lain':
         Akun = 3 
    elif Akun == 'Aset Tak Berwujud':
         Akun = 4      
    elif Akun == 'Aset Tetap Akumulasi Penyusutan':
         Akun = 5 
    elif Akun == 'Aset Tetap Aset Tetap Lainnya':
        Akun = 6 
    elif Akun == 'Aset Tetap Gedung dan Bangunan':
         Akun = 7 
    elif Akun == 'Aset Tetap Jalan, Irigasi, dan Jaringan':
        Akun = 8 
    elif Akun == 'Aset Tetap Konstruksi dalam Pengerjaan':
        Akun = 9 
    elif Akun == 'Aset Tetap Lainnya':
        Akun = 10 
    elif Akun == 'Aset Tetap Peralatan dan Mesin':
        Akun = 11 
    elif Akun == 'Aset Tetap Tanah':
        Akun = 12 
    elif Akun == 'Aset Tidak Berwujud':
        Akun = 13 
    elif Akun == 'Bagi Hasil Pajak':
        Akun = 14 
    elif Akun == 'Bagi Hasil Pendapatan Lainnya':
         Akun = 15 
    elif Akun == 'Bagi Hasil Retribusi':
        Akun = 16 
    elif Akun == 'Bantuan Keuangan':
        Akun = 17 
    elif Akun == 'Bantuan Keuangan Lainnya':
        Akun = 18 
    elif Akun == 'Bantuan Sosial':
        Akun = 19 
    elif Akun == 'Beban Bantuan Sosial':
        Akun = 20 
    elif Akun == 'Beban Hibah':
        Akun = 21 
    elif Akun == 'Beban Jasa':
        Akun = 22 
    elif Akun == 'Beban Lain-Lain':
        Akun = 23 
    elif Akun == 'Beban Lain-lain':
        Akun = 24 
    elif Akun == 'Beban Luar Biasa':
        Akun = 25 
    elif Akun == 'Beban Pegawai':
        Akun = 26 
    elif Akun == 'Beban Pemeliharaan':
     Akun = 27 
    elif Akun == 'Beban Penyisihan Piutang':
        Akun = 28 
    elif Akun == 'Beban Penyusutan':
        Akun = 29 
    elif Akun == 'Beban Penyusutan dan Amortisasi':
        Akun = 30 
    elif Akun == 'Beban Perjalanan Dinas':
        Akun = 31 
    elif Akun == 'Beban Subsidi':
        Akun = 32 
    elif Akun == 'Beban Transfer':
        Akun = 33 
    elif Akun == 'Belanja Aset Tetap Lainnya':
        Akun = 34 
    elif Akun == 'Belanja Barang dan Jasa':
        Akun = 35 
    elif Akun == 'Belanja Bunga':
        Akun = 36 
    elif Akun == 'Belanja Dibayar Dimuka':
        Akun = 37 
    elif Akun == 'Belanja Gedung dan Bangunan':
        Akun = 38
    elif Akun == 'Belanja Hibah':
        Akun = 39
    elif Akun == 'Belanja Jalan, Irigasi dan Jaringan':
        Akun = 40
    elif Akun == 'Belanja Pegawai':
        Akun = 41
    elif Akun == 'Belanja Peralatan dan Mesin':
        Akun = 42
    elif Akun == 'Belanja Subsidi':
        Akun = 43
    elif Akun == 'Belanja Tak Terduga':
        Akun = 44
    elif Akun == 'Belanja Tanah':
        Akun = 45
    elif Akun == 'Dana Otonomi Khusus':
        Akun = 46
    elif Akun == 'Dana Penyesuaian':
        Akun = 47
    elif Akun ==  'Ekuitas Dana':
        Akun = 48
    elif Akun == 'Gedung dan Bangunan':
        Akun = 49
    elif Akun == 'Hibah':
        Akun = 50
    elif Akun == 'Investasi Permanen (Penyertaan Modal Pemerintah Daerah)':
        Akun = 51
    elif Akun == 'Investasi Permanen Penyertaan Modal Pemerintah Daerah':
        Akun = 52
    elif Akun == 'Jalan Irigasi dan jaringan':
        Akun = 53
    elif Akun == 'Jalan, Irigasi dan Jaringan':
        Akun = 54
    elif Akun == 'Kas Di Bendahara JKN Puskesmas':
        Akun = 55
    elif Akun == 'Kas Lainnya':
        Akun = 56
    elif Akun == 'Kas di BLUD':
        Akun = 57
    elif Akun == 'Kas di Bendahara BOS':
        Akun = 58
    elif Akun == 'Kas di Bendahara Dana BOS':
        Akun = 59
    elif Akun == 'Kas di Bendahara FKTP':
        Akun = 60
    elif Akun == 'Kas di Bendahara Penerimaan':
        Akun = 61
    elif Akun == 'Kas di Bendahara Pengeluaran':
        Akun = 62
    elif Akun == 'Kas di Kas Daerah':
        Akun = 63 
    elif Akun == 'Kemitraan dengan Pihak Ketiga':
        Akun = 64      
    elif Akun == 'Konstruksi Dalam Pengerjaan':
        Akun = 65 
    elif Akun == 'Konstruksi dalam Pengerjaan':
        Akun = 66 
    elif Akun == 'Lain-Lain Pendapatan yang Sah':
        Akun = 67 
    elif Akun == 'Lain-lain PAD yang Sah':
        Akun = 68 
    elif Akun == 'Lain-lain Pendapatan yang Sah Pendapatan Hibah':
        Akun = 69 
    elif Akun == 'PAD - Hasil Kekayaan Daerah yang Dipisahkan':
        Akun = 70 
    elif Akun == 'PAD - Pendapatan Pajak Daerah':
        Akun = 71
    elif Akun == 'PAD - Pendapatan Retribusi Daerah':
        Akun = 72
    elif Akun == 'Pendapatan Bagi Hasil Lainnya':
        Akun = 73 
    elif Akun == 'Pendapatan Diterima Dimuka':
        Akun = 74      
    elif Akun == 'Pendapatan Hibah':
        Akun = 75 
    elif Akun == 'Pendapatan Luar Biasa':
        Akun = 76 
    elif Akun == 'Pendapatan Transfer':
        Akun = 77 
    elif Akun == 'Penerimaan Pembiayaan Penggunaan SiLPA':
        Akun = 78 
    elif Akun == 'Pengeluran Pembiayaan Penyertaan Modal Pemerintah Daerah':
        Akun = 79 
    elif Akun == 'Penggunaan Silpa':
        Akun = 80
    elif Akun == 'Penyertaan Modal Pemerintah Daerah':
        Akun = 81
    elif Akun == 'Pengunaan SILPA':
        Akun = 82
    elif Akun == 'Penyisihan Piutang':
        Akun = 83 
    elif Akun == 'Penyisihan Piutang Tak Tertagih':
        Akun = 84      
    elif Akun == 'Peralatan dan Mesin':
        Akun = 85 
    elif Akun == 'Persediaan':
        Akun = 86 
    elif Akun == 'Piutang Lainnya':
        Akun = 87 
    elif Akun == 'Piutang PAD Lainnya':
        Akun = 88 
    elif Akun == 'Piutang Pajak':
        Akun = 89 
    elif Akun == 'Piutang Pajak Daerah':
        Akun = 90
    elif Akun == 'Piutang Retribusi':
        Akun = 91
    elif Akun == 'Piutang Retribusi Daerah':
        Akun = 92
    elif Akun == 'Piutang Transfer Antar Daerah':
        Akun = 93 
    elif Akun == 'Surplus/Defisit dari Kegiatan Non Operasional Lainnya':
        Akun = 94      
    elif Akun == 'Tanah':
        Akun = 95 
    elif Akun == 'Transfer Pemerintah Provinsi BHP Pendapatan Bagi Hasil Pajak Daerah **)':
        Akun = 96 
    elif Akun == 'Transfer Pusat Dana Perimbangan Dana Bagi Hasil Pajak':
        Akun = 97 
    elif Akun == 'Transfer ke Kab/Kota Bantuan Keuangan **)':
        Akun = 98 
    elif Akun == 'Utang Jangka Pendek Lainnya':
        Akun = 99 
    elif Akun == 'Utang Pegawai':
        Akun = 100
    elif Akun == 'Utang Perhitungan Fihak Ketiga (PFK)':
        Akun = 101
    else :
        Akun = 102

    if Akun == "Akum. Amortisasi Aset Tak Berwujud":
        Akun = 0
    elif Akun == "Akumulasi Penyusutan":
        Akun = 1
    elif Akun == 'Aset Lain-lain':
        Akun = 2
    else :
        Akun = 101

    IR = st.selectbox('Inherent Risk', ['Low','Medium'])
        
    if IR == "Low" :
        IR = 0
    else :
        IR = 1
    
    CR = st.selectbox('Control Risk', ["High",'Low','Medium'])
        
    if CR == "Low" :
        CR = 0
    elif CR == "Medium":
        CR = 1
    else :
        CR = 2

    Temuan_Tahun_Sebelumnya = st.selectbox('Temuan Sebelumnya', ['Ada','Tidak'])

    if Temuan_Tahun_Sebelumnya == "Ada" :
        Temuan_Tahun_Sebelumnya = 0
    else :    
        Temuan_Tahun_Sebelumnya = 1

    Akun_Pengecualian = st.selectbox('Akun Pengecualian', ['Tidak','Ya'])

    if Akun_Pengecualian == "Tidak":
        Akun_Pengecualian = 0
    else :
        Akun_Pengecualian = 1
    
    Saldo_2022 =  st.text_input('Saldo Tahun Anggaran')

    Saldo_2021 = st.text_input('Saldo Tahun Anggaran Tahun Sebelumnya')

    Pertumbuhan_Saldo = st.text_input('Persentase Pertumbuhan Saldo')

    # code for Prediction
    prediksi = ''
    
    # creating a button for Prediction
    
    if st.button('Submit'):
        prediksi_risiko = risiko_model.predict([[
            Entitas,
            Akun, 
            IR,
            CR,
            Temuan_Tahun_Sebelumnya,
            Akun_Pengecualian,
            Saldo_2022,
            Pertumbuhan_Saldo]])
        

        if (prediksi_risiko[0] == 0):
            prediksi = f'Akun Pada Entitas Tersebut Memiliki Risiko High'
        elif (prediksi_risiko[0] == 2):
            prediksi = f'Akun Pada Entitas Tersebut Memiliki Risiko Low'
        else:
            prediksi = f'Akun Pada Entitas Tersebut Memiliki Risiko Medium'
        
    st.success(prediksi)

# Heart Disease Prediction Page
if (selected == 'Ekpsloratory Data Analysis'):

    st.markdown("<h2 style='text-align: center; color: Black;'>Eksploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)

    # Membaca data dari file CSV (misalnya, data.csv)
    @st.cache_resource
    def load_data():
        data = pd.read_csv("Data.csv", sep=";", on_bad_lines="skip")
        data.drop(['% Pertumbuhan','AR'], axis= 1, inplace= True)
        data['Pertumbuhan Saldo'] = np.round(np.where(data['Saldo 2021'] == 0, 0, 
                                   (data['Saldo 2022'] - data['Saldo 2021'])/data['Saldo 2021'] * 100 ), 
                                   decimals= 2)
        data['Temuan Tahun Sebelumnya'] =  data['Temuan Tahun Sebelumnya'].replace(['Ya','ada','tidak'], 
                                                                       ['Ada','Ada','Tidak'], regex= True)
        data['Akun Pengecualian'] =  data['Akun Pengecualian'].replace(['tidak'], 'Tidak', regex= True)
        data['CR'] = data['CR'].str.replace(" ","")
        data['Akun'] = data['Akun'].str.rstrip() # remove space behind and in front of
        data['Akun'] = data['Akun'].str.replace('Belanja Barang  dan Jasa', 'Belanja Barang dan Jasa', regex = True)
        data['Akun'] = data['Akun'].str.replace('Belanja gedung dan Bangunan', 'Belanja Gedung dan Bangunan', regex = True)
        data['Akun'] = data['Akun'].str.replace('Belanja Aset Tetap lainnya', 'Belanja Aset Tetap Lainnya', regex = True)
        data['Akun'] = data['Akun'].replace(["Pendapatan Pajak Daerah",'PAD Pendapatan Pajak Daerah'], ['PAD - Pendapatan Pajak Daerah',
                                'PAD - Pendapatan Pajak Daerah'])
        data['Akun'] = data['Akun'].replace(['PAD Pendapatan Retribusi Daerah','PAD - Pendapatan Rettribusi Daerah',
                                'Pendapatan Retribusi Daerah'], 
                                    ['PAD - Pendapatan Retribusi Daerah','PAD - Pendapatan Retribusi Daerah',
                                    'PAD - Pendapatan Retribusi Daerah'])
        data['Akun'] = data['Akun'].replace(['Pendapatan Hasil Pengelolaan Kekayaan Daerah yang Dipisahkan',
                                'PAD Pendapatan Hasil Pengelolaan Kekayaan Daerah yang Dipisahkan'], 
                                'PAD - Hasil Kekayaan Daerah yang Dipisahkan')

        return data

    # Fungsi untuk menampilkan statistik deskriptif
    def show_descriptive_stats(data):
        st.header('Statistik Deskriptif')
        st.write(data.describe())
    
    # Fungsi untuk menampilkan pie chart
    def show_pie_chart(data, column):
        st.header(f'Pie Chart {column}')
        # Menghitung jumlah kategori pada kolom tertentu
        category_counts = data[column].value_counts()

        # Membuat pie chart
        fig, ax = plt.subplots()
        ax.pie(category_counts, labels=category_counts.index, 
               autopct='%1.1f%%', 
               startangle=90)
        ax.axis('equal')  
        # Memastikan pie chart berbentuk lingkaran
        fig.set_size_inches(3, 3)
        # Menampilkan pie chart menggunakan st.pyplot
        st.pyplot(fig)

    def main():
        # st.title('Exploratory Data Analysis (EDA) dengan Streamlit')

        data = load_data()
        # Menampilkan tabel data
        st.header('Data')
        st.write(data)

        show_descriptive_stats(data)

        # Menampilkan pie chart
        selected_column = st.selectbox('Pilih kolom untuk melihat distribusi:', data.select_dtypes(include='object').columns)
        show_pie_chart(data, selected_column)

    if __name__ == '__main__':
        main()
