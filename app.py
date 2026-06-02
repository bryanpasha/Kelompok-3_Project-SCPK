import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

st.set_page_config(page_title="SPK Retensi Karyawan (Fuzzy Mamdani)", layout="wide")

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("File style.css tidak ditemukan. Menggunakan tema bawaan Streamlit.")

local_css("styles.css")

@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    return df

df = load_data()

st.sidebar.title("Navigasi SPK")
page = st.sidebar.radio("Pilih Halaman:", 
                        ["1. Halaman Data", 
                         "2. Pengaturan Parameter", 
                         "3. Hitung & Hasil Perangkingan"])

if 'batas_gaji' not in st.session_state:
    st.session_state.batas_gaji = 5000
if 'batas_jarak' not in st.session_state:
    st.session_state.batas_jarak = 15
if 'batas_tahun' not in st.session_state:
    st.session_state.batas_tahun = 3
if 'departemen' not in st.session_state:
    st.session_state.departemen = "Semua Departemen"

if page == "1. Halaman Data":
    st.title("Dataset Karyawan (HR Attrition)")
    st.write("Menampilkan dataset IBM HR Analytics Employee Attrition. Data ini digunakan untuk menganalisis tingkat risiko resign karyawan menggunakan metode Fuzzy Mamdani.")
    
    st.dataframe(df)
    st.info(f"Total Data: {df.shape[0]} Baris, {df.shape[1]} Kolom.")

elif page == "2. Pengaturan Parameter":
    st.title("Pengaturan Parameter Fuzzy & Filter Alternatif")
    st.write("Atur parameter fungsi keanggotaan (Membership Function) dan pilih kelompok alternatif yang ingin dihitung.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.batas_gaji = st.slider(
            "1. Batas Maksimal 'Gaji Rendah' (USD)", 
            min_value=1000, max_value=10000, value=st.session_state.batas_gaji, step=500,
            help="Nilai ini menggeser kurva fuzzy untuk variabel Gaji."
        )
        st.session_state.batas_jarak = st.slider(
            "2. Batas Minimal 'Jarak Jauh' (Mil)", 
            min_value=5, max_value=29, value=st.session_state.batas_jarak, step=1,
            help="Nilai ini menggeser kurva fuzzy untuk variabel Jarak dari Rumah."
        )

    with col2:
        st.session_state.batas_tahun = st.slider(
            "3. Batas Maksimal Kategori 'Karyawan Baru' (Tahun)", 
            min_value=1, max_value=10, value=st.session_state.batas_tahun, step=1,
            help="Menentukan definisi 'Karyawan Baru' pada evaluasi Fuzzy."
        )
        dept_list = ["Semua Departemen"] + list(df['Department'].unique())
        st.session_state.departemen = st.selectbox(
            "4. Pilih Departemen (Alternatif Data)", 
            options=dept_list, index=dept_list.index(st.session_state.departemen)
        )
        
    st.success("Semua parameter berhasil disimpan! Lanjut ke Halaman 3 untuk eksekusi.")

elif page == "3. Hitung & Hasil Perangkingan":
    st.title("Eksekusi SPK & Hasil Perangkingan")
    
    if st.session_state.departemen != "Semua Departemen":
        df_filtered = df[df['Department'] == st.session_state.departemen].copy()
    else:
        df_filtered = df.copy()

    st.write(f"Menyiapkan **{len(df_filtered)}** data dari **{st.session_state.departemen}** untuk diproses.")

    if st.button("Mulai Perhitungan Fuzzy (Fuzzifikasi & Defuzzifikasi)"):
        with st.spinner("Sistem sedang memproses algoritma Fuzzy Mamdani..."):
            
            # 5 Kriteria Antecedent 
            gaji = ctrl.Antecedent(np.arange(0, 20001, 1), 'gaji')
            jarak = ctrl.Antecedent(np.arange(0, 30, 1), 'jarak')
            kepuasan = ctrl.Antecedent(np.arange(1, 5, 1), 'kepuasan') 
            lama_kerja = ctrl.Antecedent(np.arange(0, 41, 1), 'lama_kerja')
            keterlibatan = ctrl.Antecedent(np.arange(1, 5, 1), 'keterlibatan') 

            risiko = ctrl.Consequent(np.arange(0, 101, 1), 'risiko') 

            # Himpunan Fuzzy Gaji
            bg = st.session_state.batas_gaji
            gaji['rendah'] = fuzz.trapmf(gaji.universe, [0, 0, bg-1000, bg+1000])
            gaji['sedang'] = fuzz.trimf(gaji.universe, [bg-500, 10000, 15000])
            gaji['tinggi'] = fuzz.trapmf(gaji.universe, [12000, 18000, 20000, 20000])

            # Himpunan Fuzzy Jarak
            bj = st.session_state.batas_jarak
            jarak['dekat'] = fuzz.trimf(jarak.universe, [0, 0, bj-2])
            jarak['sedang'] = fuzz.trimf(jarak.universe, [3, bj-1, bj+5])
            jarak['jauh'] = fuzz.trapmf(jarak.universe, [bj-2, bj+3, 29, 29])

            # Himpunan Fuzzy Kepuasan & Keterlibatan
            kepuasan.automf(names=['rendah', 'sedang', 'tinggi'])
            keterlibatan.automf(names=['rendah', 'sedang', 'tinggi'])

            # Himpunan Fuzzy Lama Kerja
            bt = st.session_state.batas_tahun
            lama_kerja['baru'] = fuzz.trimf(lama_kerja.universe, [0, 0, bt])
            lama_kerja['menengah'] = fuzz.trimf(lama_kerja.universe, [bt-1, 10, 15])
            lama_kerja['lama'] = fuzz.trapmf(lama_kerja.universe, [10, 20, 40, 40])

            # Himpunan Fuzzy Output (Risiko)
            risiko['rendah'] = fuzz.trapmf(risiko.universe, [0, 0, 25, 50])
            risiko['waspada'] = fuzz.trimf(risiko.universe, [25, 50, 75])
            risiko['tinggi'] = fuzz.trapmf(risiko.universe, [50, 75, 100, 100])

            # Rule Base dengan 5 Kriteria
            rule1 = ctrl.Rule(gaji['rendah'] & jarak['jauh'] & kepuasan['rendah'], risiko['tinggi'])
            rule2 = ctrl.Rule(gaji['tinggi'] & kepuasan['tinggi'] & keterlibatan['tinggi'], risiko['rendah']) 
            rule3 = ctrl.Rule(lama_kerja['baru'] & gaji['rendah'], risiko['tinggi'])
            rule4 = ctrl.Rule(kepuasan['sedang'] & jarak['sedang'], risiko['waspada'])
            rule5 = ctrl.Rule(jarak['dekat'] & kepuasan['tinggi'], risiko['rendah'])
            rule6 = ctrl.Rule(gaji['sedang'] | lama_kerja['menengah'], risiko['waspada'])
            rule7 = ctrl.Rule(keterlibatan['rendah'] & kepuasan['rendah'], risiko['tinggi']) 

            risiko_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
            risiko_sim = ctrl.ControlSystemSimulation(risiko_ctrl)

            hasil_risiko = []
            
            for index, row in df_filtered.iterrows():
                try:
                    risiko_sim.input['gaji'] = int(row['MonthlyIncome'])
                    risiko_sim.input['jarak'] = int(row['DistanceFromHome'])
                    risiko_sim.input['kepuasan'] = int(row['JobSatisfaction'])
                    risiko_sim.input['lama_kerja'] = int(row['YearsAtCompany'])
                    risiko_sim.input['keterlibatan'] = int(row['JobInvolvement']) 
                    
                    risiko_sim.compute()
                    skor = round(risiko_sim.output['risiko'], 2)
                except:
                    skor = 50.0
                    
                hasil_risiko.append(skor)
            
            df_filtered['Skor_Risiko'] = hasil_risiko
            
            # Tabel Proses dengan 5 Input
            df_proses = df_filtered[['EmployeeNumber', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany', 'JobSatisfaction', 'JobInvolvement', 'Skor_Risiko']]
            
            df_hasil = df_filtered[['EmployeeNumber', 'Department', 'JobRole', 'Skor_Risiko']]
            df_hasil = df_hasil.sort_values(by='Skor_Risiko', ascending=False).reset_index(drop=True)
            
            st.success("Perhitungan Selesai!")
            
            st.subheader("Tabel Proses SPK (Defuzzifikasi)")
            st.write("Tabel ini menunjukkan *Crisp Input* (5 Kriteria) yang dieksekusi menjadi *Crisp Output*.")
            st.dataframe(df_proses)

            st.subheader("Tabel Perangkingan Prioritas")
            st.write("Warna Merah = Kritis (Skor $\ge$ 70), Kuning = Waspada (40-69), Hijau = Aman (< 40).")
            
            def highlight_row_risk(row):
                skor = row['Skor_Risiko']
                if skor >= 70:
                    return ['background-color: #ffcccc; color: black'] * len(row)
                elif skor >= 40:
                    return ['background-color: #ffffcc; color: black'] * len(row)
                else:
                    return ['background-color: #ccffcc; color: black'] * len(row)
            
            st.dataframe(df_hasil.style.apply(highlight_row_risk, axis=1))

            st.subheader("Visualisasi Kurva Defuzzifikasi (Top Peringkat 1)")
            st.write("Grafik ini menampilkan batas fungsi keanggotaan output serta posisi *Crisp Output* untuk karyawan dengan risiko resign tertinggi.")
            
            top_skor = df_hasil.loc[0, 'Skor_Risiko']
            top_emp = df_hasil.loc[0, 'EmployeeNumber']
            
            fig, ax = plt.subplots(figsize=(8, 4))
            
            x_risiko = np.arange(0, 101, 1)
            y_rendah = fuzz.trapmf(x_risiko, [0, 0, 25, 50])
            y_waspada = fuzz.trimf(x_risiko, [25, 50, 75])
            y_tinggi = fuzz.trapmf(x_risiko, [50, 75, 100, 100])
            
            ax.plot(x_risiko, y_rendah, 'g', linewidth=2, label='Rendah')
            ax.plot(x_risiko, y_waspada, 'y', linewidth=2, label='Waspada')
            ax.plot(x_risiko, y_tinggi, 'r', linewidth=2, label='Tinggi')
            
            ax.axvline(x=top_skor, color='k', linestyle='--', linewidth=2, label=f'Skor Karyawan #{top_emp} ({top_skor:.2f})')
            ax.fill_between(x_risiko, 0, y_tinggi, where=(x_risiko >= 50), facecolor='red', alpha=0.1)
            
            ax.set_title("Proses Defuzzifikasi Fuzzy Mamdani (Output Risiko)")
            ax.set_xlabel("Skala Risiko Resign (0 - 100)")
            ax.set_ylabel("Derajat Keanggotaan (μ)")
            ax.legend()
            
            st.pyplot(fig)