import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re

st.set_page_config(page_title="Prediksi Cuaca BMKG Bertingkat", layout="wide")
# ========== Load daftar wilayah dari base.csv ==========
@st.cache_data
def load_data_wilayah():
    """
    Memuat dan memproses data wilayah dari GitHub sekali saja.
    Menghitung level administrasi dan membersihkan nama untuk tampilan.
    """
    url = "https://raw.githubusercontent.com/kodewilayah/permendagri-72-2019/main/dist/base.csv"
    df = pd.read_csv(url, header=None, names=["id", "nama"], dtype=str)
    
    # 0: Provinsi, 1: Kab/Kota, 2: Kecamatan, 3: Kelurahan/Desa
    df['level'] = df['id'].str.count(r'\.')
    
    def clean_name(name):
        return re.sub(r'^(KAB\. |KOTA |KEC\. |DESA |KEL\. )', '', name).title()
        
    df['nama_bersih'] = df['nama'].apply(lambda x: clean_name(x.split(',')[0]))
    df = df.set_index('id')
    return df

df_wilayah = load_data_wilayah()


# ========== Ambil data cuaca dari BMKG ==========
@st.cache_data
def get_bmkg_data(kode_wilayah_desa):
    """Mengambil data prakiraan cuaca dari API BMKG menggunakan kode DESA/KELURAHAN."""
    kode_api = kode_wilayah_desa
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={kode_api}"
    
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        # Mengembalikan error agar bisa ditampilkan di UI
        return f"Error: Gagal menghubungi server BMKG atau data tidak ditemukan. Pesan: {e}"

    j = resp.json()
    data_list = j.get("data", [])
    if not data_list:
        return "Error: Tidak ada data cuaca yang dikembalikan oleh BMKG untuk wilayah ini."

    cuaca_nested = data_list[0].get("cuaca", [])
    records = []
    for grup in cuaca_nested:
        for entry in grup:
            records.append({
                "utc": entry.get("utc_datetime"),
                "local": entry.get("local_datetime"),
                "suhu": entry.get("t"),
                "kelembaban": entry.get("hu"),
                "cuaca": entry.get("weather_desc"),
            })
    
    df = pd.DataFrame(records)
    if not df.empty:
        df['local'] = pd.to_datetime(df['local'], errors='coerce')
        df['utc'] = pd.to_datetime(df['utc'], errors='coerce')
        df = df.sort_values('local').reset_index(drop=True)
    return df


# ========== Train ML Model ==========
@st.cache_resource
def train_model(df):
    df_model = df[["suhu", "kelembaban", "cuaca"]].dropna()
    if df_model.empty:
        return None
    X = df_model[["suhu", "kelembaban"]]
    y = df_model["cuaca"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


# ========== Weather Icon Mapping ==========
def get_weather_emoji(cuaca_desc):
    """Memetakan deskripsi cuaca ke emoji."""
    if not isinstance(cuaca_desc, str): return "❓"
    cuaca_lower = cuaca_desc.lower()
    if "cerah berawan" in cuaca_lower: return "🌤️"
    if "cerah" in cuaca_lower: return "☀️"
    if "berawan" in cuaca_lower: return "☁️"
    if "hujan lebat" in cuaca_lower: return "🌧️"
    if "hujan petir" in cuaca_lower or "badai" in cuaca_lower: return "⛈️"
    if "hujan ringan" in cuaca_lower: return "🌦️"
    if "hujan" in cuaca_lower: return "🌧️"
    if "kabut" in cuaca_lower or "asap" in cuaca_lower: return "🌫️"
    return "🌏"


# ========== Filter data untuk 24 jam ke depan ==========
def filter_24_hours(df):
    if df.empty:
        return df
    
    now = datetime.now()
    next_24h = now + timedelta(hours=24)
    
    # Filter data dalam 24 jam ke depan
    df_filtered = df[
        (df['local'] >= now) & 
        (df['local'] <= next_24h)
    ].copy()
    
    return df_filtered


# ========== Streamlit App ==========
st.title("⛅ Prediksi Cuaca Detail per Wilayah")

# Inisialisasi session state untuk semua level
if 'prov_id' not in st.session_state:
    st.session_state.prov_id = None
if 'kab_id' not in st.session_state:
    st.session_state.kab_id = None
if 'kec_id' not in st.session_state:
    st.session_state.kec_id = None
if 'desa_id' not in st.session_state: # Penambahan state untuk Desa/Kelurahan
    st.session_state.desa_id = None
if 'df_cuaca' not in st.session_state:
    st.session_state.df_cuaca = None # Bisa None, DataFrame, atau String Error

# --- SIDEBAR UNTUK KONTROL ---
with st.sidebar:
    st.header("📍 Pilih Lokasi Detail")

    # Pilihan Provinsi
    df_prov = df_wilayah[df_wilayah['level'] == 0]
    st.selectbox("Provinsi", options=df_prov.index, format_func=lambda id: df_prov.loc[id, 'nama_bersih'], key="prov_id", index=None, placeholder="Pilih Provinsi...")

    # Pilihan Kabupaten/Kota
    if st.session_state.prov_id:
        df_kab = df_wilayah[(df_wilayah['level'] == 1) & (df_wilayah.index.str.startswith(st.session_state.prov_id + '.'))]
        st.selectbox("Kabupaten/Kota", options=df_kab.index, format_func=lambda id: df_kab.loc[id, 'nama_bersih'], key="kab_id", index=None, placeholder="Pilih Kabupaten/Kota...")

    # Pilihan Kecamatan
    if st.session_state.kab_id:
        df_kec = df_wilayah[(df_wilayah['level'] == 2) & (df_wilayah.index.str.startswith(st.session_state.kab_id + '.'))]
        st.selectbox("Kecamatan", options=df_kec.index, format_func=lambda id: df_kec.loc[id, 'nama_bersih'], key="kec_id", index=None, placeholder="Pilih Kecamatan...")

    # --- PERUBAHAN KUNCI: Pilihan Desa/Kelurahan ---
    if st.session_state.kec_id:
        df_desa = df_wilayah[(df_wilayah['level'] == 3) & (df_wilayah.index.str.startswith(st.session_state.kec_id + '.'))]
        st.selectbox("Desa/Kelurahan", options=df_desa.index, format_func=lambda id: df_desa.loc[id, 'nama_bersih'], key="desa_id", index=None, placeholder="Pilih Desa/Kelurahan...")

    # Tombol ambil data aktif jika desa/kelurahan sudah dipilih
    if st.session_state.desa_id:
        if st.button("🌦️ Ambil Data Cuaca", use_container_width=True, type="primary"):
            desa_nama = df_wilayah.loc[st.session_state.desa_id, 'nama_bersih']
            with st.spinner(f"Mengambil data untuk {desa_nama}..."):
                st.session_state.df_cuaca = get_bmkg_data(st.session_state.desa_id)
    else:
        st.info("Pilih wilayah hingga level Desa/Kelurahan untuk mengambil data.")


# --- KONTEN UTAMA ---
# Cek hasil dari session state
hasil_cuaca = st.session_state.df_cuaca

if hasil_cuaca is None:
    st.info("👈 Silakan lengkapi pilihan wilayah di sidebar untuk menampilkan data cuaca.")
elif isinstance(hasil_cuaca, str): # Jika ada pesan error dari fungsi get_bmkg_data
    st.error(hasil_cuaca)
elif not hasil_cuaca.empty:
    df_cuaca = hasil_cuaca
    nama_lokasi = f"{df_wilayah.loc[st.session_state.desa_id, 'nama_bersih']}, Kec. {df_wilayah.loc[st.session_state.kec_id, 'nama_bersih']}"
    st.subheader(f"Perkiraan Cuaca untuk: {nama_lokasi}")

    # Tampilan kartu cuaca (sama seperti sebelumnya)
    cols_per_row = 6
    df_display = df_cuaca.head(12)
    for i in range(0, len(df_display), cols_per_row):
        cols = st.columns(cols_per_row)
        chunk = df_display.iloc[i:i+cols_per_row]
        for idx, col in enumerate(cols):
            if idx < len(chunk):
                row = chunk.iloc[idx]
                jam = row['local'].strftime('%H:%M')
                tanggal = row['local'].strftime('%d %b %Y')
                suhu = f"{row['suhu']}°C"
                kelembaban = f"{row['kelembaban']}%"
                cuaca = row['cuaca']
                emoji = get_weather_emoji(cuaca)
                with col:
                    st.markdown(f"""
                    <div style="
                            border: 1px solid #ddd;
                            border-radius: 10px;
                            padding: 15px;
                            text-align: center;
                            background-color: #f8f9fa;
                            margin: 5px;
                        ">
                        <h2 style="font-size: 2.5em; margin: 0;">{emoji}</h2>
                        <p style="font-weight: bold; font-weight: bold; color: #666;">{jam}</p>
                        <p style="margin: 2px 0; font-size: 12px; color: #888;">{tanggal}</p>
                        <p style="font-weight: bold; color: #e74c3c; margin-top: 5px;">{suhu}</p>
                        <p style="margin: 2px 0; font-size: 12px; color: #3498db;">{kelembaban}</p>
                        <p style="font-size: 0.8em; color: #555; height: 30px;">{cuaca}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Grafik Suhu (sama seperti sebelumnya)
    st.subheader("📊 Grafik Perkiraan Suhu")
    if len(df_cuaca) > 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_cuaca['local'], df_cuaca['suhu'], color='tab:red', marker='o', linewidth=2)
        ax.set_xlabel('Waktu')
        ax.set_ylabel('Suhu (°C)', color='tab:red')
        ax.tick_params(axis='y', labelcolor='tab:red')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        st.pyplot(fig)
        plt.close(fig)
else:
    st.warning("Tidak ada data cuaca yang dapat ditampilkan untuk wilayah ini.")