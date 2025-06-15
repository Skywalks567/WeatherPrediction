import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re



st.set_page_config(page_title="Prediksi Cuaca BMKG Bertingkat", layout="wide")

def filter_24_hours(df):
    """Memfilter DataFrame untuk menampilkan data 24 jam dari sekarang."""
    if df.empty:
        return df
    
    # Dapatkan waktu saat ini sesuai timezone dari data (jika ada)
    now = pd.Timestamp.now(tz=df['local'].dt.tz)
    next_24h = now + timedelta(hours=24)
    
    # Filter data dalam rentang waktu 24 jam ke depan
    df_filtered = df[
        (df['local'] >= now) & 
        (df['local'] <= next_24h)
    ].copy()
    
    return df_filtered

def reset_selections_on_prov_change():
    """Reset pilihan di bawah provinsi jika provinsi berubah."""
    st.session_state.kab_id = None
    st.session_state.kec_id = None
    st.session_state.desa_id = None
    st.session_state.df_cuaca = None
    st.session_state.model = None

def reset_selections_on_kab_change():
    """Reset pilihan di bawah kab/kota jika kab/kota berubah."""
    st.session_state.kec_id = None
    st.session_state.desa_id = None
    st.session_state.df_cuaca = None
    st.session_state.model = None
    
def reset_selections_on_kec_change():
    """Reset pilihan di bawah kecamatan jika kecamatan berubah."""
    st.session_state.desa_id = None
    st.session_state.df_cuaca = None
    st.session_state.model = None

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
    if "hujan petir" in cuaca_lower or "badai" in cuaca_lower or "petir" in cuaca_lower: return "⛈️"
    if "hujan ringan" in cuaca_lower: return "🌦️"
    if "hujan" in cuaca_lower: return "🌧️"
    if "kabut" in cuaca_lower or "asap" in cuaca_lower or "udara kabur" in cuaca_lower: return "🌫️"
    return "🌏"

# ========== Weather Color Mapping ==========
def get_gradient_color(cuaca_desc):
    if not isinstance(cuaca_desc, str):
        return "background: linear-gradient(to bottom, #eeeeee, #cccccc);"  # default abu

    cuaca_lower = cuaca_desc.lower()

    if "cerah berawan" in cuaca_lower:
        return "background: linear-gradient(to bottom, #d0ecff, #90caf9);"  # biru muda → biru langit
    elif "cerah" in cuaca_lower:
        return "background: linear-gradient(to bottom, #fff176, #fbc02d);"  # biru cerah
    elif "berawan" in cuaca_lower and "cerah" not in cuaca_lower:
        return "background: linear-gradient(to bottom, #e0e0e0, #9e9e9e);"  # abu terang → medium
    elif "kabut" in cuaca_lower or "asap" in cuaca_lower or "udara kabur" in cuaca_lower:
        return "background: linear-gradient(to bottom, #d7ccc8, #a1887f);"  # coklat keabu-abuan → redup
    elif "hujan lebat" in cuaca_lower:
        return "background: linear-gradient(to bottom, #a9a9a9, #404040);"  # abu tua

    # ⛈️ Hujan Petir / Badai
    elif "badai" in cuaca_lower or "petir" in cuaca_lower or "hujan petir" in cuaca_lower:
        return "background: linear-gradient(to bottom, #888888, #2c3e50);"  # abu → hitam keabuan

    # 🌦️ Hujan Ringan
    elif "hujan ringan" in cuaca_lower:
        return "background: linear-gradient(to bottom, #d0d0d0, #888888);"  # abu terang → gelap

    # 🌧️ Hujan (umum)
    elif "hujan" in cuaca_lower:
        return "background: linear-gradient(to bottom, #cfd8dc, #78909c);"  # biru abu → biru kelabu

    # Mendung eksplisit
    elif "mendung" in cuaca_lower:
        return "background: linear-gradient(to bottom, #b0b0b0, #707070);"  # abu medium → gelap
    return "background: linear-gradient(to bottom, #e7f4ff, #cceeff);"  # biru langit soft

# ========== Timeline Text Mapping ==========    
def get_text_styles(cuaca_desc):
    cuaca_lower = cuaca_desc.lower() if cuaca_desc else ""

    # ☀️ Cerah → kuning keemasan
    if "cerah" in cuaca_lower and "berawan" not in cuaca_lower:
        return {
            "cuaca": "color: #f57f17;",         # orange tua
            "kelembaban": "color: #4e342e;",    # coklat tua
            "suhu": "color: #bf360c;",          # merah bata
            "tanggal": "color: #6d4c41;",       # coklat gelap
            "jam": "color: #ff8f00;"            # orange terang (kontras)
        }

    # 🌤️ Cerah Berawan → biru langit
    elif "cerah berawan" in cuaca_lower:
        return {
            "cuaca": "color: #01579b;",         
            "kelembaban": "color: #0277bd;",    
            "suhu": "color: #0288d1;",          
            "tanggal": "color: #0288d1;",       
            "jam": "color: #039be5;"            # biru muda terang
        }

    # ☁️ Berawan
    elif "berawan" in cuaca_lower and "cerah" not in cuaca_lower:
        return {
            "cuaca": "color: #eeeeee;",         
            "kelembaban": "color: #cce7ff;",    
            "suhu": "color: #ffcdd2;",          
            "tanggal": "color: #e0f7fa;",
            "jam": "color: #b3e5fc;"            # biru pucat
        }

    # 🌫️ Kabut / Asap
    elif "kabut" in cuaca_lower or "asap" in cuaca_lower or "udara kabur" in cuaca_lower:
        return {
            "cuaca":      "color: #3E2723;",  # Coklat paling gelap, hampir hitam
            "kelembaban": "color: #00695C;",  # Teal (biru kehijauan) gelap untuk kelembaban
            "suhu":       "color: #E65100;",  # Oranye/Amber tua untuk suhu
            "tanggal":    "color: #795548;",  # Coklat medium
            "jam":        "color: #A1887F;"   # Coklat muda            # coklat muda
        }

    # 🌧️ Hujan Lebat
    elif "hujan lebat" in cuaca_lower:
        return {
            "cuaca": "color: #e0f2f1;",         
            "kelembaban": "color: #b2dfdb;",    
            "suhu": "color: #ef9a9a;",          
            "tanggal": "color: #f5f5f5;",
            "jam": "color: #b2ebf2;"            # cyan muda
        }

    # ⛈️ Badai / Petir
    elif "badai" in cuaca_lower or "petir" in cuaca_lower or "hujan petir" in cuaca_lower:
        return {
            "cuaca": "color: #bbdefb;",         
            "kelembaban": "color: #ffcc80;",    
            "suhu": "color: #ef9a9a;",          
            "tanggal": "color: #e0e0e0;",
            "jam": "color: #fff176;"            # kuning terang (kontras dengan gelap)
        }

    # 🌦️ Hujan Ringan
    elif "hujan ringan" in cuaca_lower:
        return {
            "cuaca": "color: #455a64;",         
            "kelembaban": "color: #bbdefb;",    
            "suhu": "color: #ef9a9a;",          
            "tanggal": "color: #eceff1;",
            "jam": "color: #666;"            # abu terang kebiruan
        }

    # 🌧️ Hujan Umum
    elif "hujan" in cuaca_lower:
        return {
            "cuaca": "color: #546e7a;",         
            "kelembaban": "color: #78909c;",    
            "suhu": "color: #ff8a80;",          
            "tanggal": "color: #eceff1;",
            "jam": "color: #b0bec5;"            # abu biru soft
        }

    # ☁️ Mendung eksplisit
    elif "mendung" in cuaca_lower:
        return {
            "cuaca": "color: #757575;",
            "kelembaban": "color: #9e9e9e;",
            "suhu": "color: #e57373;",
            "tanggal": "color: #cfd8dc;",
            "jam": "color: #bdbdbd;"            # abu terang
        }

    # 🌏 Default
    return {
        "cuaca": "color: #37474f;",             
        "kelembaban": "color: #90a4ae;",        
        "suhu": "color: #ef9a9a;",              
        "tanggal": "color: #eceff1;",
        "jam": "color: #cfd8dc;"
    }


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
    st.selectbox("Provinsi", options=df_prov.index, format_func=lambda id: df_prov.loc[id, 'nama_bersih'], key="prov_id",on_change=reset_selections_on_kec_change, index=None, placeholder="Pilih Provinsi...")

    # Pilihan Kabupaten/Kota
    if st.session_state.prov_id:
        df_kab = df_wilayah[(df_wilayah['level'] == 1) & (df_wilayah.index.str.startswith(st.session_state.prov_id + '.'))]
        st.selectbox("Kabupaten/Kota", options=df_kab.index, format_func=lambda id: df_kab.loc[id, 'nama_bersih'], key="kab_id",on_change=reset_selections_on_kec_change, index=None, placeholder="Pilih Kabupaten/Kota...")

    # Pilihan Kecamatan
    if st.session_state.kab_id:
        df_kec = df_wilayah[(df_wilayah['level'] == 2) & (df_wilayah.index.str.startswith(st.session_state.kab_id + '.'))]
        st.selectbox("Kecamatan", options=df_kec.index, format_func=lambda id: df_kec.loc[id, 'nama_bersih'], key="kec_id", on_change=reset_selections_on_kec_change, index=None, placeholder="Pilih Kecamatan...")

    # --- PERUBAHAN KUNCI: Pilihan Desa/Kelurahan ---
    if st.session_state.kec_id:
        df_desa = df_wilayah[(df_wilayah['level'] == 3) & (df_wilayah.index.str.startswith(st.session_state.kec_id + '.'))]
        st.selectbox("Desa/Kelurahan", options=df_desa.index, format_func=lambda id: df_desa.loc[id, 'nama_bersih'], key="desa_id", index=None, placeholder="Pilih Desa/Kelurahan...")

    # Tombol ambil data aktif jika desa/kelurahan sudah dipilih
    if st.session_state.desa_id:
        if st.button("🌦️ Ambil Data Cuaca", use_container_width=True, type="primary"):
            desa_nama = df_wilayah.loc[st.session_state.desa_id, 'nama_bersih']
            with st.spinner(f"Mengambil data untuk {desa_nama}..."):
                # Ambil data cuaca
                hasil_data = get_bmkg_data(st.session_state.desa_id)
                st.session_state.df_cuaca = hasil_data
                # Jika data berhasil didapat (bukan string error), latih model
                if isinstance(hasil_data, pd.DataFrame) and not hasil_data.empty:
                    st.session_state.model = train_model(hasil_data)
                else:
                    # Jika gagal, pastikan model lama dihapus
                    st.session_state.model = None
    else:
        st.info("Pilih wilayah hingga level Desa/Kelurahan untuk mengambil data.")


# --- KONTEN UTAMA ---
# Cek hasil dari session state
hasil_cuaca = st.session_state.df_cuaca
col1, col2 = st.columns([2, 1])

with col1:
    if hasil_cuaca is None:
        st.info("👈 Silakan lengkapi pilihan wilayah di sidebar untuk menampilkan data cuaca.")
    elif isinstance(hasil_cuaca, str): # Jika ada pesan error dari fungsi get_bmkg_data
        st.error(hasil_cuaca)
    elif not hasil_cuaca.empty:
        df_cuaca = hasil_cuaca
        nama_lokasi = f"{df_wilayah.loc[st.session_state.desa_id, 'nama_bersih']}, Kec. {df_wilayah.loc[st.session_state.kec_id, 'nama_bersih']}"
        st.subheader(f"Perkiraan Cuaca untuk: {nama_lokasi}")

        # Tampilan kartu cuaca (sama seperti sebelumnya)
        cols_per_row = 4
        df_display = df_cuaca.head(8)
        for i in range(0, len(df_display), cols_per_row):
            cols = st.columns(cols_per_row)
            chunk = df_display.iloc[i:i+cols_per_row]
            st.markdown("""
                <style>
                    .card-cuaca {
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        padding: 15px;
                        text-align: center;
                        background-color: #f8f9fa;
                        margin: 5px;
                        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out, background-color 0.3s ease-in-out;
                        cursor: pointer;
                    }

                    .card-cuaca:hover {
                        transform: scale(1.05);
                        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                        filter: brightness(1.05);
                    }
                </style>
            """, unsafe_allow_html=True)
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
                        style_background = get_gradient_color(cuaca)
                        text_style = get_text_styles(cuaca)
                        st.markdown(f"""
                        
                        <div class="card-cuaca" style="
                                border: 1px solid #ddd;
                                border-radius: 10px;
                                padding: 15px;
                                text-align: center;
                                background-color: #f8f9fa;
                                {style_background}
                                margin: 5px;
                                transition: 0.3s ease-in-out;
                                transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
                            ">
                            <div style="display: inline-flex; justify-content: center; align-items: center; background: rgba(255,255,255,0.6); border-radius: 50%; aspect-ratio: 1 / 1; width: 3rem;">
                                <span style="font-size: 2rem;">{emoji}</span>
                            </div>
                            <p style="font-weight: bold; font-weight: bold; color: #666;{text_style['jam']}">{jam}</p>
                            <p style="margin: 2px 0; font-size: 12px; color: #888;{text_style['tanggal']}">{tanggal}</p>
                            <p style="font-weight: bold; color: #e74c3c; margin-top: 5px;{text_style['suhu']}">{suhu}</p>
                            <p style="margin: 2px 0; font-size: 12px; color: #3498db;{text_style['kelembaban']}">{kelembaban}</p>
                            <p style="font-size: 0.8em; color: #555; height: 30px;{text_style['cuaca']}">{cuaca}</p>
                        </div>
                        """, unsafe_allow_html=True)

        # Grafik Suhu (sama seperti sebelumnya)
        st.subheader("📊 Grafik Perkiraan 24 Jam ke Depan")

        # Filter data untuk 24 jam ke depan menggunakan fungsi yang baru ditambahkan
        df_24h = filter_24_hours(df_cuaca)

        if len(df_24h) > 1:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot suhu pada sumbu Y pertama (kiri)
            color = 'tab:red'
            ax1.set_xlabel('Waktu (24 Jam ke Depan)')
            ax1.set_ylabel('Suhu (°C)', color=color)
            ax1.plot(df_24h['local'], df_24h['suhu'], color=color, marker='o', linewidth=2, label='Suhu')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
            
            # Format sumbu X untuk menampilkan jam dengan interval 2 jam
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
            
            # Membuat sumbu Y kedua yang berbagi sumbu X yang sama (twinx)
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Kelembaban (%)', color=color)
            ax2.plot(df_24h['local'], df_24h['kelembaban'], color=color, marker='s', linestyle='--', linewidth=2, label='Kelembaban')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Judul dan layout
            plt.title('Perkiraan Suhu dan Kelembaban', fontsize=16, fontweight='bold')
            fig.tight_layout()  # Menyesuaikan layout agar tidak ada yang terpotong
            
            # Menambahkan legenda gabungan dari kedua sumbu
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            st.pyplot(fig)
            plt.close(fig)  # Tutup figure untuk menghemat memori
        else:
            st.warning("Data tidak cukup untuk membuat grafik 24 jam.")
    else:
        st.warning("Tidak ada data cuaca yang dapat ditampilkan untuk wilayah ini.")

with col2:
    # Kondisi sekarang memeriksa 'df_cuaca' dan 'model'
    if ("df_cuaca" in st.session_state and 
        isinstance(st.session_state.df_cuaca, pd.DataFrame) and 
        not st.session_state.df_cuaca.empty):
        
        df_cuaca = st.session_state.df_cuaca

        # Bagian Prediksi Manual sekarang akan muncul jika model ada
        if "model" in st.session_state and st.session_state.model:
            st.subheader("🧠 Prediksi Cuaca Manual")
            st.markdown("Masukkan nilai suhu dan kelembaban untuk prediksi cuaca:")
            
            input_suhu = st.slider("Suhu (°C)", 10, 40, 28)
            input_kelembaban = st.slider("Kelembaban (%)", 20, 100, 70)

            if st.button("🔍 Prediksi", use_container_width=True):
                df_input = pd.DataFrame([{"suhu": input_suhu, "kelembaban": input_kelembaban}])
                
                # Gunakan model dari session state
                model = st.session_state.model
                hasil = model.predict(df_input)[0]
                emoji = get_weather_emoji(hasil)
                
                st.markdown(f"""
                <div style="border: 2px solid #27ae60; border-radius: 10px; padding: 20px; text-align: center; background-color: #d5f4e6; margin: 10px 0;">
                    <h2 style="margin: 0; color: #27ae60;">{emoji}</h2>
                    <h4 style="margin: 10px 0; color: #27ae60;">Prediksi Cuaca:</h4>
                    <h3 style="margin: 0; color: #2c3e50;">{hasil}</h3>
                </div>
                """, unsafe_allow_html=True)
        
        # Info tambahan
        st.subheader("📈 Statistik Data")
        df_stats = df_cuaca[['suhu', 'kelembaban']].describe()
        st.dataframe(df_stats)
        
        st.subheader("🌤️ Jenis Cuaca")
        if 'cuaca' in df_cuaca.columns:
            cuaca_counts = df_cuaca['cuaca'].value_counts()
            
            with st.container(border=True):
                for cuaca, count in cuaca_counts.head(5).items():
                    emoji = get_weather_emoji(cuaca)
                    st.write(f"{emoji} **{cuaca}:** {count} kali")

    # Jika belum ada data, tampilkan placeholder
    else:
        st.subheader("🧠 Info Tambahan")
        st.info("Prediksi manual dan statistik data akan muncul di sini setelah data cuaca berhasil diambil.")
    