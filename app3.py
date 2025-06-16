import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re

st.set_page_config(page_title="Prediksi Cuaca", layout="wide")

def default_theme():
    default_css = """
    <style>
        @keyframes gradient_animation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stApp {
            background: linear-gradient(-45deg, #F8FAFC, #D9EAFD, #BCCCDC, #9AA6B2);
            background-size: 400% 400%;
            animation: gradient_animation 20s ease infinite;
        }

        [data-testid="stSidebar"]{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        [data-testid="stAppViewContainer"] > .main {
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: 15px;
        }
        
        [data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
            color: #1f2937;
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #111827;
        }

        div[data-baseweb="select"] > div {
            background-color: rgb(165, 192, 221, 0.3);
            border: 2px solid rgb(165, 192, 221, 0.1);
            border-radius: 8px;
            color: black;
            backdrop-filter: blur(50px);
            -webkit-backdrop-filter: blur(10px); 
            box-shadow: 0 4px 10px rgba(0,0,0,0.1); 
        }
        
        .stDataFrame div[data-testid="stHorizontalBlock"] {
            background: rgba(255, 255, 255, 0.5);
            border-radius: 10px;
            padding: 10px;
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }

        .stDataFrame td {
            font-size: 14px;
            border: 1px solid #ccc;
        }
    </style>
    """
    st.markdown(default_css, unsafe_allow_html=True)

default_theme()

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
        return "background: linear-gradient(to bottom, #eeeeee, #cccccc);"

    cuaca_lower = cuaca_desc.lower()

    if "cerah berawan" in cuaca_lower:
        return "background: linear-gradient(to bottom, #d0ecff, #90caf9);"
    elif "cerah" in cuaca_lower:
        return "background: linear-gradient(to bottom, #fff176, #fbc02d);"
    elif "berawan" in cuaca_lower and "cerah" not in cuaca_lower:
        return "background: linear-gradient(to bottom, #e0e0e0, #9e9e9e);"
    elif "kabut" in cuaca_lower or "asap" in cuaca_lower or "udara kabur" in cuaca_lower:
        return "background: linear-gradient(to bottom, #d7ccc8, #a1887f);"
    elif "hujan lebat" in cuaca_lower:
        return "background: linear-gradient(to bottom, #a9a9a9, #404040);"

    # ⛈️ Hujan Petir / Badai
    elif "badai" in cuaca_lower or "petir" in cuaca_lower or "hujan petir" in cuaca_lower:
        return "background: linear-gradient(to bottom, #888888, #2c3e50);"

    # 🌦️ Hujan Ringan
    elif "hujan ringan" in cuaca_lower:
        return "background: linear-gradient(to bottom, #d0d0d0, #888888);"

    # 🌧️ Hujan (umum)
    elif "hujan" in cuaca_lower:
        return "background: linear-gradient(to bottom, #cfd8dc, #78909c);" 

    # Mendung eksplisit
    elif "mendung" in cuaca_lower:
        return "background: linear-gradient(to bottom, #b0b0b0, #707070);" 
    return "background: linear-gradient(to bottom, #e7f4ff, #cceeff);" 

# ========== Timeline Text Mapping ==========    
def get_text_styles(cuaca_desc):
    cuaca_lower = cuaca_desc.lower() if cuaca_desc else ""

    if "cerah" in cuaca_lower and "berawan" not in cuaca_lower:
        return {
            "cuaca": "color: #f57f17;",   
            "kelembaban": "color: #4e342e;",  
            "suhu": "color: #bf360c;",    
            "tanggal": "color: #6d4c41;", 
            "jam": "color: #ff8f00;"   
        }

    elif "cerah berawan" in cuaca_lower:
        return {
            "cuaca": "color: #01579b;",         
            "kelembaban": "color: #0277bd;",    
            "suhu": "color: #0288d1;",          
            "tanggal": "color: #0288d1;",       
            "jam": "color: #039be5;"  
        }

    elif "berawan" in cuaca_lower and "cerah" not in cuaca_lower:
        return {
            "cuaca": "color: #eeeeee;",         
            "kelembaban": "color: #cce7ff;",    
            "suhu": "color: #ffcdd2;",          
            "tanggal": "color: #e0f7fa;",
            "jam": "color: #b3e5fc;" 
        }

    elif "kabut" in cuaca_lower or "asap" in cuaca_lower or "udara kabur" in cuaca_lower:
        return {
            "cuaca":      "color: #3E2723;", 
            "kelembaban": "color: #00695C;", 
            "suhu":       "color: #E65100;",
            "tanggal":    "color: #795548;", 
            "jam":        "color: #A1887F;" 
        }

    elif "hujan lebat" in cuaca_lower:
        return {
            "cuaca": "color: #e0f2f1;",         
            "kelembaban": "color: #b2dfdb;",    
            "suhu": "color: #ef9a9a;",          
            "tanggal": "color: #f5f5f5;",
            "jam": "color: #b2ebf2;"
        }

    elif "badai" in cuaca_lower or "petir" in cuaca_lower or "hujan petir" in cuaca_lower:
        return {
            "cuaca": "color: #bbdefb;",         
            "kelembaban": "color: #ffcc80;",    
            "suhu": "color: #ef9a9a;",          
            "tanggal": "color: #e0e0e0;",
            "jam": "color: #fff176;"
        }

    elif "hujan ringan" in cuaca_lower:
        return {
            "cuaca": "color: #455a64;",         
            "kelembaban": "color: #bbdefb;",    
            "suhu": "color: #ef9a9a;",          
            "tanggal": "color: #eceff1;",
            "jam": "color: #666;" 
        }

    elif "hujan" in cuaca_lower:
        return {
            "cuaca": "color: #546e7a;",         
            "kelembaban": "color: #78909c;",    
            "suhu": "color: #ff8a80;",          
            "tanggal": "color: #eceff1;",
            "jam": "color: #b0bec5;"  
        }

    elif "mendung" in cuaca_lower:
        return {
            "cuaca": "color: #757575;",
            "kelembaban": "color: #9e9e9e;",
            "suhu": "color: #e57373;",
            "tanggal": "color: #cfd8dc;",
            "jam": "color: #bdbdbd;"
        }
    
    return {
        "cuaca": "color: #37474f;",             
        "kelembaban": "color: #90a4ae;",        
        "suhu": "color: #ef9a9a;",              
        "tanggal": "color: #eceff1;",
        "jam": "color: #cfd8dc;"
    }

def hex_to_rgb(hex_color):
    """Konversi warna hex (#RRGGBB) ke tuple RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_dominant_colors(df_display, num_colors=3):
    """
    Menganalisis warna dari kartu-kartu yang ditampilkan dan menemukan N warna dominan.
    """
    all_colors_hex = []
    # Kumpulkan semua warna dari gradien kartu
    for _, row in df_display.iterrows():
        grad_css = get_gradient_color(row['cuaca'])
        # Ekstrak warna hex dari string CSS
        hex_codes = re.findall(r'#(?:[0-9a-fA-F]{6})', grad_css)
        all_colors_hex.extend(hex_codes)
    
    if not all_colors_hex:
        return ["#6dd5ed", "#2193b0", "#B0BEC5"] # Fallback default

    # Konversi hex ke RGB untuk clustering
    all_colors_rgb = [hex_to_rgb(c) for c in all_colors_hex]
    
    # Gunakan KMeans untuk menemukan N cluster warna
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init='auto')
    kmeans.fit(all_colors_rgb)
    
    # Pusat cluster adalah warna dominan (dalam RGB)
    dominant_rgb = kmeans.cluster_centers_.astype(int)
    
    # Konversi kembali ke hex untuk digunakan di CSS
    dominant_hex = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in dominant_rgb]
    return dominant_hex

def get_page_background_style(colors):
    """Membuat CSS untuk latar belakang gradien animasi dari 3 warna."""
    if len(colors) < 3:
        colors = ["#6dd5ed", "#2193b0", "#B0BEC5"] # Fallback

    return f"""
    <style>
    @keyframes gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    .stApp {{
        background: linear-gradient(-45deg, {colors[0]}, {colors[1]}, {colors[2]});
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }}

    [data-testid="stAppViewContainer"] > .main {{
        background-color: rgba(255, 255, 255, 0.92);
        border-radius: 15px;
    }}
    .st-emotion-cache-16txtl3 {{
        padding: 2rem 2rem;
    }}

    </style>
    """

# ========== Streamlit App ==========
st.title("⛅ Prediksi Cuaca Detail per Wilayah")

# Inisialisasi session state untuk semua level
if 'prov_id' not in st.session_state:
    st.session_state.prov_id = None
if 'kab_id' not in st.session_state:
    st.session_state.kab_id = None
if 'kec_id' not in st.session_state:
    st.session_state.kec_id = None
if 'desa_id' not in st.session_state:
    st.session_state.desa_id = None
if 'df_cuaca' not in st.session_state:
    st.session_state.df_cuaca = None

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
        nama_lokasi = f"{df_wilayah.loc[st.session_state.desa_id, 'nama_bersih']}, Kec. {df_wilayah.loc[st.session_state.kec_id, 'nama_bersih']}, Kab. {df_wilayah.loc[st.session_state.kab_id, 'nama_bersih']}, Prov. {df_wilayah.loc[st.session_state.prov_id, 'nama_bersih']}"
        st.subheader(f"Perkiraan Cuaca untuk: {nama_lokasi}")

        # Tampilan kartu cuaca (sama seperti sebelumnya)
        cols_per_row = 4
        df_display = df_cuaca.head(8)

        # 1. Dapatkan 3 warna dominan dari 8 kartu yang akan ditampilkan
        dominant_colors = get_dominant_colors(df_display, num_colors=3)
        
        # 2. CSS untuk latar belakang halaman
        page_bg_css = get_page_background_style(dominant_colors)
        st.markdown(page_bg_css, unsafe_allow_html=True)

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

        # Grafik Suhu
        st.subheader("📊 Grafik Perkiraan 24 Jam ke Depan")

        # Filter data untuk 24 jam
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
            
            # Format sumbu X untuk menampilkan jam dengan interval 3 jam
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
        
        html_table = df_stats.to_html(classes="custom-blur-table", border=0)

        # Inject CSS
        st.markdown("""
            <style>
            .custom-blur-table {
                width: 100%;
                border-collapse: collapse;
                background: rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 12px;
                overflow: hidden;
                font-size: 14px;
                color: black;
            }

            .custom-blur-table th {
                background-color: rgb(108, 155, 207, 0.3);
                color: white;
                padding: 10px;
            }

            .custom-blur-table td {
                padding: 8px;
                border: 1px solid rgba(255, 255, 255, 0.4);
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)

        # Tampilkan HTML-nya
        st.markdown(html_table, unsafe_allow_html=True)
        # st.dataframe(df_stats)
    
        if 'cuaca' in df_cuaca.columns:
            cuaca_counts = df_cuaca['cuaca'].value_counts()
            if 'cuaca' in df_cuaca.columns and not df_cuaca['cuaca'].isnull().all():
                cuaca_dominan = df_cuaca['cuaca'].value_counts().idxmax()
                style_background_page = get_page_background_style(cuaca_dominan)
                st.markdown(style_background_page, unsafe_allow_html=True)\
            
            st.markdown(f"""
                <style>
                .st-emotion-cache-1d8vwwt.e1lln2w84, #Jenis-Cuaca{{
                    background: rgba(255, 255, 255, 0.15);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}
                </style>
                """, unsafe_allow_html=True)
            
            with st.container(border=True):
                st.subheader("🌤️ Jenis Cuaca")
                for cuaca, count in cuaca_counts.head(5).items():
                    emoji = get_weather_emoji(cuaca)
                    st.write(f"{emoji} **{cuaca}:** {count} kali")

    # Jika belum ada data, tampilkan placeholder
    else:
        st.subheader("🧠 Info Tambahan")
        st.info("Prediksi manual dan statistik data akan muncul di sini setelah data cuaca berhasil diambil.")
    
