import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ========== Load daftar wilayah dari base.csv ==========
@st.cache_data
def get_wilayah_list():
    url = "https://raw.githubusercontent.com/kodewilayah/permendagri-72-2019/main/dist/base.csv"
    df = pd.read_csv(url, header=None, names=["id", "nama"], dtype=str)
    
    # Pecah nama: "Ciwidey, Kab. Bandung, Jawa Barat"
    wilayah = []
    for _, row in df.iterrows():
        nama_parts = row["nama"].split(",")
        kecamatan = nama_parts[0].strip() if len(nama_parts) > 0 else ""
        kota = nama_parts[1].strip() if len(nama_parts) > 1 else ""
        wilayah.append({
            "id": row["id"],
            "kecamatan": kecamatan,
            "kota": kota
        })
    return wilayah


# ========== Ambil data cuaca dari BMKG ==========
@st.cache_data
def get_bmkg_data(kode_wilayah):
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={kode_wilayah}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return pd.DataFrame()

    j = resp.json()
    data_list = j.get("data", [])
    if not data_list:
        return pd.DataFrame()

    cuaca_nested = data_list[0].get("cuaca", [])
    records = []
    for grup in cuaca_nested:
        for entry in grup:
            records.append({
                "utc":        entry.get("utc_datetime"),
                "local":      entry.get("local_datetime"),
                "suhu":       entry.get("t"),
                "kelembaban": entry.get("hu"),
                "cuaca":      entry.get("weather_desc"),
                "ikon":       entry.get("image"),
                "angin":      entry.get("ws"),
                "arah_angin": entry.get("wd"),
                "awan":       entry.get("tcc"),
                "jarak_pandang": entry.get("vs_text")
            })
    
    df = pd.DataFrame(records)
    
    # Convert datetime columns
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
    if not cuaca_desc:
        return "❓"
    
    cuaca_lower = cuaca_desc.lower()
    if "cerah" in cuaca_lower:
        return "☀️"
    elif "berawan" in cuaca_lower:
        return "☁️"
    elif "mendung" in cuaca_lower:
        return "🌫️"
    elif "hujan" in cuaca_lower:
        if "lebat" in cuaca_lower:
            return "🌧️"
        else:
            return "🌦️"
    elif "badai" in cuaca_lower or "petir" in cuaca_lower:
        return "⛈️"
    else:
        return "🌤️"


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
st.set_page_config(page_title="Prediksi Cuaca BMKG", layout="wide")
st.title("⛅ Prediksi Cuaca BMKG - Perkiraan 24 Jam")

# Sidebar untuk kontrol
with st.sidebar:
    st.header("🔧 Pengaturan")
    wilayah_list = get_wilayah_list()
    wilayah_names = [f"{w['kecamatan']} - {w['kota']} ({w['id']})" for w in wilayah_list]

    selected = st.selectbox("Pilih wilayah", wilayah_names)
    selected_id = selected.split("(")[-1].replace(")", "")

    if st.button("🔄 Ambil Data Cuaca", use_container_width=True):
        with st.spinner("Mengambil data dari BMKG..."):
            df = get_bmkg_data(selected_id)

            if not df.empty:
                st.session_state.df = df
                st.session_state.model = train_model(df)
                st.success("✅ Data berhasil diambil!")
            else:
                st.error("❌ Tidak ada data cuaca tersedia.")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Jika data sudah tersedia, tampilkan perkiraan cuaca
    if "df" in st.session_state and not st.session_state.df.empty:
        df_24h = filter_24_hours(st.session_state.df)
        
        if not df_24h.empty:
            st.subheader("🕐 Perkiraan Cuaca 24 Jam Ke Depan")
            
            # Tampilkan dalam format kartu per jam
            st.markdown("### 📅 Timeline Cuaca")
            
            # Buat grid untuk menampilkan cuaca per jam
            cols_per_row = 4
            rows = []
            current_row = []
            
            for idx, row in df_24h.iterrows():
                if pd.notna(row['local']):
                    jam = row['local'].strftime('%H:%M')
                    tanggal = row['local'].strftime('%d/%m')
                    suhu = f"{row['suhu']}°C" if pd.notna(row['suhu']) else "N/A"
                    kelembaban = f"{row['kelembaban']}%" if pd.notna(row['kelembaban']) else "N/A"
                    cuaca = row['cuaca'] if pd.notna(row['cuaca']) else "N/A"
                    emoji = get_weather_emoji(cuaca)
                    
                    current_row.append({
                        'jam': jam,
                        'tanggal': tanggal,
                        'emoji': emoji,
                        'suhu': suhu,
                        'kelembaban': kelembaban,
                        'cuaca': cuaca
                    })
                    
                    if len(current_row) == cols_per_row:
                        rows.append(current_row)
                        current_row = []
            
            # Tambahkan sisa data jika ada
            if current_row:
                rows.append(current_row)
            
            # Tampilkan grid cuaca
            for row_data in rows:
                cols = st.columns(len(row_data))
                for i, data in enumerate(row_data):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 10px;
                            padding: 15px;
                            text-align: center;
                            background-color: #f8f9fa;
                            margin: 5px;
                        ">
                            <h3 style="margin: 0; color: #333;">{data['emoji']}</h3>
                            <p style="margin: 5px 0; font-weight: bold; color: #666;">{data['jam']}</p>
                            <p style="margin: 2px 0; font-size: 12px; color: #888;">{data['tanggal']}</p>
                            <p style="margin: 5px 0; font-weight: bold; color: #e74c3c;">{data['suhu']}</p>
                            <p style="margin: 2px 0; font-size: 12px; color: #3498db;">{data['kelembaban']}</p>
                            <p style="margin: 5px 0; font-size: 11px; color: #27ae60;">{data['cuaca']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Grafik suhu dan kelembaban dengan matplotlib
            st.subheader("📊 Grafik Suhu dan Kelembaban")
            
            if len(df_24h) > 1:
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # Plot suhu
                color = 'tab:red'
                ax1.set_xlabel('Waktu')
                ax1.set_ylabel('Suhu (°C)', color=color)
                ax1.plot(df_24h['local'], df_24h['suhu'], color=color, marker='o', linewidth=2, label='Suhu')
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.grid(True, alpha=0.3)
                
                # Format x-axis untuk menampilkan jam
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                
                # Plot kelembaban pada sumbu y kedua
                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.set_ylabel('Kelembaban (%)', color=color)
                ax2.plot(df_24h['local'], df_24h['kelembaban'], color=color, marker='s', linewidth=2, label='Kelembaban')
                ax2.tick_params(axis='y', labelcolor=color)
                
                # Judul dan layout
                plt.title('Perkiraan Suhu dan Kelembaban 24 Jam', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # Tambahkan legenda
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                st.pyplot(fig)
                plt.close()  # Tutup figure untuk menghemat memori
                
        else:
            st.info("📅 Tidak ada data cuaca untuk 24 jam ke depan.")
            
        # Tampilkan tabel data lengkap
        st.subheader("📋 Data Cuaca Lengkap")
        st.dataframe(st.session_state.df[['local', 'suhu', 'kelembaban', 'cuaca', 'angin', 'arah_angin']].head(20))
        
    else:
        st.info("👈 Silakan pilih wilayah dan klik 'Ambil Data Cuaca' di sidebar untuk melihat perkiraan cuaca.")

with col2:
    # Panel prediksi manual (tetap dipertahankan)
    if "df" in st.session_state and "model" in st.session_state and st.session_state.model:
        st.subheader("🧠 Prediksi Manual")
        st.markdown("Masukkan nilai suhu dan kelembaban untuk prediksi cuaca:")
        
        input_suhu = st.slider("Suhu (°C)", 10, 40, 28)
        input_kelembaban = st.slider("Kelembaban (%)", 20, 100, 70)

        if st.button("🔍 Prediksi", use_container_width=True):
            df_input = pd.DataFrame([{
                "suhu": input_suhu,
                "kelembaban": input_kelembaban
            }])
            hasil = st.session_state.model.predict(df_input)[0]
            emoji = get_weather_emoji(hasil)
            
            st.markdown(f"""
            <div style="
                border: 2px solid #27ae60;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                background-color: #d5f4e6;
                margin: 10px 0;
            ">
                <h2 style="margin: 0; color: #27ae60;">{emoji}</h2>
                <h4 style="margin: 10px 0; color: #27ae60;">Prediksi Cuaca:</h4>
                <h3 style="margin: 0; color: #2c3e50;">{hasil}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    # Info tambahan
    if "df" in st.session_state and not st.session_state.df.empty:
        st.subheader("📈 Statistik Data")
        df_stats = st.session_state.df[['suhu', 'kelembaban']].describe()
        st.dataframe(df_stats)
        
        st.subheader("🌤️ Jenis Cuaca")
        if 'cuaca' in st.session_state.df.columns:
            cuaca_counts = st.session_state.df['cuaca'].value_counts()
            for cuaca, count in cuaca_counts.head(5).items():
                emoji = get_weather_emoji(cuaca)
                st.write(f"{emoji} {cuaca}: {count}x")