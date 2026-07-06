# ============================================================================
# DASBOR INTERAKTIF PRODUKSI TANAMAN PERKEBUNAN INDONESIA
# ============================================================================
# Framework: Streamlit | Data: Pandas/NumPy | Visualization: Plotly
# ML: scikit-learn | Stats: SciPy | Version: 1.0.0
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import math
from datetime import datetime
import io
import base64

warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI APLIKASI
# ============================================================================

APP_VERSION = "1.0.0"
APP_NAME = "Dasbor Produksi Perkebunan Indonesia"
APP_ICON = "🌿"
DATA_YEAR = 2024

KOMODITAS_LIST = [
    'Kelapa_Sawit', 'Kelapa', 'Karet', 'Kopi',
    'Kakao', 'Teh', 'Tebu'
]

KOMODITAS_LABELS = {
    'Kelapa_Sawit': '🌴 Kelapa Sawit',
    'Kelapa': '🥥 Kelapa',
    'Karet': '🌳 Karet',
    'Kopi': '☕ Kopi',
    'Kakao': '🍫 Kakao',
    'Teh': '🍵 Teh',
    'Tebu': '🎋 Tebu'
}

KOMODITAS_COLORS = {
    'Kelapa_Sawit': '#f39c12',
    'Kelapa': '#8d6e63',
    'Karet': '#5d4037',
    'Kopi': '#4e342e',
    'Kakao': '#6d4c41',
    'Teh': '#2e7d32',
    'Tebu': '#9ccc65'
}

REGION_COLORS = {
    'Sumatera': '#2ecc71',
    'Jawa': '#f1c40f',
    'Bali & Nusa Tenggara': '#e67e22',
    'Kalimantan': '#27ae60',
    'Sulawesi': '#16a085',
    'Maluku': '#3498db',
    'Papua': '#9b59b6'
}

PROV_COORDS = {
    'ACEH': (5.5483, 95.3238), 'SUMATERA UTARA': (3.5952, 98.6722),
    'SUMATERA BARAT': (-0.7893, 100.3291), 'RIAU': (1.0, 101.4474),
    'JAMBI': (-1.6101, 103.6131), 'SUMATERA SELATAN': (-3.3194, 104.9146),
    'BENGKULU': (-3.7926, 102.2614), 'LAMPUNG': (-5.3971, 105.2668),
    'KEP. BANGKA BELITUNG': (-2.7411, 106.4406), 'KEP. RIAU': (3.9453, 108.1428),
    'DKI JAKARTA': (-6.2088, 106.8456), 'JAWA BARAT': (-6.9175, 107.6191),
    'JAWA TENGAH': (-7.1509, 110.1403), 'DI YOGYAKARTA': (-7.7956, 110.3695),
    'JAWA TIMUR': (-7.5361, 112.2384), 'BANTEN': (-6.4058, 106.0640),
    'BALI': (-8.3405, 115.0920), 'NUSA TENGGARA BARAT': (-8.5833, 116.1167),
    'NUSA TENGGARA TIMUR': (-8.6574, 121.0794), 'KALIMANTAN BARAT': (-0.0263, 109.3425),
    'KALIMANTAN TENGAH': (-1.6815, 113.3824), 'KALIMANTAN SELATAN': (-3.3186, 114.5944),
    'KALIMANTAN TIMUR': (1.2379, 116.8529), 'KALIMANTAN UTARA': (3.0731, 116.0414),
    'SULAWESI UTARA': (1.4748, 124.8421), 'SULAWESI TENGAH': (-0.8950, 119.8376),
    'SULAWESI SELATAN': (-3.6688, 119.9741), 'SULAWESI TENGGARA': (-3.9563, 122.5097),
    'GORONTALO': (0.5417, 123.0594), 'SULAWESI BARAT': (-2.6748, 119.1051),
    'MALUKU': (-3.2385, 130.1453), 'MALUKU UTARA': (1.5710, 127.7893),
    'PAPUA BARAT': (-1.3361, 132.4085), 'PAPUA BARAT DAYA': (-1.5897, 131.2011),
    'PAPUA': (-2.5916, 140.6690), 'PAPUA SELATAN': (-7.4833, 140.7500),
    'PAPUA TENGAH': (-3.3171, 137.3811), 'PAPUA PEGUNUNGAN': (-4.0433, 138.9611)
}

# ============================================================================
# CSS CUSTOM (MINIMALIS VERSI)
# ============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;900&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-primary: #0a1f14;
    --bg-secondary: #0f2a1c;
    --emerald: #2ecc71;
    --gold: #f1c40f;
    --text-primary: #e8f5e9;
    --text-secondary: #a8d5ba;
    --border: rgba(46, 204, 113, 0.3);
}

.stApp { background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%); 
         color: var(--text-primary); font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d3320 0%, var(--bg-primary) 100%);
                                   border-right: 2px solid var(--emerald); }

.main-title {
    background: linear-gradient(90deg, #2ecc71 0%, #58d68d 25%, #f1c40f 50%, #f7dc6f 75%, #27ae60 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3em;
    font-weight: 900;
    text-align: center;
    margin-bottom: 10px;
    font-family: 'Poppins', sans-serif;
}

.sub-title { text-align: center; color: var(--text-secondary); font-size: 1.1em; margin-bottom: 30px; }

.page-header {
    background: linear-gradient(135deg, rgba(46, 204, 113, 0.15) 0%, rgba(241, 196, 15, 0.08) 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(46, 204, 113, 0.2);
}

.metric-card {
    background: linear-gradient(135deg, rgba(46, 204, 113, 0.18) 0%, rgba(241, 196, 15, 0.12) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 10px 40px rgba(46, 204, 113, 0.2);
    transition: all 0.3s ease;
}

.metric-card:hover { transform: translateY(-8px); box-shadow: 0 20px 50px rgba(46, 204, 113, 0.3); }

.metric-value { font-size: 1.8em; font-weight: 800; color: var(--gold); }
.metric-label { color: var(--text-secondary); font-size: 0.85em; margin-top: 8px; }

.insight-box {
    background: linear-gradient(135deg, rgba(46, 204, 113, 0.12) 0%, rgba(46, 204, 113, 0.05) 100%);
    border-left: 6px solid var(--emerald);
    padding: 18px;
    border-radius: 12px;
    margin: 12px 0;
}

.recommend-box {
    background: linear-gradient(135deg, rgba(241, 196, 15, 0.12) 0%, rgba(241, 196, 15, 0.05) 100%);
    border-left: 6px solid var(--gold);
    padding: 18px;
    border-radius: 12px;
    margin: 12px 0;
}

h1, h2, h3 { color: var(--emerald) !important; font-family: 'Poppins', sans-serif; }

.stButton>button {
    background: linear-gradient(135deg, #0d3320 0%, var(--emerald) 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton>button:hover { transform: translateY(-2px); 
                         background: linear-gradient(135deg, var(--emerald) 0%, var(--gold) 100%); }

::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: #0d3320; border-radius: 5px; }

.app-footer {
    background: linear-gradient(90deg, rgba(13, 51, 32, 0.8) 0%, rgba(15, 42, 28, 0.8) 100%);
    border-top: 2px solid var(--emerald);
    padding: 25px;
    text-align: center;
    margin-top: 50px;
    border-radius: 15px 15px 0 0;
}

.footer-title { color: var(--emerald); font-weight: 700; font-size: 1.1em; }
.app-footer p { color: var(--text-secondary); margin: 5px 0; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# FUNGSI-FUNGSI UTILITY
# ============================================================================

def load_data(file_path="produksi_tanaman.csv"):
    try:
        import os
        if not os.path.exists(file_path):
            return None, f"❌ File '{file_path}' tidak ditemukan"
        df = pd.read_csv(file_path)
        if df.empty or 'Provinsi' not in df.columns:
            return None, "❌ Format data tidak sesuai"
        return df, None
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def assign_region(provinsi):
    prov_upper = str(provinsi).upper().strip()
    region_mapping = {
        'Sumatera': ['ACEH', 'SUMATERA', 'RIAU', 'JAMBI', 'BENGKULU', 'LAMPUNG', 'BANGKA', 'KEP. RIAU'],
        'Jawa': ['DKI', 'JAWA', 'YOGYAKARTA', 'BANTEN', 'JAKARTA'],
        'Bali & Nusa Tenggara': ['BALI', 'NUSA TENGGARA'],
        'Kalimantan': ['KALIMANTAN'],
        'Sulawesi': ['SULAWESI', 'GORONTALO'],
        'Maluku': ['MALUKU'],
        'Papua': ['PAPUA']
    }
    for region, keywords in region_mapping.items():
        for keyword in keywords:
            if keyword in prov_upper: return region
    return 'Lainnya'

def format_number(num, decimals=0):
    try:
        if pd.isna(num): return "N/A"
        if decimals == 0:
            return f"{int(num):,}".replace(',', '.')
        else:
            return f"{num:,.{decimals}f}".replace(',', '#').replace('.', ',').replace('#', '.')
    except: return "N/A"

def calculate_herfindahl_index(df, komoditas_columns):
    hhi_results = {}
    for idx, row in df.iterrows():
        provinsi = row['Provinsi']
        values = [row[col] for col in komoditas_columns]
        total = sum(values)
        if total > 0:
            shares = [v / total for v in values]
            hhi = sum(s ** 2 for s in shares)
            hhi_results[provinsi] = hhi
        else:
            hhi_results[provinsi] = 0.0
    return hhi_results

def classify_diversification(hhi):
    if hhi < 0.15: return "🌈 Sangat Diversifikasi"
    elif hhi < 0.25: return "🎨 Moderat Diversifikasi"
    elif hhi < 0.40: return "⚖️ Kurang Diversifikasi"
    else: return "🎯 Sangat Spesialisasi"

def apply_plotly_theme(fig, title=None, height=600):
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(10, 31, 20, 0.3)",
        font=dict(family="Inter, sans-serif", color="#e8f5e9", size=13),
        title=dict(text=title if title else "", font=dict(size=20, color="#58d68d"), x=0.5, xanchor="center"),
        height=height,
        margin=dict(l=50, r=50, t=60, b=50),
        hoverlabel=dict(bgcolor="#0f2a1c", font_size=14, bordercolor="#2ecc71"),
        legend=dict(bgcolor="rgba(15, 42, 28, 0.8)", bordercolor="#2ecc71", borderwidth=1, font=dict(color="#e8f5e9"))
    )
    return fig

def create_metric_card(icon, value, label, change=None):
    change_html = f'<div style="margin-top: 8px; color: #2ecc71;">{change}</div>' if change else ""
    return f'''
    <div class="metric-card">
        <div style="font-size: 2em; margin-bottom: 8px;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {change_html}
    </div>
    '''

def calculate_gini_coefficient(series):
    try:
        sorted_values = series.sort_values().reset_index(drop=True)
        n = len(sorted_values)
        if sorted_values.sum() == 0: return 0.0
        cumsum = sorted_values.cumsum()
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values)) / (n * cumsum.iloc[-1])) - (n + 1) / n
        return max(0, min(1, gini))
    except: return np.nan

# ============================================================================
# LOAD & PREPROCESS DATA
# ============================================================================

@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    df_raw, error = load_data()
    if error: return None, None, error
    
    df = df_raw.copy()
    df['Wilayah'] = df['Provinsi'].apply(assign_region)
    df['Total_Produksi'] = df[KOMODITAS_LIST].sum(axis=1)
    df['Latitude'] = df['Provinsi'].map(lambda p: PROV_COORDS.get(p, (np.nan, np.nan))[0])
    df['Longitude'] = df['Provinsi'].map(lambda p: PROV_COORDS.get(p, (np.nan, np.nan))[1])
    df['Rank_Total'] = df['Total_Produksi'].rank(ascending=False, method='min').astype(int)
    
    def get_dominant_commodity(row):
        values = {k: row[k] for k in KOMODITAS_LIST}
        max_key = max(values, key=values.get)
        return KOMODITAS_LABELS.get(max_key, max_key)
    
    df['Komoditas_Dominan'] = df.apply(get_dominant_commodity, axis=1)
    hhi_dict = calculate_herfindahl_index(df, KOMODITAS_LIST)
    df['HHI_Index'] = df['Provinsi'].map(hhi_dict)
    df['Diversifikasi'] = df['HHI_Index'].apply(classify_diversification)
    
    return df_raw, df, None

df_raw, df, data_error = load_and_preprocess_data()

if data_error:
    st.error(data_error)
    st.info("📋 Pastikan file `produksi_tanaman.csv` ada di folder yang sama dengan `app.py`")
    st.stop()

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================

st.set_page_config(
    page_title=f"{APP_ICON} {APP_NAME}",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SIDEBAR NAVIGASI
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; border-bottom: 2px solid rgba(46, 204, 113, 0.3);">
        <div style="font-size: 3em;">🌿🌾🌴</div>
        <div style="color: #58d68d; font-size: 1.1em; font-weight: 700; margin-top: 10px;">
            DASHBOARD PERKEBUNAN
        </div>
        <div style="color: #f7dc6f; font-size: 0.85em; margin-top: 5px;">
            Indonesia 🇮🇩
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Navigasi")
    
    page = st.radio(
        "Pilih halaman:",
        [
            "📊 Overview",
            "🧹 Data Cleaning",
            "📈 EDA & 3D Visualization",
            "🗺️ Peta Distribusi",
            "🔗 Korelasi & Regresi",
            "🎯 Machine Learning",
            "💡 Insights",
            "📚 Tentang"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(f"### 📊 Info Dataset\n- **Provinsi:** {len(df)}\n- **Komoditas:** {len(KOMODITAS_LIST)}\n- **Total Produksi:** {format_number(df['Total_Produksi'].sum())}")

# ============================================================================
# HEADER UTAMA
# ============================================================================

st.markdown(f'<h1 class="main-title">🌿 Dasbor Produksi Perkebunan Indonesia</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analisis Visual 3D Interaktif • Data Visualization 2026</p>', unsafe_allow_html=True)

# ============================================================================
# PAGES
# ============================================================================

if "Overview" in page:
    st.markdown("""
    <div class="page-header">
        <h1>📊 Overview & Data Understanding</h1>
        <p>Gambaran umum struktur dan konten dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    total_semua = df[KOMODITAS_LIST].sum().sum()
    with col1:
        st.markdown(create_metric_card("🌾", format_number(total_semua), "Total Produksi", "ribu ton"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("🌴", format_number(df['Kelapa_Sawit'].sum()), "Kelapa Sawit", f"{(df['Kelapa_Sawit'].sum()/total_semua*100):.1f}%"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("☕", format_number(df['Kopi'].sum()), "Kopi", f"{(df['Kopi'].sum()/total_semua*100):.1f}%"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("🍫", format_number(df['Kakao'].sum()), "Kakao", f"{(df['Kakao'].sum()/total_semua*100):.1f}%"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📋 Data Mentah")
    n_rows = st.slider("Jumlah baris:", 5, 38, 15)
    st.dataframe(df.head(n_rows).style.background_gradient(subset=KOMODITAS_LIST, cmap='Greens'), use_container_width=True)
    
    csv_data = df.to_csv(index=False)
    st.download_button("💾 Download Data (CSV)", csv_data, "produksi_tanaman.csv", "text/csv", use_container_width=True)
    
    st.markdown("---")
    st.subheader("📉 Statistik Deskriptif")
    stats_df = pd.DataFrame({
        'Komoditas': KOMODITAS_LIST,
        'Mean': [df[k].mean() for k in KOMODITAS_LIST],
        'Median': [df[k].median() for k in KOMODITAS_LIST],
        'Std Dev': [df[k].std() for k in KOMODITAS_LIST],
        'Min': [df[k].min() for k in KOMODITAS_LIST],
        'Max': [df[k].max() for k in KOMODITAS_LIST]
    })
    st.dataframe(stats_df.style.format('{:.2f}'), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📦 Box Plot Distribusi")
    df_melted = df.melt(id_vars=['Provinsi'], value_vars=KOMODITAS_LIST, var_name='Komoditas', value_name='Produksi')
    fig_box = px.box(df_melted, x='Komoditas', y='Produksi', color='Komoditas',
                     color_discrete_sequence=[KOMODITAS_COLORS.get(k, '#2ecc71') for k in KOMODITAS_LIST])
    fig_box = apply_plotly_theme(fig_box, title="Distribusi Produksi per Komoditas", height=450)
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

elif "Data Cleaning" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🧹 Data Cleaning & Preprocessing</h1>
        <p>Pembersihan dan validasi data</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("1️⃣ Missing Values")
    missing_count = df_raw.isnull().sum().sum()
    if missing_count == 0:
        st.success("✅ Dataset bersih - tidak ada missing values")
    else:
        st.warning(f"⚠️ Ditemukan {missing_count} missing values")
    
    st.subheader("2️⃣ Data Duplikat")
    n_duplicates = df_raw.duplicated().sum()
    if n_duplicates == 0:
        st.success("✅ Dataset bersih - tidak ada duplikat")
    else:
        st.warning(f"⚠️ Ditemukan {n_duplicates} duplikat")
    
    st.subheader("3️⃣ Feature Engineering")
    st.markdown("""
    ✨ Fitur baru yang dibuat:
    - **Wilayah**: Pengelompokan geografis
    - **Total_Produksi**: Jumlah semua komoditas
    - **Rank_Total**: Ranking berdasarkan total
    - **Komoditas_Dominan**: Komoditas dengan produksi tertinggi
    - **HHI_Index**: Indeks diversifikasi
    - **Diversifikasi**: Kategori diversifikasi
    """)
    
    st.dataframe(df[['Provinsi', 'Wilayah', 'Total_Produksi', 'Komoditas_Dominan', 'HHI_Index', 'Diversifikasi']].head(10), use_container_width=True)

elif "EDA & 3D" in page:
    st.markdown("""
    <div class="page-header">
        <h1>📈 Exploratory Data Analysis & 3D Visualizations</h1>
        <p>Eksplorasi data dengan visualisasi 3D interaktif</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🔬 Analisis Univariat")
    komoditas_uni = st.selectbox("Pilih komoditas:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x, x), key="uni")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{df[komoditas_uni].mean():.2f}")
    with col2:
        st.metric("Median", f"{df[komoditas_uni].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df[komoditas_uni].std():.2f}")
    with col4:
        st.metric("Min", f"{df[komoditas_uni].min():.2f}")
    
    fig_hist = px.histogram(df, x=komoditas_uni, nbinsx=15, title=f"Distribusi {KOMODITAS_LABELS.get(komoditas_uni)}")
    fig_hist = apply_plotly_theme(fig_hist, height=400)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🧊 Visualisasi 3D #1: Scatter 3D")
    
    col_3d1, col_3d2, col_3d3 = st.columns(3)
    with col_3d1:
        axis_x = st.selectbox("Sumbu X:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), key="3d_x")
    with col_3d2:
        axis_y = st.selectbox("Sumbu Y:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), index=1, key="3d_y")
    with col_3d3:
        axis_z = st.selectbox("Sumbu Z:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), index=2, key="3d_z")
    
    fig_3d = px.scatter_3d(df, x=axis_x, y=axis_y, z=axis_z, color='Wilayah', size='Total_Produksi',
                           hover_name='Provinsi', color_discrete_map=REGION_COLORS, title="Scatter 3D Interaktif")
    fig_3d = apply_plotly_theme(fig_3d, height=650)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🏔️ Visualisasi 3D #2: Surface Plot")
    
    n_prov = st.slider("Jumlah provinsi teratas:", 5, 30, 15)
    df_sorted = df.nlargest(n_prov, 'Total_Produksi').set_index('Provinsi')[KOMODITAS_LIST]
    Z = df_sorted.values
    X = np.arange(len(KOMODITAS_LIST))
    Y = np.arange(len(df_sorted))
    
    fig_surface = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig_surface.update_layout(
        scene=dict(
            xaxis_title='Komoditas',
            yaxis_title='Provinsi',
            zaxis_title='Produksi'
        ),
        height=600
    )
    fig_surface = apply_plotly_theme(fig_surface, title="Topografi Produksi", height=650)
    st.plotly_chart(fig_surface, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🎨 Visualisasi 3D #3: Stacked Bar Komposisi")
    
    n_stack = st.slider("Jumlah provinsi (stacked):", 5, 25, 15)
    top_stack = df.nlargest(n_stack, 'Total_Produksi').reset_index(drop=True)
    
    fig_stacked = go.Figure()
    for komoditas in KOMODITAS_LIST:
        fig_stacked.add_trace(go.Bar(x=top_stack['Provinsi'], y=top_stack[komoditas],
                                    name=KOMODITAS_LABELS.get(komoditas, komoditas),
                                    marker_color=KOMODITAS_COLORS.get(komoditas, '#2ecc71')))
    
    fig_stacked = apply_plotly_theme(fig_stacked, title=f"Komposisi Produksi - Top {n_stack} Provinsi", height=500)
    fig_stacked.update_layout(barmode='stack', xaxis_tickangle=-45)
    st.plotly_chart(fig_stacked, use_container_width=True)

elif "Peta Distribusi" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🗺️ Peta Distribusi Produksi Indonesia</h1>
        <p>Visualisasi geospasial di seluruh provinsi</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_map1, col_map2 = st.columns([2, 1])
    with col_map1:
        komoditas_peta = st.selectbox("Pilih komoditas:", ['Total_Produksi'] + KOMODITAS_LIST,
                                     format_func=lambda x: "🌾 Total" if x == 'Total_Produksi' else KOMODITAS_LABELS.get(x, x), key="peta")
    with col_map2:
        color_scale = st.selectbox("Color scale:", ['Greens', 'Viridis', 'Plasma'], key="color_scale")
    
    df_map = df.dropna(subset=['Latitude', 'Longitude'])
    
    fig_geo = px.scatter_geo(df_map, lat='Latitude', lon='Longitude', color=komoditas_peta, size=komoditas_peta,
                            hover_name='Provinsi', scope='asia', color_continuous_scale=color_scale,
                            title=f"Distribusi {komoditas_peta.replace('_', ' ')}")
    fig_geo.update_geos(showcountries=True, showland=True, oceancolor='rgba(52, 152, 219, 0.3)')
    fig_geo = apply_plotly_theme(fig_geo, height=650)
    st.plotly_chart(fig_geo, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🎯 Komoditas Dominan per Provinsi")
    
    fig_dominant = go.Figure()
    for komoditas in KOMODITAS_LIST:
        label = KOMODITAS_LABELS.get(komoditas, komoditas)
        color = KOMODITAS_COLORS.get(komoditas, '#2ecc71')
        mask = df_map['Komoditas_Dominan'] == label
        subset = df_map[mask]
        
        if len(subset) > 0:
            fig_dominant.add_trace(go.Scattergeo(lat=subset['Latitude'], lon=subset['Longitude'],
                                               text=subset['Provinsi'], mode='markers+text',
                                               marker=dict(size=15, color=color, line=dict(color='white', width=1)),
                                               name=label))
    
    fig_dominant.update_layout(geo=dict(scope='asia', showland=True, oceancolor='rgba(52, 152, 219, 0.3)'),
                              height=700)
    fig_dominant = apply_plotly_theme(fig_dominant, title="Komoditas Dominan per Provinsi", height=750)
    st.plotly_chart(fig_dominant, use_container_width=True)

elif "Korelasi" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🔗 Analisis Korelasi & Regresi</h1>
        <p>Analisis hubungan antar komoditas dan model prediktif</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🔥 Matriks Korelasi Pearson")
    corr_matrix = df[KOMODITAS_LIST].corr(method='pearson')
    
    fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                        x=[KOMODITAS_LABELS.get(c, c) for c in corr_matrix.columns],
                                        y=[KOMODITAS_LABELS.get(c, c) for c in corr_matrix.index],
                                        colorscale='RdYlGn', zmin=-1, zmax=1,
                                        text=corr_matrix.values.round(2), texttemplate="%{text}"))
    fig_corr = apply_plotly_theme(fig_corr, title="Matriks Korelasi Pearson", height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Analisis Regresi Linear")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        target_var = st.selectbox("Target (diprediksi):", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), key="reg_target")
    with col_r2:
        test_size_ml = st.slider("Test size (%):", 10, 40, 20, key="test_size_ml") / 100
    
    feature_vars = [k for k in KOMODITAS_LIST if k != target_var]
    X = df[feature_vars].values
    y = df[target_var].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ml, random_state=42)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_sc, y_train)
    
    y_pred_train = model.predict(X_train_sc)
    y_pred_test = model.predict(X_test_sc)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("R² Train", f"{r2_train:.4f}")
    with col_m2:
        st.metric("R² Test", f"{r2_test:.4f}")
    with col_m3:
        st.metric("MAE Test", f"{mae_test:.2f}")
    with col_m4:
        st.metric("RMSE Test", f"{rmse_test:.2f}")
    
    st.markdown("---")
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(x=y_test, y=y_pred_test, mode='markers', name='Predictions',
                               marker=dict(size=10, color='#2ecc71')))
    all_vals = np.concatenate([y_test, y_pred_test])
    min_v, max_v = min(all_vals), max(all_vals)
    fig_avp.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', name='Perfect',
                               line=dict(color='#e74c3c', dash='dash')))
    fig_avp = apply_plotly_theme(fig_avp, title="Actual vs Predicted", height=500)
    st.plotly_chart(fig_avp, use_container_width=True)

elif "Machine Learning" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🎯 Machine Learning Lanjutan</h1>
        <p>Perbandingan berbagai algoritma regresi</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_ml1, col_ml2 = st.columns(2)
    with col_ml1:
        target_ml = st.selectbox("Target:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), key="ml_target")
    with col_ml2:
        test_size_ml = st.slider("Test size:", 10, 40, 20, key="ml_size") / 100
    
    selected_models = st.multiselect("Pilih model:", [
        "Linear Regression", "Ridge", "Lasso", "Random Forest", "Gradient Boosting"
    ], default=["Linear Regression", "Random Forest"])
    
    feature_vars_ml = [k for k in KOMODITAS_LIST if k != target_ml]
    X_ml = df[feature_vars_ml].values
    y_ml = df[target_ml].values
    
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=test_size_ml, random_state=42)
    scaler_ml = StandardScaler()
    X_train_ml_sc = scaler_ml.fit_transform(X_train_ml)
    X_test_ml_sc = scaler_ml.transform(X_test_ml)
    
    results = []
    progress_bar = st.progress(0)
    
    for idx, model_name in enumerate(selected_models):
        if model_name == "Linear Regression":
            model = LinearRegression()
            X_tr, X_te = X_train_ml_sc, X_test_ml_sc
        elif model_name == "Ridge":
            model = Ridge(alpha=1.0)
            X_tr, X_te = X_train_ml_sc, X_test_ml_sc
        elif model_name == "Lasso":
            model = Lasso(alpha=1.0, max_iter=10000)
            X_tr, X_te = X_train_ml_sc, X_test_ml_sc
        elif model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            X_tr, X_te = X_train_ml, X_test_ml
        else:
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            X_tr, X_te = X_train_ml, X_test_ml
        
        model.fit(X_tr, y_train_ml)
        y_pred_te = model.predict(X_te)
        
        r2 = r2_score(y_test_ml, y_pred_te)
        mae = mean_absolute_error(y_test_ml, y_pred_te)
        rmse = np.sqrt(mean_squared_error(y_test_ml, y_pred_te))
        
        results.append({'Model': model_name, 'R² Test': r2, 'MAE': mae, 'RMSE': rmse})
        progress_bar.progress((idx + 1) / len(selected_models))
    
    progress_bar.empty()
    results_df = pd.DataFrame(results).sort_values('R² Test', ascending=False)
    
    st.subheader("📊 Hasil Perbandingan Model")
    st.dataframe(results_df.style.highlight_max(subset=['R² Test']), use_container_width=True, hide_index=True)
    
    fig_compare = px.bar(results_df, x='Model', y=['R² Test'], title='Perbandingan R² Score Model')
    fig_compare = apply_plotly_theme(fig_compare, height=450)
    st.plotly_chart(fig_compare, use_container_width=True)

elif "Insights" in page:
    st.markdown("""
    <div class="page-header">
        <h1>💡 Insights & Rekomendasi</h1>
        <p>Kesimpulan dan rekomendasi strategis</p>
    </div>
    """, unsafe_allow_html=True)
    
    top_prov = df.loc[df['Kelapa_Sawit'].idxmax(), 'Provinsi']
    top_val = df['Kelapa_Sawit'].max()
    pct_sawit = (df['Kelapa_Sawit'].sum() / df['Total_Produksi'].sum()) * 100
    gini = calculate_gini_coefficient(df['Total_Produksi'])
    
    st.markdown(f"""
    <div class="insight-box">
        <h3>🔍 Insight Utama:</h3>
        <ul>
            <li>Kelapa Sawit mendominasi <b>{pct_sawit:.1f}%</b> produksi nasional</li>
            <li><b>{top_prov}</b> adalah produsen terbesar ({top_val:,.0f} ribu ton)</li>
            <li>Koefisien Gini = <b>{gini:.3f}</b> (ketimpangan {"tinggi" if gini > 0.5 else "sedang"})</li>
            <li>Diversifikasi komoditas masih rendah di banyak provinsi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🎯 5 Rekomendasi Strategis")
    
    st.markdown("""
    <div class="recommend-box">
        <h3 style="color: var(--gold);">🎯 #1: Program Diversifikasi Komoditas</h3>
        <p>Kurangi ketergantungan pada sawit melalui tumpang sari dan insentif fiskal untuk komoditas alternatif.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommend-box">
        <h3 style="color: var(--gold);">🎯 #2: Pengembangan Infrastruktur Indonesia Timur</h3>
        <p>Percepatan pembangunan jalan, pelabuhan, dan cold storage untuk aktivasi potensi belum tergarap.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommend-box">
        <h3 style="color: var(--gold);">🎯 #3: Sertifikasi Sustainability</h3>
        <p>Percepatan ISPO/RSPO dan blockchain traceability untuk akses pasar premium global.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommend-box">
        <h3 style="color: var(--gold);">🎯 #4: Digitalisasi Sistem Informasi</h3>
        <p>Dashboard nasional real-time, IoT sensor, dan mobile app untuk petani dan produsen.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommend-box">
        <h3 style="color: var(--gold);">🎯 #5: Hilirisasi dan Nilai Tambah</h3>
        <p>Insentif industri pengolahan untuk meningkatkan rasio produk olahan dari 30% menjadi 60%.</p>
    </div>
    """, unsafe_allow_html=True)

elif "Tentang" in page:
    st.markdown("""
    <div class="page-header">
        <h1>📚 Tentang Aplikasi</h1>
        <p>Informasi lengkap tentang dasbor dan teknologi</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🌿 Deskripsi")
    st.markdown(f"""
    **Dasbor Produksi Perkebunan Indonesia** adalah aplikasi web interaktif berbasis Streamlit untuk menganalisis 
    dan memvisualisasikan data produksi tanaman perkebunan di **{len(df)} provinsi Indonesia**.
    
    Dibangun sebagai tugas **UAS Visualisasi Data** dengan tema *Sains Data & Pertanian Indonesia*.
    """)
    
    st.subheader("🛠️ Teknologi yang Digunakan")
    tech_data = {
        'Teknologi': ['Streamlit', 'Pandas', 'NumPy', 'Plotly', 'scikit-learn', 'SciPy', 'Python'],
        'Fungsi': ['Framework web', 'Data manipulation', 'Numeric computing', 'Visualization', 'Machine Learning', 'Statistics', 'Core Language']
    }
    st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)
    
    st.subheader("📊 Dataset Info")
    st.markdown(f"""
    - **File:** produksi_tanaman.csv
    - **Total Provinsi:** {len(df)}
    - **Total Komoditas:** {len(KOMODITAS_LIST)}
    - **Total Produksi:** {format_number(df['Total_Produksi'].sum())} ribu ton
    - **Format Kolom:** Provinsi, {', '.join([k.replace('_', ' ') for k in KOMODITAS_LIST])}
    """)
    
    st.subheader("🚀 Cara Deployment")
    st.markdown("""
    1. **GitHub**: Push kode ke repository GitHub
    2. **Streamlit Cloud**: Connect GitHub account ke https://share.streamlit.io
    3. **Deploy**: Pilih repo → branch → main file → Deploy
    4. **Share**: URL aplikasi akan tersedia dalam 3-5 menit
    """)
    
    st.subheader("📁 Struktur Repositori")
    st.code("""
dasbor-perkebunan/
├── app.py
├── produksi_tanaman.csv
├── requirements.txt
└── .streamlit/
    └── config.toml
    """, language='text')
    
    st.subheader("📜 Lisensi")
    st.markdown("""
    Proyek ini dibuat untuk keperluan akademik (Tugas UAS Visualisasi Data 2026).
    Dataset bersifat edukatif dan disesuaikan untuk keperluan pembelajaran.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div class="app-footer">
    <p class="footer-title">🌿 Dasbor Produksi Perkebunan Indonesia 🌾</p>
    <p>Dibuat dengan ❤️ menggunakan Streamlit, Plotly, dan scikit-learn</p>
    <p style="margin-top: 10px; font-size: 0.9em;">Tugas UAS Visualisasi Data | Sains Data & Pertanian Indonesia | © 2026</p>
    <p style="font-size: 0.85em; color: #a8d5ba;">Versi {APP_VERSION} | Last Updated: July 2026</p>
</div>
""", unsafe_allow_html=True)
