# ============================================================================
# app.py - DASBOR INTERAKTIF PRODUKSI TANAMAN PERKEBUNAN INDONESIA
# ============================================================================
# Proyek Tugas UAS Visualisasi Data
# Tema: Sains Data & Pertanian Indonesia
# Deskripsi: Aplikasi Streamlit untuk menganalisis produksi tanaman perkebunan
#            di 38 provinsi Indonesia dengan visualisasi 3D interaktif
# Versi: 1.0.0
# Author: Mahasiswa Sains Data
# Tanggal: 2026
# ============================================================================

# ============================================================================
# BAGIAN 1: IMPORT LIBRARY UTAMA
# ============================================================================
# Import library standar untuk manipulasi data, visualisasi, dan machine learning

import streamlit as st                    # Framework utama untuk membangun dasbor web interaktif
import pandas as pd                       # Library utama untuk manipulasi dan analisis data tabular
import numpy as np                        # Library untuk komputasi numerik dan array
import plotly.express as px               # Plotly Express untuk visualisasi cepat dan high-level
import plotly.graph_objects as go         # Plotly Graph Objects untuk visualisasi kustom dan 3D
from plotly.subplots import make_subplots # Membuat subplot multi-grafik dalam satu figure
import plotly.io as pio                   # Input/output untuk template plotly
import plotly.colors as pc               # Manipulasi warna plotly
import warnings                           # Mengatur peringatan Python
import os                                 # Operasi sistem file
import sys                                # Akses ke variabel dan fungsi interpreter Python
import io                                 # Operasi I/O untuk export data
import base64                             # Encoding untuk download link
import json                               # Parse dan dump data JSON
import math                               # Fungsi matematika dasar
from datetime import datetime, timedelta  # Manipulasi tanggal dan waktu
import hashlib                            # Hashing untuk cache key

# Import library machine learning dari scikit-learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)
from sklearn.cluster import KMeans        # Algoritma clustering K-Means
from sklearn.decomposition import PCA     # Principal Component Analysis untuk reduksi dimensi
from scipy import stats                   # Statistik ilmiah dari scipy
from scipy.stats import pearsonr, spearmanr  # Koefisien korelasi
import scipy.cluster.hierarchy as sch     # Clustering hierarki

# Konfigurasi peringatan: sembunyikan warning yang tidak kritis
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# BAGIAN 2: KONSTANTA GLOBAL DAN KONFIGURASI APLIKASI
# ============================================================================
# Mendefinisikan konstanta yang digunakan di seluruh aplikasi

# Versi aplikasi
APP_VERSION = "1.0.0"

# Nama aplikasi
APP_NAME = "Dasbor Produksi Perkebunan Indonesia"

# Ikon emoji untuk aplikasi
APP_ICON = "🌿"

# Tahun referensi data
DATA_YEAR = 2024

# Daftar nama kolom komoditas perkebunan utama
KOMODITAS_LIST = [
    'Kelapa_Sawit',   # Komoditas ekspor utama Indonesia
    'Kelapa',         # Komoditas tradisional
    'Karet',          # Komoditas perkebunan besar
    'Kopi',           # Komoditas ekspor penting
    'Kakao',          # Komoditas cokelat
    'Teh',            # Komoditas minuman
    'Tebu'            # Komoditas gula
]

# Label tampilan yang lebih ramah untuk komoditas
KOMODITAS_LABELS = {
    'Kelapa_Sawit': '🌴 Kelapa Sawit',
    'Kelapa': '🥥 Kelapa',
    'Karet': '🌳 Karet',
    'Kopi': '☕ Kopi',
    'Kakao': '🍫 Kakao',
    'Teh': '🍵 Teh',
    'Tebu': '🎋 Tebu'
}

# Warna tema untuk setiap komoditas (konsisten di seluruh aplikasi)
KOMODITAS_COLORS = {
    'Kelapa_Sawit': '#f39c12',  # Oranye (minyak sawit)
    'Kelapa': '#8d6e63',        # Coklat (tempurung kelapa)
    'Karet': '#5d4037',         # Coklat tua (getah karet)
    'Kopi': '#4e342e',          # Coklat espresso
    'Kakao': '#6d4c41',         # Coklat susu
    'Teh': '#2e7d32',           # Hijau teh
    'Tebu': '#9ccc65'           # Hijau muda tebu
}

# Palet warna regional untuk wilayah Indonesia
REGION_COLORS = {
    'Sumatera': '#2ecc71',              # Hijau emerald
    'Jawa': '#f1c40f',                  # Kuning emas
    'Bali & Nusa Tenggara': '#e67e22',  # Oranye
    'Kalimantan': '#27ae60',            # Hijau hutan
    'Sulawesi': '#16a085',              # Teal
    'Maluku': '#3498db',                # Biru laut
    'Papua': '#9b59b6'                  # Ungu
}

# ============================================================================
# BAGIAN 3: KOORDINAT GEOGRAFIS PROVINSI INDONESIA
# ============================================================================
# Koordinat centroid (lat, lon) untuk setiap provinsi, digunakan dalam peta

PROV_COORDS = {
    # Wilayah Sumatera
    'ACEH':                    (5.5483, 95.3238),   # Banda Aceh
    'SUMATERA UTARA':          (3.5952, 98.6722),   # Medan
    'SUMATERA BARAT':          (-0.7893, 100.3291), # Padang
    'RIAU':                    (1.0, 101.4474),     # Pekanbaru
    'JAMBI':                   (-1.6101, 103.6131), # Jambi
    'SUMATERA SELATAN':        (-3.3194, 104.9146), # Palembang
    'BENGKULU':                (-3.7926, 102.2614), # Bengkulu
    'LAMPUNG':                 (-5.3971, 105.2668), # Bandar Lampung
    'KEP. BANGKA BELITUNG':    (-2.7411, 106.4406), # Pangkal Pinang
    'KEP. RIAU':               (3.9453, 108.1428),  # Tanjung Pinang
    
    # Wilayah Jawa
    'DKI JAKARTA':             (-6.2088, 106.8456), # Jakarta
    'JAWA BARAT':              (-6.9175, 107.6191), # Bandung
    'JAWA TENGAH':             (-7.1509, 110.1403), # Semarang
    'DI YOGYAKARTA':           (-7.7956, 110.3695), # Yogyakarta
    'JAWA TIMUR':              (-7.5361, 112.2384), # Surabaya
    'BANTEN':                  (-6.4058, 106.0640), # Serang
    
    # Wilayah Bali & Nusa Tenggara
    'BALI':                    (-8.3405, 115.0920), # Denpasar
    'NUSA TENGGARA BARAT':     (-8.5833, 116.1167), # Mataram
    'NUSA TENGGARA TIMUR':     (-8.6574, 121.0794), # Kupang
    
    # Wilayah Kalimantan
    'KALIMANTAN BARAT':        (-0.0263, 109.3425), # Pontianak
    'KALIMANTAN TENGAH':       (-1.6815, 113.3824), # Palangkaraya
    'KALIMANTAN SELATAN':      (-3.3186, 114.5944), # Banjarmasin
    'KALIMANTAN TIMUR':        (1.2379, 116.8529),  # Samarinda
    'KALIMANTAN UTARA':        (3.0731, 116.0414),  # Tanjung Selor
    
    # Wilayah Sulawesi
    'SULAWESI UTARA':          (1.4748, 124.8421),  # Manado
    'SULAWESI TENGAH':         (-0.8950, 119.8376), # Palu
    'SULAWESI SELATAN':        (-3.6688, 119.9741), # Makassar
    'SULAWESI TENGGARA':       (-3.9563, 122.5097), # Kendari
    'GORONTALO':               (0.5417, 123.0594),  # Gorontalo
    'SULAWESI BARAT':          (-2.6748, 119.1051), # Mamuju
    
    # Wilayah Maluku
    'MALUKU':                  (-3.2385, 130.1453), # Ambon
    'MALUKU UTARA':            (1.5710, 127.7893),  # Sofifi
    
    # Wilayah Papua
    'PAPUA BARAT':             (-1.3361, 132.4085), # Manokwari
    'PAPUA BARAT DAYA':        (-1.5897, 131.2011), # Sorong
    'PAPUA':                   (-2.5916, 140.6690), # Jayapura
    'PAPUA SELATAN':           (-7.4833, 140.7500), # Merauke
    'PAPUA TENGAH':            (-3.3171, 137.3811), # Nabire
    'PAPUA PEGUNUNGAN':        (-4.0433, 138.9611)  # Wamena
}

# ============================================================================
# BAGIAN 4: KONFIGURASI HALAMAN STREAMLIT
# ============================================================================
# Mengatur konfigurasi dasar halaman Streamlit

st.set_page_config(
    page_title=f"{APP_ICON} {APP_NAME}",      # Judul tab browser
    page_icon=APP_ICON,                        # Favicon
    layout="wide",                             # Layout lebar penuh
    initial_sidebar_state="expanded",          # Sidebar terbuka saat load
    menu_items={
        'Get Help': 'https://streamlit.io/',   # Link bantuan
        'Report a bug': 'https://github.com/', # Link pelaporan bug
        'About': f"### {APP_NAME}\nVersi {APP_VERSION}\n\nDasbor analisis produksi tanaman perkebunan Indonesia dengan visualisasi 3D interaktif untuk Tugas UAS Visualisasi Data."
    }
)

# ============================================================================
# BAGIAN 5: CSS KUSTOM DAN STYLING (DESAIN MODERN AGRIKULTUR)
# ============================================================================
# Mendefinisikan CSS lengkap untuk tampilan modern dengan tema dark mode

CUSTOM_CSS = """
<style>
    /* ============================================ */
    /* IMPOR FONT DARI CDN FONTSOURCE               */
    /* ============================================ */
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/poppins@5.0.0/index.min.css');
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/inter@5.0.0/index.min.css');
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.0.0/index.min.css');
    
    /* ============================================ */
    /* VARIABEL WARNA TEMA (DESIGN TOKENS)          */
    /* ============================================ */
    :root {
        --bg-primary: #0a1f14;           /* Latar utama gelap */
        --bg-secondary: #0f2a1c;         /* Latar sekunder */
        --bg-tertiary: #1a3d2e;          /* Latar tersier */
        --emerald-dark: #0d3320;         /* Emerald gelap */
        --emerald: #2ecc71;              /* Emerald primer */
        --emerald-light: #58d68d;        /* Emerald terang */
        --gold: #f1c40f;                 /* Emas padi */
        --gold-light: #f7dc6f;           /* Emas terang */
        --text-primary: #e8f5e9;         /* Teks primer */
        --text-secondary: #a8d5ba;       /* Teks sekunder */
        --text-muted: #6b8f7a;           /* Teks redup */
        --border: rgba(46, 204, 113, 0.3); /* Border */
        --shadow-emerald: rgba(46, 204, 113, 0.2);
        --shadow-gold: rgba(241, 196, 15, 0.2);
    }
    
    /* ============================================ */
    /* LAYOUT UTAMA APLIKASI                        */
    /* ============================================ */
    .stApp {
        background: linear-gradient(
            135deg, 
            var(--bg-primary) 0%, 
            var(--bg-secondary) 40%, 
            #1a1a2e 100%
        );
        color: var(--text-primary);
        font-family: 'Inter', 'Poppins', sans-serif;
        min-height: 100vh;
    }
    
    /* Main container dengan padding */
    .main .block-container {
        padding: 2rem 3rem 4rem 3rem;
        max-width: 1600px;
    }
    
    /* ============================================ */
    /* SIDEBAR STYLING                              */
    /* ============================================ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, 
            var(--emerald-dark) 0%, 
            var(--emerald-dark) 50%,
            #0a2818 100%
        );
        color: var(--text-primary);
        border-right: 2px solid var(--emerald);
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        color: var(--emerald-light);
        font-weight: 600;
        font-size: 1.1em;
    }
    
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 0.5rem;
    }
    
    section[data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(46, 204, 113, 0.1);
        border-radius: 10px;
        padding: 12px 15px;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        color: var(--text-secondary);
    }
    
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(46, 204, 113, 0.25);
        border-color: var(--emerald);
        transform: translateX(5px);
    }
    
    section[data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(90deg, rgba(46, 204, 113, 0.3), rgba(241, 196, 15, 0.15));
        border-color: var(--gold);
        color: var(--gold);
        font-weight: 700;
    }
    
    /* ============================================ */
    /* JUDUL UTAMA (MAIN TITLE)                     */
    /* ============================================ */
    .main-title {
        background: linear-gradient(
            90deg, 
            #2ecc71 0%, 
            #58d68d 25%,
            #f1c40f 50%, 
            #f7dc6f 75%,
            #27ae60 100%
        );
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.2em;
        font-weight: 900;
        text-align: center;
        margin-bottom: 10px;
        letter-spacing: -1px;
        animation: shimmer 3s linear infinite;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    .sub-title {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.15em;
        margin-bottom: 30px;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* ============================================ */
    /* HEADER HALAMAN (PAGE HEADER)                 */
    /* ============================================ */
    .page-header {
        background: linear-gradient(
            135deg, 
            rgba(46, 204, 113, 0.15) 0%, 
            rgba(241, 196, 15, 0.08) 100%
        );
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 25px 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px var(--shadow-emerald);
        backdrop-filter: blur(10px);
    }
    
    .page-header h1 {
        color: var(--emerald-light);
        margin: 0;
        font-size: 2em;
        font-weight: 700;
    }
    
    .page-header p {
        color: var(--text-secondary);
        margin-top: 8px;
        font-size: 1em;
    }
    
    /* ============================================ */
    /* KARTU METRIK KPI                             */
    /* ============================================ */
    .metric-card {
        background: linear-gradient(
            135deg, 
            rgba(46, 204, 113, 0.18) 0%, 
            rgba(241, 196, 15, 0.12) 100%
        );
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 10px 40px var(--shadow-emerald);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--emerald), transparent);
        transition: left 0.6s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 50px var(--shadow-emerald);
        border-color: var(--emerald);
    }
    
    .metric-icon {
        font-size: 2em;
        margin-bottom: 8px;
        filter: drop-shadow(0 0 10px var(--shadow-gold));
    }
    
    .metric-value {
        font-size: 1.9em;
        font-weight: 800;
        color: var(--gold);
        line-height: 1;
        margin-bottom: 8px;
        font-family: 'JetBrains Mono', monospace;
        text-shadow: 0 0 20px var(--shadow-gold);
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.82em;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 500;
    }
    
    .metric-change {
        font-size: 0.85em;
        margin-top: 8px;
        padding: 3px 10px;
        border-radius: 20px;
        display: inline-block;
    }
    
    .metric-change.positive {
        background: rgba(46, 204, 113, 0.2);
        color: var(--emerald-light);
    }
    
    .metric-change.negative {
        background: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
    }
    
    /* ============================================ */
    /* KOTAK INSIGHT & REKOMENDASI                  */
    /* ============================================ */
    .insight-box {
        background: linear-gradient(
            135deg, 
            rgba(46, 204, 113, 0.12) 0%, 
            rgba(46, 204, 113, 0.05) 100%
        );
        border-left: 6px solid var(--emerald);
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .insight-box:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 25px var(--shadow-emerald);
    }
    
    .recommend-box {
        background: linear-gradient(
            135deg, 
            rgba(241, 196, 15, 0.12) 0%, 
            rgba(241, 196, 15, 0.05) 100%
        );
        border-left: 6px solid var(--gold);
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .recommend-box:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 25px var(--shadow-gold);
    }
    
    .warning-box {
        background: linear-gradient(
            135deg, 
            rgba(230, 126, 34, 0.15) 0%, 
            rgba(230, 126, 34, 0.05) 100%
        );
        border-left: 6px solid #e67e22;
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .info-box {
        background: linear-gradient(
            135deg, 
            rgba(52, 152, 219, 0.15) 0%, 
            rgba(52, 152, 219, 0.05) 100%
        );
        border-left: 6px solid #3498db;
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    /* ============================================ */
    /* TYPOGRAPHY (H1, H2, H3)                      */
    /* ============================================ */
    h1, h2, h3 {
        color: var(--emerald-light) !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    
    h1 {
        border-bottom: 3px solid var(--emerald);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    h2 {
        border-left: 5px solid var(--gold);
        padding-left: 15px;
        margin-top: 30px;
    }
    
    h3 {
        color: var(--gold-light) !important;
        margin-top: 25px;
    }
    
    /* ============================================ */
    /* TABEL DATAFRAME STYLING                      */
    /* ============================================ */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* ============================================ */
    /* BUTTON STYLING                               */
    /* ============================================ */
    .stButton > button {
        background: linear-gradient(
            135deg, 
            var(--emerald-dark) 0%, 
            var(--emerald) 100%
        );
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px var(--shadow-emerald);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px var(--shadow-emerald);
        background: linear-gradient(
            135deg, 
            var(--emerald) 0%, 
            var(--gold) 100%
        );
    }
    
    /* ============================================ */
    /* SELECTBOX DAN INPUT                          */
    /* ============================================ */
    .stSelectbox > div > div {
        background-color: rgba(15, 42, 28, 0.8) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    
    /* ============================================ */
    /* DIVIDER (PEMISAH)                            */
    /* ============================================ */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(
            90deg, 
            transparent, 
            var(--emerald), 
            var(--gold), 
            var(--emerald), 
            transparent
        );
        margin: 30px 0;
    }
    
    /* ============================================ */
    /* TAB STYLING                                  */
    /* ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 42, 28, 0.5);
        border-radius: 12px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(
            135deg, 
            var(--emerald-dark) 0%, 
            var(--emerald) 100%
        );
        color: white;
        font-weight: 600;
    }
    
    /* ============================================ */
    /* EXPANDER STYLING                             */
    /* ============================================ */
    .streamlit-expanderHeader {
        background: rgba(15, 42, 28, 0.5);
        border-radius: 10px;
        color: var(--emerald-light);
        font-weight: 600;
    }
    
    /* ============================================ */
    /* FOOTER APLIKASI                              */
    /* ============================================ */
    .app-footer {
        background: linear-gradient(
            90deg, 
            rgba(13, 51, 32, 0.8) 0%, 
            rgba(15, 42, 28, 0.8) 50%, 
            rgba(13, 51, 32, 0.8) 100%
        );
        border-top: 2px solid var(--emerald);
        padding: 25px;
        text-align: center;
        margin-top: 50px;
        border-radius: 15px 15px 0 0;
        box-shadow: 0 -5px 30px var(--shadow-emerald);
    }
    
    .app-footer p {
        color: var(--text-secondary);
        margin: 5px 0;
    }
    
    .app-footer .footer-title {
        color: var(--emerald-light);
        font-weight: 700;
        font-size: 1.1em;
    }
    
    /* ============================================ */
    /* BADGE / LABEL STATUS                         */
    /* ============================================ */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 2px;
    }
    
    .badge-success {
        background: rgba(46, 204, 113, 0.2);
        color: var(--emerald-light);
        border: 1px solid var(--emerald);
    }
    
    .badge-warning {
        background: rgba(241, 196, 15, 0.2);
        color: var(--gold);
        border: 1px solid var(--gold);
    }
    
    .badge-error {
        background: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
    
    /* ============================================ */
    /* CHART CONTAINER                              */
    /* ============================================ */
    .chart-container {
        background: rgba(10, 31, 20, 0.4);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 5px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* ============================================ */
    /* PROGRESS BAR KUSTOM                          */
    /* ============================================ */
    .custom-progress {
        background: rgba(15, 42, 28, 0.5);
        border-radius: 10px;
        height: 30px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .custom-progress-bar {
        height: 100%;
        background: linear-gradient(
            90deg, 
            var(--emerald) 0%, 
            var(--gold) 100%
        );
        border-radius: 10px;
        transition: width 1s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
    }
    
    /* ============================================ */
    /* SCROLLBAR KUSTOM                             */
    /* ============================================ */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--emerald-dark);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--emerald);
    }
    
    /* ============================================ */
    /* ANIMASI ENTRANCE                             */
    /* ============================================ */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card, .insight-box, .recommend-box {
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* ============================================ */
    /* RESPONSIVE DESIGN (Mobile)                   */
    /* ============================================ */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2em;
        }
        .main .block-container {
            padding: 1rem;
        }
        .metric-value {
            font-size: 1.4em;
        }
    }
</style>
"""

# Render CSS kustom ke halaman Streamlit
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# BAGIAN 6: FUNGSI-FUNGSI HELPER (UTILITY FUNCTIONS)
# ============================================================================
# Kumpulan fungsi bantuan yang digunakan di berbagai bagian aplikasi

def load_data(file_path="produksi_tanaman.csv"):
    """
    Membaca data CSV dengan penanganan error yang komprehensif.
    
    Parameters:
    -----------
    file_path : str
        Path ke file CSV yang akan dibaca
        
    Returns:
    --------
    tuple : (DataFrame, error_message)
        DataFrame jika berhasil, None jika gagal
        error_message berisi pesan error jika gagal, None jika sukses
    """
    try:
        # Cek apakah file ada
        if not os.path.exists(file_path):
            return None, f"❌ File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang sama dengan app.py"
        
        # Baca file CSV
        df = pd.read_csv(file_path)
        
        # Validasi: cek apakah dataframe kosong
        if df.empty:
            return None, "❌ File CSV kosong (tidak ada data)."
        
        # Validasi: cek apakah kolom Provinsi ada
        if 'Provinsi' not in df.columns:
            return None, "❌ Kolom 'Provinsi' tidak ditemukan dalam dataset."
        
        # Validasi: cek apakah kolom komoditas lengkap
        missing_cols = [col for col in KOMODITAS_LIST if col not in df.columns]
        if missing_cols:
            return None, f"❌ Kolom komoditas tidak lengkap. Kolom yang hilang: {', '.join(missing_cols)}"
        
        return df, None
        
    except pd.errors.EmptyDataError:
        return None, "❌ File CSV kosong atau tidak dapat dibaca."
    except pd.errors.ParserError as e:
        return None, f"❌ Error parsing CSV: {str(e)}"
    except UnicodeDecodeError:
        # Coba dengan encoding lain
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
            return df, None
        except Exception as e:
            return None, f"❌ Error encoding: {str(e)}"
    except Exception as e:
        return None, f"❌ Terjadi error tidak terduga: {str(e)}"


def assign_region(provinsi):
    """
    Mengelompokkan provinsi ke dalam wilayah geografis Indonesia.
    
    Parameters:
    -----------
    provinsi : str
        Nama provinsi
        
    Returns:
    --------
    str : Nama wilayah (Sumatera, Jawa, Kalimantan, dll)
    """
    # Normalisasi nama provinsi ke uppercase
    prov_upper = str(provinsi).upper().strip()
    
    # Dictionary pemetaan wilayah
    region_mapping = {
        'Sumatera': [
            'ACEH', 'SUMATERA', 'RIAU', 'JAMBI', 'BENGKULU', 
            'LAMPUNG', 'BANGKA', 'KEP. RIAU'
        ],
        'Jawa': [
            'DKI', 'JAWA', 'YOGYAKARTA', 'BANTEN', 'JAKARTA'
        ],
        'Bali & Nusa Tenggara': [
            'BALI', 'NUSA TENGGARA'
        ],
        'Kalimantan': [
            'KALIMANTAN'
        ],
        'Sulawesi': [
            'SULAWESI', 'GORONTALO'
        ],
        'Maluku': [
            'MALUKU'
        ],
        'Papua': [
            'PAPUA'
        ]
    }
    
    # Loop untuk mencari wilayah yang sesuai
    for region, keywords in region_mapping.items():
        for keyword in keywords:
            if keyword in prov_upper:
                return region
    
    # Default jika tidak cocok
    return 'Lainnya'


def format_number(num, decimals=0):
    """
    Format angka dengan pemisah ribuan (titik) dan desimal (koma).
    
    Parameters:
    -----------
    num : float/int
        Angka yang akan diformat
    decimals : int
        Jumlah digit desimal
        
    Returns:
    --------
    str : String angka yang sudah diformat
    """
    try:
        # Handle nilai NaN atau None
        if pd.isna(num):
            return "N/A"
        
        # Format dengan ribuan separator
        if decimals == 0:
            return f"{int(num):,}".replace(',', '.')
        else:
            return f"{num:,.{decimals}f}".replace(',', '#').replace('.', ',').replace('#', '.')
    except Exception:
        return "N/A"


def format_large_number(num):
    """
    Format angka besar dengan suffix (K, M, B, T).
    
    Parameters:
    -----------
    num : float/int
        Angka yang akan diformat
        
    Returns:
    --------
    str : String angka dengan suffix
    """
    try:
        if pd.isna(num):
            return "N/A"
        
        abs_num = abs(num)
        sign = '-' if num < 0 else ''
        
        if abs_num >= 1e12:
            return f"{sign}{abs_num/1e12:.2f}T"
        elif abs_num >= 1e9:
            return f"{sign}{abs_num/1e9:.2f}B"
        elif abs_num >= 1e6:
            return f"{sign}{abs_num/1e6:.2f}M"
        elif abs_num >= 1e3:
            return f"{sign}{abs_num/1e3:.2f}K"
        else:
            return f"{sign}{abs_num:.2f}"
    except Exception:
        return "N/A"


def detect_outliers_iqr(series, kolom_name=""):
    """
    Deteksi outlier menggunakan metode IQR (Interquartile Range).
    
    Parameters:
    -----------
    series : pd.Series
        Series data yang akan dianalisis
    kolom_name : str
        Nama kolom untuk logging
        
    Returns:
    --------
    dict : Dictionary berisi informasi outlier
    """
    # Hitung kuartil
    Q1 = series.quantile(0.25)      # Kuartil pertama (25%)
    Q2 = series.quantile(0.50)      # Median (50%)
    Q3 = series.quantile(0.75)      # Kuartil ketiga (75%)
    
    # Hitung IQR
    IQR = Q3 - Q1
    
    # Batas bawah dan atas
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identifikasi outlier
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    
    # Hasil analisis
    result = {
        'kolom': kolom_name,
        'Q1': Q1,
        'Q2': Q2,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_count': len(outliers),
        'outlier_percentage': (len(outliers) / len(series)) * 100,
        'outlier_indices': outliers.index.tolist()
    }
    
    return result


def calculate_skewness_kurtosis(series):
    """
    Hitung skewness (kemencengan) dan kurtosis (keruncingan).
    
    Parameters:
    -----------
    series : pd.Series
        Series data numerik
        
    Returns:
    --------
    dict : Dictionary berisi skewness dan kurtosis
    """
    try:
        skew = series.skew()
        kurt = series.kurtosis()
        
        # Interpretasi skewness
        if abs(skew) < 0.5:
            skew_interpretation = "Simetris"
        elif skew > 0.5:
            skew_interpretation = "Mencondong Kanan (Positif)"
        else:
            skew_interpretation = "Mencondong Kiri (Negatif)"
        
        # Interpretasi kurtosis
        if abs(kurt) < 0.5:
            kurt_interpretation = "Mesokurtik (Normal)"
        elif kurt > 0.5:
            kurt_interpretation = "Leptokurtik (Lancip)"
        else:
            kurt_interpretation = "Platikurtik (Datar)"
        
        return {
            'skewness': skew,
            'kurtosis': kurt,
            'skewness_interpretation': skew_interpretation,
            'kurtosis_interpretation': kurt_interpretation
        }
    except Exception:
        return {
            'skewness': np.nan,
            'kurtosis': np.nan,
            'skewness_interpretation': "Error",
            'kurtosis_interpretation': "Error"
        }


def calculate_correlation_significance(x, y, alpha=0.05):
    """
    Hitung korelasi Pearson dan signifikansi statistiknya.
    
    Parameters:
    -----------
    x, y : array-like
        Dua variabel yang akan dikorelasikan
    alpha : float
        Tingkat signifikansi (default 0.05)
        
    Returns:
    --------
    dict : Dictionary berisi koefisien korelasi dan p-value
    """
    try:
        # Hitung korelasi Pearson
        pearson_r, pearson_p = pearsonr(x, y)
        
        # Hitung korelasi Spearman (non-parametrik)
        spearman_r, spearman_p = spearmanr(x, y)
        
        # Tentukan signifikansi
        pearson_significant = pearson_p < alpha
        spearman_significant = spearman_p < alpha
        
        # Interpretasi kekuatan korelasi
        def interpret_strength(r):
            abs_r = abs(r)
            if abs_r < 0.3:
                return "Lemah"
            elif abs_r < 0.5:
                return "Sedang"
            elif abs_r < 0.7:
                return "Kuat"
            elif abs_r < 0.9:
                return "Sangat Kuat"
            else:
                return "Hampir Sempurna"
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'pearson_significant': pearson_significant,
            'pearson_strength': interpret_strength(pearson_r),
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'spearman_significant': spearman_significant,
            'spearman_strength': interpret_strength(spearman_r)
        }
    except Exception as e:
        return {
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'pearson_significant': False,
            'spearman_r': np.nan,
            'spearman_p': np.nan,
            'spearman_significant': False,
            'error': str(e)
        }


def get_top_n_provinces(df, kolom, n=10, ascending=False):
    """
    Ambil N provinsi dengan nilai tertinggi/terendah.
    
    Parameters:
    -----------
    df : DataFrame
        Data utama
    kolom : str
        Kolom untuk sorting
    n : int
        Jumlah provinsi yang diambil
    ascending : bool
        Jika True, ambil nilai terendah
        
    Returns:
    --------
    DataFrame : N provinsi terurut
    """
    return df.nsmallest(n, kolom) if ascending else df.nlargest(n, kolom)


def create_metric_card(icon, value, label, change=None, change_type="positive"):
    """
    Buat HTML untuk kartu metrik KPI.
    
    Parameters:
    -----------
    icon : str
        Emoji/icon
    value : str
        Nilai metrik
    label : str
        Label metrik
    change : str, optional
        Perubahan (misal "+5%")
    change_type : str
        "positive" atau "negative"
        
    Returns:
    --------
    str : HTML kartu metrik
    """
    change_html = ""
    if change:
        change_html = f'<div class="metric-change {change_type}">{change}</div>'
    
    return f'''
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {change_html}
    </div>
    '''


def create_insight_box(title, content, icon="🔍"):
    """
    Buat HTML untuk kotak insight.
    
    Parameters:
    -----------
    title : str
        Judul insight
    content : str
        Konten insight
    icon : str
        Emoji icon
        
    Returns:
    --------
    str : HTML insight box
    """
    return f'''
    <div class="insight-box">
        <strong>{icon} {title}</strong>
        <p style="margin:8px 0 0 0; color: var(--text-secondary);">{content}</p>
    </div>
    '''


def create_recommend_box(number, title, content):
    """
    Buat HTML untuk kotak rekomendasi.
    
    Parameters:
    -----------
    number : int
        Nomor rekomendasi
    title : str
        Judul rekomendasi
    content : str
        Konten rekomendasi
        
    Returns:
    --------
    str : HTML recommend box
    """
    return f'''
    <div class="recommend-box">
        <strong>🎯 Rekomendasi #{number}: {title}</strong>
        <p style="margin:8px 0 0 0; color: var(--text-secondary);">{content}</p>
    </div>
    '''


def apply_plotly_theme(fig, title=None, height=600):
    """
    Terapkan tema kustom pada figure Plotly.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure yang akan distyling
    title : str, optional
        Judul figure
    height : int
        Tinggi figure dalam pixel
        
    Returns:
    --------
    plotly.graph_objects.Figure : Figure yang sudah distyling
    """
    # Layout umum
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",       # Transparan
        plot_bgcolor="rgba(10, 31, 20, 0.3)",    # Latar plot gelap
        font=dict(
            family="Inter, sans-serif",
            color="#e8f5e9",                      # Warna teks terang
            size=13
        ),
        title=dict(
            text=title if title else "",
            font=dict(size=20, color="#58d68d", family="Poppins"),
            x=0.5,                                 # Center
            xanchor="center"
        ),
        height=height,
        margin=dict(l=50, r=50, t=60, b=50),
        hoverlabel=dict(
            bgcolor="#0f2a1c",
            font_size=14,
            font_family="Inter",
            bordercolor="#2ecc71"
        ),
        legend=dict(
            bgcolor="rgba(15, 42, 28, 0.8)",
            bordercolor="#2ecc71",
            borderwidth=1,
            font=dict(color="#e8f5e9")
        )
    )
    
    # Jika figure memiliki scene 3D
    if hasattr(fig, 'layout') and hasattr(fig.layout, 'scene'):
        try:
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        backgroundcolor="rgb(10, 31, 20)",
                        gridcolor="rgba(46, 204, 113, 0.2)",
                        zerolinecolor="rgba(46, 204, 113, 0.3)",
                        color="#e8f5e9",
                        title_font=dict(color="#2ecc71", size=14)
                    ),
                    yaxis=dict(
                        backgroundcolor="rgb(10, 31, 20)",
                        gridcolor="rgba(46, 204, 113, 0.2)",
                        zerolinecolor="rgba(46, 204, 113, 0.3)",
                        color="#e8f5e9",
                        title_font=dict(color="#2ecc71", size=14)
                    ),
                    zaxis=dict(
                        backgroundcolor="rgb(10, 31, 20)",
                        gridcolor="rgba(46, 204, 113, 0.2)",
                        zerolinecolor="rgba(46, 204, 113, 0.3)",
                        color="#e8f5e9",
                        title_font=dict(color="#f1c40f", size=14)
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    )
                )
            )
        except Exception:
            pass
    
    return fig


def export_dataframe_download_link(df, filename="data_export.csv", label="📥 Download CSV"):
    """
    Buat link download untuk DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        Data yang akan di-export
    filename : str
        Nama file output
    label : str
        Label tombol
        
    Returns:
    --------
    str : HTML download link
    """
    # Convert ke CSV
    csv = df.to_csv(index=False)
    
    # Encode ke base64
    b64 = base64.b64encode(csv.encode()).decode()
    
    # Buat link HTML
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="stButton"><button>{label}</button></a>'
    
    return href


def calculate_gini_coefficient(series):
    """
    Hitung koefisien Gini untuk mengukur ketimpangan distribusi.
    
    Parameters:
    -----------
    series : pd.Series
        Data numerik
        
    Returns:
    --------
    float : Koefisien Gini (0 = sempurna merata, 1 = sempurna timpang)
    """
    try:
        # Sort dan reset index
        sorted_values = series.sort_values().reset_index(drop=True)
        n = len(sorted_values)
        
        # Handle kasus semua nilai 0
        if sorted_values.sum() == 0:
            return 0.0
        
        # Hitung Gini
        cumsum = sorted_values.cumsum()
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values)) / (n * cumsum.iloc[-1])) - (n + 1) / n
        
        return max(0, min(1, gini))  # Clamp antara 0 dan 1
    except Exception:
        return np.nan


def calculate_herfindahl_index(df, komoditas_columns):
    """
    Hitung Herfindahl-Hirschman Index (HHI) untuk diversifikasi.
    
    Parameters:
    -----------
    df : DataFrame
        Data produksi
    komoditas_columns : list
        Daftar kolom komoditas
        
    Returns:
    --------
    dict : HHI untuk setiap provinsi
    """
    hhi_results = {}
    
    for idx, row in df.iterrows():
        provinsi = row['Provinsi']
        values = [row[col] for col in komoditas_columns]
        total = sum(values)
        
        if total > 0:
            # Hitung pangsa pasar setiap komoditas
            shares = [v / total for v in values]
            # HHI = sum of squared shares
            hhi = sum(s ** 2 for s in shares)
            hhi_results[provinsi] = hhi
        else:
            hhi_results[provinsi] = 0.0
    
    return hhi_results


def classify_diversification(hhi):
    """
    Klasifikasikan tingkat diversifikasi berdasarkan HHI.
    
    Parameters:
    -----------
    hhi : float
        Herfindahl-Hirschman Index
        
    Returns:
    --------
    str : Kategori diversifikasi
    """
    if hhi < 0.15:
        return "🌈 Sangat Diversifikasi"
    elif hhi < 0.25:
        return "🎨 Moderat Diversifikasi"
    elif hhi < 0.40:
        return "⚖️ Kurang Diversifikasi"
    else:
        return "🎯 Sangat Spesialisasi"


# ============================================================================
# BAGIAN 7: MEMUAT DATA DAN PREPROCESSING AWAL
# ============================================================================
# Load data dari file CSV dan lakukan preprocessing awal

# Gunakan cache untuk menghindari loading berulang
@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """
    Load data dan lakukan preprocessing awal.
    
    Returns:
    --------
    tuple : (DataFrame asli, DataFrame terpreprocessing, error_message)
    """
    # Load data mentah
    df_raw, error = load_data()
    
    if error:
        return None, None, error
    
    # Buat salinan untuk preprocessing
    df = df_raw.copy()
    
    # Tambahkan kolom Wilayah
    df['Wilayah'] = df['Provinsi'].apply(assign_region)
    
    # Tambahkan kolom Total_Produksi
    df['Total_Produksi'] = df[KOMODITAS_LIST].sum(axis=1)
    
    # Tambahkan koordinat geografis
    df['Latitude'] = df['Provinsi'].map(lambda p: PROV_COORDS.get(p, (np.nan, np.nan))[0])
    df['Longitude'] = df['Provinsi'].map(lambda p: PROV_COORDS.get(p, (np.nan, np.nan))[1])
    
    # Tambahkan kolom untuk ranking
    df['Rank_Total'] = df['Total_Produksi'].rank(ascending=False, method='min').astype(int)
    
    # Tambahkan kolom komoditas dominan
    def get_dominant_commodity(row):
        """Mendapatkan komoditas dengan produksi tertinggi."""
        values = {k: row[k] for k in KOMODITAS_LIST}
        max_key = max(values, key=values.get)
        return KOMODITAS_LABELS.get(max_key, max_key)
    
    df['Komoditas_Dominan'] = df.apply(get_dominant_commodity, axis=1)
    
    # Hitung HHI untuk diversifikasi
    hhi_dict = calculate_herfindahl_index(df, KOMODITAS_LIST)
    df['HHI_Index'] = df['Provinsi'].map(hhi_dict)
    df['Diversifikasi'] = df['HHI_Index'].apply(classify_diversification)
    
    return df_raw, df, None


# Load data
df_raw, df, data_error = load_and_preprocess_data()

# Tampilkan error jika ada
if data_error:
    st.error(data_error)
    st.info("""
    **📋 Panduan Penyelesaian:**
    1. Pastikan file `produksi_tanaman.csv` berada di folder yang sama dengan `app.py`
    2. Format CSV harus memiliki kolom: Provinsi, Kelapa_Sawit, Kelapa, Karet, Kopi, Kakao, Teh, Tebu
    3. Jalankan ulang aplikasi dengan perintah: `streamlit run app.py`
    """)
    st.stop()

# ============================================================================
# BAGIAN 8: SIDEBAR NAVIGASI
# ============================================================================
# Membuat sidebar dengan navigasi antar halaman

# Header sidebar
with st.sidebar:
    # Logo/branding sidebar
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; border-bottom: 2px solid rgba(46, 204, 113, 0.3);">
        <div style="font-size: 3em;">🌿🌾🌴</div>
        <div style="color: #58d68d; font-size: 1.1em; font-weight: 700; margin-top: 10px;">
            DASHBOARD<br>PERKEBUNAN
        </div>
        <div style="color: #f7dc6f; font-size: 0.85em; margin-top: 5px;">
            Indonesia 🇮🇩
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Spacer
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Label navigasi
    st.markdown("### 🗺️ Navigasi Halaman")
    
    # Radio button untuk memilih halaman
    page = st.radio(
        "Pilih halaman yang ingin Anda kunjungi:",
        [
            "📊 Page 1: Overview & Data Understanding",
            "🧹 Page 2: Data Cleaning & Preprocessing",
            "📈 Page 3: EDA & 3D Visualizations",
            "🗺️ Page 3b: Peta Distribusi Indonesia",
            "🔗 Page 4: Analisis Korelasi & Regresi",
            "🎯 Page 4b: Machine Learning Lanjutan",
            "💡 Page 5: Insights & Rekomendasi",
            "📚 Tentang Aplikasi"
        ],
        label_visibility="collapsed"
    )
    
    # Info data di sidebar
    st.markdown("---")
    st.markdown("### 📊 Info Dataset")
    st.markdown(f"""
    - **Total Provinsi:** {len(df)}
    - **Total Komoditas:** {len(KOMODITAS_LIST)}
    - **Total Produksi:** {format_number(df['Total_Produksi'].sum())}
    """)
    
    # Versi aplikasi
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b8f7a; font-size: 0.85em;">
        <p>🌿 Versi {APP_VERSION}</p>
        <p>UAS Visualisasi Data 2026</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# BAGIAN 9: HEADER UTAMA APLIKASI
# ============================================================================
# Menampilkan header dengan judul dan subjudul

st.markdown(f'<h1 class="main-title">🌿 Dasbor Produksi Perkebunan Indonesia</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analisis Visual 3D Interaktif • Sektor Agrikultur Modern • Tugas UAS Visualisasi Data</p>', unsafe_allow_html=True)

# ============================================================================
# BAGIAN 10: PAGE 1 - OVERVIEW & DATA UNDERSTANDING
# ============================================================================
# Halaman pertama: gambaran umum dataset dan metrik utama

if "Page 1: Overview" in page:
    # Header halaman
    st.markdown("""
    <div class="page-header">
        <h1>📊 Overview & Data Understanding</h1>
        <p>Memahami struktur, konten, dan karakteristik dataset produksi tanaman perkebunan Indonesia</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Deskripsi dataset
    st.markdown("""
    Dataset ini berisi informasi mengenai **produksi 7 komoditas perkebunan utama** di **38 provinsi** 
    Indonesia. Komoditas yang tercakup meliputi Kelapa Sawit, Kelapa, Karet, Kopi, Kakao, Teh, dan Tebu.
    Data ini memberikan gambaran komprehensif tentang distribusi dan kontribusi setiap provinsi terhadap 
    sektor perkebunan nasional.
    """)
    
    st.markdown("---")
    
    # ========================================
    # KPI UTAMA (KEY PERFORMANCE INDICATORS)
    # ========================================
    st.subheader("🏆 Key Performance Indicators (KPI)")
    st.markdown("Metrik-metrik kunci yang menggambarkan kondisi sektor perkebunan Indonesia secara agregat.")
    
    # Hitung metrik utama
    total_semua = df[KOMODITAS_LIST].sum().sum()
    total_sawit = df['Kelapa_Sawit'].sum()
    total_karet = df['Karet'].sum()
    total_kopi = df['Kopi'].sum()
    total_kakao = df['Kakao'].sum()
    total_kelapa = df['Kelapa'].sum()
    total_teh = df['Teh'].sum()
    total_tebu = df['Tebu'].sum()
    
    # Hitung persentase kontribusi setiap komoditas
    pct_sawit = (total_sawit / total_semua) * 100 if total_semua > 0 else 0
    pct_karet = (total_karet / total_semua) * 100 if total_semua > 0 else 0
    pct_kopi = (total_kopi / total_semua) * 100 if total_semua > 0 else 0
    
    # Tampilkan KPI dalam grid 4 kolom
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            icon="🌾",
            value=format_number(total_semua),
            label="Total Produksi Semua Komoditas",
            change="ribu ton",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            icon="🌴",
            value=format_number(total_sawit),
            label="Total Kelapa Sawit",
            change=f"{pct_sawit:.1f}% dari total",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            icon="🌳",
            value=format_number(total_karet),
            label="Total Karet",
            change=f"{pct_karet:.1f}% dari total",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card(
            icon="☕",
            value=format_number(total_kopi),
            label="Total Kopi",
            change=f"{pct_kopi:.1f}% dari total",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    # KPI baris kedua
    st.markdown("<br>", unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown(create_metric_card(
            icon="🥥",
            value=format_number(total_kelapa),
            label="Total Kelapa",
            change="ribu ton",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    with col6:
        st.markdown(create_metric_card(
            icon="🍫",
            value=format_number(total_kakao),
            label="Total Kakao",
            change="ribu ton",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    with col7:
        st.markdown(create_metric_card(
            icon="🍵",
            value=format_number(total_teh),
            label="Total Teh",
            change="ribu ton",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    with col8:
        st.markdown(create_metric_card(
            icon="🎋",
            value=format_number(total_tebu),
            label="Total Tebu",
            change="ribu ton",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================
    # CUPlikan DATA MENTAH
    # ========================================
    st.subheader("📋 Cuplikan Data Mentah (Raw Data Preview)")
    st.markdown("Berikut adalah 15 baris pertama dari dataset untuk memberikan gambaran awal tentang struktur data.")
    
    # Pilihan jumlah baris preview
    n_rows = st.slider("Jumlah baris yang ditampilkan:", min_value=5, max_value=38, value=15, step=1)
    
    # Tampilkan data
    st.dataframe(
        df.head(n_rows).style
        .background_gradient(subset=KOMODITAS_LIST, cmap='Greens')
        .format({col: '{:,.2f}' for col in KOMODITAS_LIST + ['Total_Produksi', 'HHI_Index']}),
        use_container_width=True,
        height=500
    )
    
    # Tombol download data
    st.markdown("### 📥 Download Dataset")
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="💾 Download Data Lengkap (CSV)",
        data=csv_data,
        file_name="produksi_tanaman_processed.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # ========================================
    # STRUKTUR TIPE DATA
    # ========================================
    st.subheader("🧬 Struktur dan Tipe Data")
    st.markdown("Informasi lengkap tentang setiap kolom dalam dataset beserta tipe datanya.")
    
    col_struct1, col_struct2 = st.columns([1, 1])
    
    with col_struct1:
        # Tabel struktur data
        struct_df = pd.DataFrame({
            'Kolom': df.columns.tolist(),
            'Tipe Data': [str(dt) for dt in df.dtypes],
            'Non-Null': [df[col].notnull().sum() for col in df.columns],
            'Null': [df[col].isnull().sum() for col in df.columns],
            'Unique': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(struct_df, use_container_width=True, hide_index=True)
    
    with col_struct2:
        # Ringkasan dimensional
        st.markdown("### 📐 Dimensi Dataset")
        st.info(f"""
        **Jumlah Baris:** {df.shape[0]} (provinsi)
        
        **Jumlah Kolom:** {df.shape[1]}
        
        **Ukuran Memori:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB
        
        **Kolom Numerik:** {df.select_dtypes(include=[np.number]).shape[1]}
        
        **Kolom Kategorikal:** {df.select_dtypes(exclude=[np.number]).shape[1]}
        """)
    
    st.markdown("---")
    
    # ========================================
    # STATISTIK DESKRIPTIF
    # ========================================
    st.subheader("📉 Statistik Deskriptif Komprehensif")
    st.markdown("Ringkasan statistik untuk setiap variabel numerik dalam dataset.")
    
    # Tab untuk berbagai jenis statistik
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Central Tendency",
        "📏 Dispersion", 
        "📐 Distribution Shape",
        "🏆 Percentiles"
    ])
    
    with tab1:
        # Tendensi sentral: mean, median, mode
        central_stats = pd.DataFrame({
            'Komoditas': KOMODITAS_LIST,
            'Mean (Rata-rata)': [df[k].mean() for k in KOMODITAS_LIST],
            'Median (Nilai Tengah)': [df[k].median() for k in KOMODITAS_LIST],
            'Mode (Modus)': [df[k].mode().iloc[0] if len(df[k].mode()) > 0 else np.nan for k in KOMODITAS_LIST],
            'Trimmed Mean (10%)': [stats.trim_mean(df[k], 0.1) for k in KOMODITAS_LIST]
        })
        
        st.markdown("""
        **Tendensi Sentral** mengukur lokasi pusat dari distribusi data.
        - **Mean**: Rata-rata aritmatika (sensitif terhadap outlier)
        - **Median**: Nilai tengah (robust terhadap outlier)
        - **Mode**: Nilai yang paling sering muncul
        - **Trimmed Mean**: Rata-rata setelah menghapus 10% data ekstrem
        """)
        
        st.dataframe(
            central_stats.style.format({
                'Mean (Rata-rata)': '{:,.2f}',
                'Median (Nilai Tengah)': '{:,.2f}',
                'Mode (Modus)': '{:,.2f}',
                'Trimmed Mean (10%)': '{:,.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    with tab2:
        # Dispersi: std, variance, range, IQR
        dispersion_stats = pd.DataFrame({
            'Komoditas': KOMODITAS_LIST,
            'Std Deviation': [df[k].std() for k in KOMODITAS_LIST],
            'Variance': [df[k].var() for k in KOMODITAS_LIST],
            'Range': [df[k].max() - df[k].min() for k in KOMODITAS_LIST],
            'IQR': [df[k].quantile(0.75) - df[k].quantile(0.25) for k in KOMODITAS_LIST],
            'CV (%)': [(df[k].std() / df[k].mean() * 100) if df[k].mean() > 0 else 0 for k in KOMODITAS_LIST]
        })
        
        st.markdown("""
        **Dispersi (Penyebaran)** mengukur seberapa luas data tersebar.
        - **Std Deviation**: Simpangan baku (deviasi rata-rata dari mean)
        - **Variance**: Variansi (kuadrat dari std dev)
        - **Range**: Selisih nilai maksimum dan minimum
        - **IQR**: Interquartile Range (Q3 - Q1)
        - **CV**: Coefficient of Variation (ukuran dispersi relatif)
        """)
        
        st.dataframe(
            dispersion_stats.style.format({
                'Std Deviation': '{:,.2f}',
                'Variance': '{:,.2f}',
                'Range': '{:,.2f}',
                'IQR': '{:,.2f}',
                'CV (%)': '{:,.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        # Bentuk distribusi: skewness dan kurtosis
        shape_data = []
        for k in KOMODITAS_LIST:
            sk_info = calculate_skewness_kurtosis(df[k])
            shape_data.append({
                'Komoditas': k,
                'Skewness': sk_info['skewness'],
                'Interpretasi Skewness': sk_info['skewness_interpretation'],
                'Kurtosis': sk_info['kurtosis'],
                'Interpretasi Kurtosis': sk_info['kurtosis_interpretation']
            })
        
        shape_df = pd.DataFrame(shape_data)
        
        st.markdown("""
        **Bentuk Distribusi** menggambarkan karakteristik distribusi data.
        - **Skewness**: Kemencengan (0 = simetris, positif = kanan, negatif = kiri)
        - **Kurtosis**: Keruncingan (0 = normal/mesokurtik, positif = lancip, negatif = datar)
        """)
        
        st.dataframe(
            shape_df.style.format({
                'Skewness': '{:,.3f}',
                'Kurtosis': '{:,.3f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    with tab4:
        # Percentiles
        percentiles_stats = pd.DataFrame({
            'Komoditas': KOMODITAS_LIST,
            'Min (0%)': [df[k].min() for k in KOMODITAS_LIST],
            'P10 (10%)': [df[k].quantile(0.10) for k in KOMODITAS_LIST],
            'Q1 (25%)': [df[k].quantile(0.25) for k in KOMODITAS_LIST],
            'Q2 (50%)': [df[k].quantile(0.50) for k in KOMODITAS_LIST],
            'Q3 (75%)': [df[k].quantile(0.75) for k in KOMODITAS_LIST],
            'P90 (90%)': [df[k].quantile(0.90) for k in KOMODITAS_LIST],
            'Max (100%)': [df[k].max() for k in KOMODITAS_LIST]
        })
        
        st.markdown("""
        **Percentiles** menunjukkan nilai pada posisi persentase tertentu dalam distribusi.
        Berguna untuk memahami distribusi data secara lebih detail daripada hanya mean/median.
        """)
        
        st.dataframe(
            percentiles_stats.style.format({
                'Min (0%)': '{:,.2f}',
                'P10 (10%)': '{:,.2f}',
                'Q1 (25%)': '{:,.2f}',
                'Q2 (50%)': '{:,.2f}',
                'Q3 (75%)': '{:,.2f}',
                'P90 (90%)': '{:,.2f}',
                'Max (100%)': '{:,.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # ========================================
    # VISUALISASI RINGKASAN (BOX PLOT)
    # ========================================
    st.subheader("📦 Visualisasi Distribusi (Box Plot)")
    st.markdown("Box plot menunjukkan distribusi, median, dan outlier untuk setiap komoditas.")
    
    # Melt dataframe untuk box plot
    df_melted = df.melt(
        id_vars=['Provinsi'], 
        value_vars=KOMODITAS_LIST,
        var_name='Komoditas',
        value_name='Produksi'
    )
    
    # Buat box plot
    fig_box = px.box(
        df_melted,
        x='Komoditas',
        y='Produksi',
        color='Komoditas',
        title='Distribusi Produksi per Komoditas (Box Plot)',
        labels={'Produksi': 'Produksi (ribu ton)', 'Komoditas': 'Komoditas'},
        color_discrete_sequence=[KOMODITAS_COLORS.get(k, '#2ecc71') for k in KOMODITAS_LIST]
    )
    
    # Apply custom theme
    fig_box = apply_plotly_theme(fig_box, height=500)
    fig_box.update_layout(showlegend=False)
    
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Insight box plot
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Insight dari Box Plot:</strong>
        <p>Terlihat jelas bahwa <b>Kelapa Sawit</b> memiliki distribusi yang paling lebar dengan outlier 
        yang sangat tinggi (provinsi Riau dan Kalimantan Tengah). Komoditas Teh dan Tebu memiliki 
        distribusi yang sangat sempit dengan banyak nilai nol, menunjukkan bahwa komoditas ini hanya 
        diproduksi di beberapa provinsi tertentu.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================
    # KOMODITAS DOMINAN PER PROVINSI
    # ========================================
    st.subheader("🎯 Komoditas Dominan per Provinsi")
    st.markdown("Setiap provinsi memiliki komoditas unggulan yang berbeda berdasarkan produksi tertinggi.")
    
    # Hitung jumlah provinsi per komoditas dominan
    dominant_counts = df['Komoditas_Dominan'].value_counts().reset_index()
    dominant_counts.columns = ['Komoditas Dominan', 'Jumlah Provinsi']
    
    col_dom1, col_dom2 = st.columns([1, 2])
    
    with col_dom1:
        st.dataframe(dominant_counts, use_container_width=True, hide_index=True)
    
    with col_dom2:
        # Pie chart komoditas dominan
        fig_pie = px.pie(
            dominant_counts,
            values='Jumlah Provinsi',
            names='Komoditas Dominan',
            title='Propinsi Berdasarkan Komoditas Dominan',
            hole=0.4,  # Donut chart
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig_pie = apply_plotly_theme(fig_pie, height=450)
        fig_pie.update_traces(
            textposition='outside',
            textinfo='percent+label',
            pull=[0.05 if i == 0 else 0 for i in range(len(dominant_counts))]
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================================
# BAGIAN 11: PAGE 2 - DATA CLEANING & PREPROCESSING
# ============================================================================
# Halaman kedua: pembersihan data dan preprocessing

elif "Page 2: Data Cleaning" in page:
    # Header halaman
    st.markdown("""
    <div class="page-header">
        <h1>🧹 Data Cleaning & Preprocessing</h1>
        <p>Tahapan pembersihan dan persiapan data untuk memastikan kualitas analisis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Tahapan data cleaning sangat krusial dalam proses data science karena kualitas output analisis 
    sangat bergantung pada kualitas input data. Prinsip **GIGO (Garbage In, Garbage Out)** 
    selalu berlaku - data yang buruk akan menghasilkan insight yang menyesatkan.
    """)
    
    st.markdown("---")
    
    # ========================================
    # 1. ANALISIS MISSING VALUES
    # ========================================
    st.subheader("1️⃣ Analisis Missing Values (Nilai Hilang)")
    st.markdown("Missing values dapat menyebabkan bias dalam analisis jika tidak ditangani dengan benar.")
    
    # Hitung missing values
    missing_count = df_raw.isnull().sum()
    missing_pct = (missing_count / len(df_raw)) * 100
    missing_df = pd.DataFrame({
        'Kolom': missing_count.index,
        'Jumlah Missing': missing_count.values,
        'Persentase (%)': missing_pct.values,
        'Tipe Data': [str(df_raw[col].dtype) for col in df_raw.columns]
    })
    
    # Tampilkan tabel
    col_mv1, col_mv2 = st.columns([2, 1])
    
    with col_mv1:
        st.dataframe(
            missing_df.style
            .background_gradient(subset=['Persentase (%)'], cmap='RdYlGn_r')
            .format({'Persentase (%)': '{:,.2f}'}),
            use_container_width=True,
            hide_index=True
        )
    
    with col_mv2:
        total_missing = missing_count.sum()
        if total_missing == 0:
            st.success(f"""
            ### ✅ Dataset Bersih
            
            Total missing values: **{total_missing}**
            
            Dataset ini tidak memiliki nilai hilang sehingga dapat langsung dianalisis.
            """)
        else:
            st.warning(f"""
            ### ⚠️ Missing Values Terdeteksi
            
            Total: **{total_missing}** nilai hilang
            
            Perlu penanganan lebih lanjut.
            """)
    
    # Visualisasi missing values
    st.markdown("### 📊 Visualisasi Missing Values")
    fig_missing = go.Figure()
    
    fig_missing.add_trace(go.Bar(
        x=missing_df['Kolom'],
        y=missing_df['Jumlah Missing'],
        marker=dict(
            color=[
                '#2ecc71' if v == 0 else '#e74c3c' 
                for v in missing_df['Jumlah Missing']
            ],
            line=dict(color='#f1c40f', width=1)
        ),
        text=missing_df['Jumlah Missing'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Missing: %{y}<extra></extra>'
    ))
    
    fig_missing = apply_plotly_theme(
        fig_missing, 
        title="Jumlah Missing Values per Kolom",
        height=400
    )
    fig_missing.update_layout(
        xaxis_title="Kolom",
        yaxis_title="Jumlah Missing"
    )
    
    st.plotly_chart(fig_missing, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # 2. ANALISIS DUPLIKASI
    # ========================================
    st.subheader("2️⃣ Analisis Data Duplikat")
    st.markdown("Data duplikat dapat menyebabkan bias dalam statistik dan analisis.")
    
    # Cek duplikasi
    n_duplicates = df_raw.duplicated().sum()
    duplicate_pct = (n_duplicates / len(df_raw)) * 100
    
    # Cek duplikasi per subset kolom
    dup_provinsi = df_raw.duplicated(subset=['Provinsi']).sum()
    
    # Tampilkan hasil
    col_dup1, col_dup2, col_dup3 = st.columns(3)
    
    with col_dup1:
        st.markdown(create_metric_card(
            icon="📋",
            value=str(n_duplicates),
            label="Total Duplikat (Semua Kolom)",
            change=f"{duplicate_pct:.2f}%",
            change_type="positive" if n_duplicates == 0 else "negative"
        ), unsafe_allow_html=True)
    
    with col_dup2:
        st.markdown(create_metric_card(
            icon="🏛️",
            value=str(dup_provinsi),
            label="Duplikat Provinsi",
            change="Nama provinsi sama",
            change_type="positive" if dup_provinsi == 0 else "negative"
        ), unsafe_allow_html=True)
    
    with col_dup3:
        total_rows = len(df_raw)
        unique_rows = total_rows - n_duplicates
        st.markdown(create_metric_card(
            icon="✅",
            value=str(unique_rows),
            label="Baris Unik",
            change=f"{(unique_rows/total_rows)*100:.1f}% dari total",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    # Status duplikasi
    if n_duplicates == 0 and dup_provinsi == 0:
        st.success("✅ **Dataset bersih dari duplikasi** - setiap provinsi tercatat satu kali dengan data yang unik.")
    else:
        st.warning(f"⚠️ Ditemukan {n_duplicates} duplikat yang perlu direview.")
    
    # Tampilkan detail duplikat jika ada
    if n_duplicates > 0:
        with st.expander("🔍 Lihat Detail Data Duplikat"):
            duplicates = df_raw[df_raw.duplicated(keep=False)]
            st.dataframe(duplicates, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # 3. DETEKSI DAN ANALISIS OUTLIER
    # ========================================
    st.subheader("3️⃣ Deteksi dan Analisis Outlier")
    st.markdown("""
    Outlier adalah observasi yang secara signifikan berbeda dari observasi lainnya. 
    Dalam konteks data produksi pertanian, outlier seringkali **bukan error** melainkan 
    merepresentasikan provinsi produsen utama (contoh: Riau untuk Kelapa Sawit).
    """)
    
    # Metode deteksi outlier
    metode_outlier = st.radio(
        "Pilih Metode Deteksi Outlier:",
        ["IQR Method (Tukey's Fences)", "Z-Score Method", "Modified Z-Score (MAD)"],
        horizontal=True
    )
    
    # Hasil analisis outlier
    outlier_results = []
    
    for komoditas in KOMODITAS_LIST:
        series = df[komoditas]
        
        if "IQR" in metode_outlier:
            # Metode IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = series[(series < lower) | (series > upper)]
            
            outlier_results.append({
                'Komoditas': komoditas,
                'Metode': 'IQR',
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'Lower Bound': lower,
                'Upper Bound': upper,
                'Jumlah Outlier': len(outliers),
                'Persentase (%)': (len(outliers) / len(series)) * 100,
                'Outlier Provinsi': ', '.join(df.loc[outliers.index, 'Provinsi'].tolist())
            })
        
        elif "Z-Score" in metode_outlier:
            # Metode Z-Score (threshold = 3)
            mean = series.mean()
            std = series.std()
            z_scores = np.abs((series - mean) / std) if std > 0 else pd.Series([0]*len(series))
            outliers = series[z_scores > 3]
            
            outlier_results.append({
                'Komoditas': komoditas,
                'Metode': 'Z-Score',
                'Mean': mean,
                'Std Dev': std,
                'Threshold': 3.0,
                'Jumlah Outlier': len(outliers),
                'Persentase (%)': (len(outliers) / len(series)) * 100,
                'Outlier Provinsi': ', '.join(df.loc[outliers.index, 'Provinsi'].tolist())
            })
        
        else:
            # Modified Z-Score (MAD)
            median = series.median()
            mad = np.median(np.abs(series - median))
            if mad > 0:
                modified_z = 0.6745 * (series - median) / mad
                outliers = series[np.abs(modified_z) > 3.5]
            else:
                outliers = pd.Series([], dtype=float)
            
            outlier_results.append({
                'Komoditas': komoditas,
                'Metode': 'Modified Z-Score',
                'Median': median,
                'MAD': mad,
                'Threshold': 3.5,
                'Jumlah Outlier': len(outliers),
                'Persentase (%)': (len(outliers) / len(series)) * 100,
                'Outlier Provinsi': ', '.join(df.loc[outliers.index, 'Provinsi'].tolist())
            })
    
    # Buat dataframe hasil
    outlier_df = pd.DataFrame(outlier_results)
    
    # Tampilkan tabel
    st.dataframe(
        outlier_df[['Komoditas', 'Jumlah Outlier', 'Persentase (%)', 'Outlier Provinsi']].style
        .background_gradient(subset=['Jumlah Outlier'], cmap='YlOrRd')
        .format({'Persentase (%)': '{:,.2f}'}),
        use_container_width=True,
        hide_index=True
    )
    
    # Catatan interpretasi
    st.markdown("""
    <div class="info-box">
        <strong>📝 Catatan Interpretasi:</strong>
        <p>Dalam konteks data produksi perkebunan, <b>outlier umumnya merepresentasikan provinsi produsen utama</b> 
        (contoh: Riau untuk Kelapa Sawit dengan produksi 9,136 ribu ton) dan <b>BUKAN error data</b>. 
        Outlier ini merupakan karakteristik alami dari distribusi produksi yang sangat timpang 
        antar provinsi Indonesia.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualisasi box plot per komoditas dengan outlier ditandai
    st.markdown("### 📊 Visualisasi Outlier dengan Box Plot")
    
    komoditas_terpilih = st.selectbox(
        "Pilih komoditas untuk visualisasi:",
        KOMODITAS_LIST,
        format_func=lambda x: KOMODITAS_LABELS.get(x, x)
    )
    
    # Ambil data komoditas terpilih
    data_komoditas = df[['Provinsi', komoditas_terpilih]].copy()
    data_komoditas.columns = ['Provinsi', 'Produksi']
    
    # Identifikasi outlier untuk komoditas ini
    Q1 = data_komoditas['Produksi'].quantile(0.25)
    Q3 = data_komoditas['Produksi'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    
    data_komoditas['Is_Outlier'] = (
        (data_komoditas['Produksi'] > upper_bound) | 
        (data_komoditas['Produksi'] < lower_bound)
    )
    data_komoditas['Category'] = data_komoditas['Is_Outlier'].map({
        True: '🔴 Outlier',
        False: '🟢 Normal'
    })
    
    # Buat scatter plot dengan outlier ditandai
    fig_outlier_viz = px.scatter(
        data_komoditas.sort_values('Produksi', ascending=False).reset_index(drop=True),
        x='Provinsi',
        y='Produksi',
        color='Category',
        size='Produksi',
        size_max=40,
        hover_data=['Provinsi', 'Produksi', 'Category'],
        title=f'Identifikasi Outlier: {KOMODITAS_LABELS.get(komoditas_terpilih, komoditas_terpilih)}',
        color_discrete_map={'🔴 Outlier': '#e74c3c', '🟢 Normal': '#2ecc71'}
    )
    
    # Tambah garis upper bound
    fig_outlier_viz.add_hline(
        y=upper_bound, 
        line_dash="dash", 
        line_color="#f1c40f",
        annotation_text=f"Upper Bound: {upper_bound:,.2f}",
        annotation_position="top right"
    )
    
    fig_outlier_viz = apply_plotly_theme(fig_outlier_viz, height=500)
    fig_outlier_viz.update_layout(xaxis_tickangle=-45)
    
    st.plotly_chart(fig_outlier_viz, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # 4. FEATURE ENGINEERING
    # ========================================
    st.subheader("4️⃣ Feature Engineering")
    st.markdown("Membuat fitur-fitur baru untuk memperkaya analisis.")
    
    # Tampilkan fitur baru yang sudah dibuat
    st.markdown("""
    ### ✨ Fitur Baru yang Dibuat:
    
    1. **`Wilayah`**: Pengelompokan provinsi ke dalam 7 wilayah geografis (Sumatera, Jawa, Kalimantan, dll)
    2. **`Total_Produksi`**: Jumlah produksi seluruh komoditas per provinsi
    3. **`Rank_Total`**: Ranking provinsi berdasarkan total produksi
    4. **`Komoditas_Dominan`**: Komoditas dengan produksi tertinggi di setiap provinsi
    5. **`HHI_Index`**: Herfindahl-Hirschman Index untuk mengukur diversifikasi
    6. **`Diversifikasi`**: Kategori diversifikasi berdasarkan HHI
    7. **`Latitude` & `Longitude`**: Koordinat geografis untuk visualisasi peta
    """)
    
    # Tampilkan preview fitur baru
    st.markdown("### 📋 Preview Data dengan Fitur Baru")
    
    kolom_preview = st.multiselect(
        "Pilih kolom yang ingin ditampilkan:",
        df.columns.tolist(),
        default=['Provinsi', 'Wilayah', 'Total_Produksi', 'Rank_Total', 'Komoditas_Dominan', 'HHI_Index', 'Diversifikasi']
    )
    
    if kolom_preview:
        st.dataframe(
            df[kolom_preview].style
            .background_gradient(subset=['Total_Produksi', 'HHI_Index'], cmap='Greens'),
            use_container_width=True,
            height=500
        )
    
    st.markdown("---")
    
    # ========================================
    # 5. NORMALISASI & STANDARDISASI
    # ========================================
    st.subheader("5️⃣ Normalisasi & Standardisasi")
    st.markdown("Membandingkan teknik scaling yang berbeda untuk data numerik.")
    
    # Pilihan teknik scaling
    teknik_scaling = st.selectbox(
        "Pilih Teknik Scaling:",
        ["StandardScaler (Z-score)", "MinMaxScaler (0-1)", "RobustScaler (median-based)"],
        key="scaling_tech"
    )
    
    # Inisialisasi scaler
    if "Standard" in teknik_scaling:
        scaler = StandardScaler()
        scaler_name = "StandardScaler"
        scaler_desc = "Mengubah data sehingga memiliki mean=0 dan std=1. Cocok untuk data yang terdistribusi normal."
    elif "MinMax" in teknik_scaling:
        scaler = MinMaxScaler()
        scaler_name = "MinMaxScaler"
        scaler_desc = "Mengubah data ke rentang [0, 1]. Sensitif terhadap outlier."
    else:
        scaler = RobustScaler()
        scaler_name = "RobustScaler"
        scaler_desc = "Menggunakan median dan IQR, robust terhadap outlier."
    
    st.info(f"**{scaler_name}:** {scaler_desc}")
    
    # Apply scaling
    scaled_data = scaler.fit_transform(df[KOMODITAS_LIST])
    df_scaled = pd.DataFrame(scaled_data, columns=KOMODITAS_LIST)
    df_scaled.insert(0, 'Provinsi', df['Provinsi'])
    
    # Tampilkan hasil scaling
    st.markdown("### 📊 Hasil Scaling")
    st.dataframe(
        df_scaled.style.background_gradient(subset=KOMODITAS_LIST, cmap='RdYlGn'),
        use_container_width=True,
        height=500
    )
    
    # Perbandingan sebelum dan sesudah scaling
    st.markdown("### 📈 Perbandingan Distribusi Sebelum vs Sesudah Scaling")
    
    komoditas_compare = st.selectbox(
        "Pilih komoditas untuk perbandingan:",
        KOMODITAS_LIST,
        format_func=lambda x: KOMODITAS_LABELS.get(x, x),
        key="compare_komoditas"
    )
    
    fig_compare = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribusi Original', 'Distribusi Setelah Scaling'),
        horizontal_spacing=0.1
    )
    
    # Histogram original
    fig_compare.add_trace(
        go.Histogram(
            x=df[komoditas_compare],
            name='Original',
            marker_color='#f1c40f',
            opacity=0.7,
            nbinsx=15
        ),
        row=1, col=1
    )
    
    # Histogram scaled
    fig_compare.add_trace(
        go.Histogram(
            x=df_scaled[komoditas_compare],
            name=f'Scaled ({scaler_name})',
            marker_color='#2ecc71',
            opacity=0.7,
            nbinsx=15
        ),
        row=1, col=2
    )
    
    fig_compare = apply_plotly_theme(fig_compare, height=400)
    fig_compare.update_layout(showlegend=False)
    
    st.plotly_chart(fig_compare, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # 6. RINGKASAN KUALITAS DATA
    # ========================================
    st.subheader("6️⃣ Ringkasan Kualitas Data")
    st.markdown("Evaluasi menyeluruh terhadap kualitas dataset.")
    
    # Hitung skor kualitas
    completeness = ((df_raw.notnull().sum().sum()) / (df_raw.shape[0] * df_raw.shape[1])) * 100
    uniqueness = ((len(df_raw) - n_duplicates) / len(df_raw)) * 100
    
    # Total outlier dari IQR method
    total_outliers_pct = sum([
        detect_outliers_iqr(df[k], k)['outlier_percentage'] 
        for k in KOMODITAS_LIST
    ]) / len(KOMODITAS_LIST)
    
    consistency = 100 - total_outliers_pct
    
    # Skor keseluruhan
    overall_score = (completeness + uniqueness + consistency) / 3
    
    # Tampilkan skor
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    
    with col_q1:
        st.markdown(create_metric_card(
            icon="✅",
            value=f"{completeness:.1f}%",
            label="Kelengkapan Data",
            change="Completeness",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    with col_q2:
        st.markdown(create_metric_card(
            icon="🎯",
            value=f"{uniqueness:.1f}%",
            label="Keunikan Data",
            change="Uniqueness",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    with col_q3:
        st.markdown(create_metric_card(
            icon="📏",
            value=f"{consistency:.1f}%",
            label="Konsistensi Data",
            change="1 - Outlier %",
            change_type="positive" if consistency > 70 else "negative"
        ), unsafe_allow_html=True)
    
    with col_q4:
        st.markdown(create_metric_card(
            icon="⭐",
            value=f"{overall_score:.1f}%",
            label="Skor Kualitas",
            change="Overall",
            change_type="positive" if overall_score > 80 else "negative"
        ), unsafe_allow_html=True)
    
    # Progress bar visual
    st.markdown("### 📊 Skor Kualitas Visual")
    
    st.markdown(f"**Completeness (Kelengkapan)**")
    st.progress(completeness / 100)
    
    st.markdown(f"**Uniqueness (Keunikan)**")
    st.progress(uniqueness / 100)
    
    st.markdown(f"**Consistency (Konsistensi)**")
    st.progress(consistency / 100)
    
    # Rekomendasi berdasarkan kualitas
    st.markdown("### 💡 Rekomendasi Berdasarkan Kualitas Data")
    
    if overall_score >= 90:
        st.success("🌟 **Excellent!** Dataset memiliki kualitas sangat baik dan siap untuk analisis mendalam.")
    elif overall_score >= 75:
        st.info("👍 **Good.** Dataset memiliki kualitas yang baik dengan beberapa hal yang perlu diperhatikan.")
    elif overall_score >= 60:
        st.warning("⚠️ **Fair.** Dataset perlu beberapa pembersihan lebih lanjut sebelum analisis.")
    else:
        st.error("❌ **Poor.** Dataset memerlukan pembersihan signifikan sebelum dapat dianalisis.")

# ============================================================================
# BAGIAN 12: PAGE 3 - EDA & 3D VISUALIZATIONS
# ============================================================================
# Halaman ketiga: Exploratory Data Analysis dengan visualisasi 3D

elif "Page 3: EDA" in page:
    # Header halaman
    st.markdown("""
    <div class="page-header">
        <h1>📈 Exploratory Data Analysis & 3D Visualizations</h1>
        <p>Eksplorasi mendalam dengan visualisasi 3D interaktif untuk mengungkap pola tersembunyi</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Visualisasi 3D memungkinkan kita untuk menganalisis **tiga dimensi sekaligus** dalam satu grafik, 
    memberikan perspektif yang lebih kaya dibandingkan visualisasi 2D tradisional. Dengan Plotly, 
    kita dapat membuat grafik 3D yang interaktif dan dapat diputar untuk eksplorasi optimal.
    """)
    
    st.markdown("---")
    
    # ========================================
    # EDA UNIVARIAT
    # ========================================
    st.subheader("🔬 Analisis Univariat")
    st.markdown("Eksplorasi distribusi setiap komoditas secara terpisah.")
    
    # Pilihan komoditas
    komoditas_uni = st.selectbox(
        "Pilih komoditas untuk analisis:",
        KOMODITAS_LIST,
        format_func=lambda x: KOMODITAS_LABELS.get(x, x),
        key="uni_komoditas"
    )
    
    # Ambil data
    data_uni = df[komoditas_uni]
    
    # Hitung statistik
    stat_uni = {
        'Mean': data_uni.mean(),
        'Median': data_uni.median(),
        'Std Dev': data_uni.std(),
        'Min': data_uni.min(),
        'Max': data_uni.max(),
        'Skewness': data_uni.skew(),
        'Kurtosis': data_uni.kurtosis()
    }
    
    # Tampilkan metrik
    cols_uni = st.columns(4)
    stat_items = list(stat_uni.items())
    for i, (key, val) in enumerate(stat_items):
        with cols_uni[i % 4]:
            st.markdown(create_metric_card(
                icon="📊" if i < 4 else "📈",
                value=format_number(val, 2),
                label=key
            ), unsafe_allow_html=True)
    
    # Grafik distribusi
    col_u1, col_u2 = st.columns(2)
    
    with col_u1:
        # Histogram + KDE
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=data_uni,
            name='Frekuensi',
            marker_color='#2ecc71',
            opacity=0.7,
            nbinsx=15
        ))
        
        # Tambah KDE (Kernel Density Estimation)
        kde_x = np.linspace(data_uni.min(), data_uni.max(), 100)
        kde = stats.gaussian_kde(data_uni[data_uni > 0]) if (data_uni > 0).sum() > 1 else None
        
        if kde is not None:
            kde_y = kde(kde_x)
            # Scale untuk match histogram
            kde_y_scaled = kde_y * len(data_uni) * (data_uni.max() - data_uni.min()) / 15
            
            fig_hist.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y_scaled,
                mode='lines',
                name='KDE',
                line=dict(color='#f1c40f', width=3)
            ))
        
        fig_hist = apply_plotly_theme(
            fig_hist, 
            title=f'Distribusi {KOMODITAS_LABELS.get(komoditas_uni, komoditas_uni)}',
            height=400
        )
        fig_hist.update_layout(xaxis_title="Produksi (ribu ton)", yaxis_title="Frekuensi")
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_u2:
        # QQ Plot (Normal Probability Plot)
        fig_qq = go.Figure()
        
        # Hitung theoretical quantiles
        (osm, osr), (slope, intercept, r) = stats.probplot(data_uni, dist="norm")
        
        fig_qq.add_trace(go.Scatter(
            x=osm,
            y=osr,
            mode='markers',
            name='Data',
            marker=dict(color='#2ecc71', size=8, line=dict(color='#f1c40f', width=1))
        ))
        
        # Garis referensi
        line_x = np.array([min(osm), max(osm)])
        line_y = intercept + slope * line_x
        
        fig_qq.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            name='Reference Line',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        fig_qq = apply_plotly_theme(
            fig_qq,
            title=f'QQ Plot: {KOMODITAS_LABELS.get(komoditas_uni, komoditas_uni)}',
            height=400
        )
        fig_qq.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        
        st.plotly_chart(fig_qq, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # EDA BIVARIAT
    # ========================================
    st.subheader("🔍 Analisis Bivariat")
    st.markdown("Eksplorasi hubungan antara dua komoditas.")
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        komoditas_x = st.selectbox(
            "Komoditas X (horizontal):",
            KOMODITAS_LIST,
            format_func=lambda x: KOMODITAS_LABELS.get(x, x),
            key="bi_x"
        )
    
    with col_b2:
        komoditas_y = st.selectbox(
            "Komoditas Y (vertikal):",
            KOMODITAS_LIST,
            format_func=lambda x: KOMODITAS_LABELS.get(x, x),
            index=1,
            key="bi_y"
        )
    
    # Hitung korelasi
    corr_info = calculate_correlation_significance(df[komoditas_x], df[komoditas_y])
    
    # Tampilkan metrik korelasi
    col_corr1, col_corr2, col_corr3 = st.columns(3)
    
    with col_corr1:
        st.markdown(create_metric_card(
            icon="📐",
            value=f"{corr_info['pearson_r']:.3f}",
            label="Pearson Correlation",
            change=corr_info['pearson_strength'],
            change_type="positive" if corr_info['pearson_significant'] else "negative"
        ), unsafe_allow_html=True)
    
    with col_corr2:
        st.markdown(create_metric_card(
            icon="📊",
            value=f"{corr_info['spearman_r']:.3f}",
            label="Spearman Correlation",
            change=corr_info['spearman_strength'],
            change_type="positive" if corr_info['spearman_significant'] else "negative"
        ), unsafe_allow_html=True)
    
    with col_corr3:
        p_value_display = f"{corr_info['pearson_p']:.4f}" if corr_info['pearson_p'] > 0.0001 else "< 0.0001"
        st.markdown(create_metric_card(
            icon="🎯",
            value=p_value_display,
            label="P-value (Pearson)",
            change="Signifikan" if corr_info['pearson_significant'] else "Tidak Signifikan",
            change_type="positive" if corr_info['pearson_significant'] else "negative"
        ), unsafe_allow_html=True)
    
    # Scatter plot bivariate
    fig_biv = px.scatter(
        df,
        x=komoditas_x,
        y=komoditas_y,
        size='Total_Produksi',
        color='Wilayah',
        hover_name='Provinsi',
        hover_data=[komoditas_x, komoditas_y, 'Total_Produksi'],
        title=f'Hubungan {KOMODITAS_LABELS.get(komoditas_x)} vs {KOMODITAS_LABELS.get(komoditas_y)}',
        color_discrete_map=REGION_COLORS,
        trendline='ols'
    )
    
    fig_biv = apply_plotly_theme(fig_biv, height=550)
    fig_biv.update_layout(
        xaxis_title=f'{KOMODITAS_LABELS.get(komoditas_x)} (ribu ton)',
        yaxis_title=f'{KOMODITAS_LABELS.get(komoditas_y)} (ribu ton)'
    )
    
    st.plotly_chart(fig_biv, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # GRAFIK 3D #1: SCATTER 3D (3 KOMODITAS)
    # ========================================
    st.subheader("🧊 Visualisasi 3D #1: Scatter 3D Interaktif")
    st.markdown("""
    Scatter plot 3D memvisualisasikan hubungan **tiga komoditas sekaligus**. Setiap titik mewakili satu provinsi, 
    dengan posisi 3D berdasarkan produksi tiga komoditas yang dipilih.
    """)
    
    # Pilih 3 komoditas
    col_3d1, col_3d2, col_3d3 = st.columns(3)
    
    with col_3d1:
        axis_x_3d = st.selectbox(
            "Sumbu X:",
            KOMODITAS_LIST,
            format_func=lambda x: KOMODITAS_LABELS.get(x, x),
            key="3d_x"
        )
    
    with col_3d2:
        axis_y_3d = st.selectbox(
            "Sumbu Y:",
            KOMODITAS_LIST,
            format_func=lambda x: KOMODITAS_LABELS.get(x, x),
            index=1,
            key="3d_y"
        )
    
    with col_3d3:
        axis_z_3d = st.selectbox(
            "Sumbu Z:",
            KOMODITAS_LIST,
            format_func=lambda x: KOMODITAS_LABELS.get(x, x),
            index=2,
            key="3d_z"
        )
    
    # Pilihan grup warna
    color_by_3d = st.radio(
        "Warnai berdasarkan:",
        ["Wilayah", "Komoditas_Dominan", "Diversifikasi"],
        horizontal=True
    )
    
    # Buat scatter 3D
    fig_scatter_3d = px.scatter_3d(
        df,
        x=axis_x_3d,
        y=axis_y_3d,
        z=axis_z_3d,
        color=color_by_3d,
        size='Total_Produksi',
        size_max=35,
        hover_name='Provinsi',
        hover_data={
            axis_x_3d: ':,.2f',
            axis_y_3d: ':,.2f',
            axis_z_3d: ':,.2f',
            'Total_Produksi': ':,.2f'
        },
        title=f'🌐 Scatter 3D: {axis_x_3d.replace("_", " ")} × {axis_y_3d.replace("_", " ")} × {axis_z_3d.replace("_", " ")}',
        opacity=0.85
    )
    
    # Styling
    fig_scatter_3d = apply_plotly_theme(fig_scatter_3d, height=650)
    fig_scatter_3d.update_layout(
        scene=dict(
            xaxis_title=f'{axis_x_3d.replace("_", " ")} (ribu ton)',
            yaxis_title=f'{axis_y_3d.replace("_", " ")} (ribu ton)',
            zaxis_title=f'{axis_z_3d.replace("_", " ")} (ribu ton)'
        )
    )
    
    st.plotly_chart(fig_scatter_3d, use_container_width=True)
    
    # Tips penggunaan
    st.markdown("""
    <div class="info-box">
        <strong>💡 Tips Interaksi Grafik 3D:</strong>
        <ul style="margin: 8px 0 0 0; color: var(--text-secondary);">
            <li>🖱️ <b>Drag kiri</b>: Putar grafik untuk melihat dari berbagai sudut</li>
            <li>🖱️ <b>Scroll</b>: Zoom in/out</li>
            <li>🖱️ <b>Drag kanan</b>: Geser (pan) grafik</li>
            <li>🖱️ <b>Double click</b>: Reset view</li>
            <li>💾 Klik ikon kamera untuk menyimpan gambar</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================
    # GRAFIK 3D #2: SURFACE PLOT (TOPOGRAFI PRODUKSI)
    # ========================================
    st.subheader("🏔️ Visualisasi 3D #2: Surface Plot (Topografi Produksi)")
    st.markdown("""
    Surface plot menggambarkan 'topografi' produksi komoditas di seluruh provinsi. 
    Gunung tinggi = produksi tinggi, lembah = produksi rendah. Visualisasi ini membantu 
    mengidentifikasi pola produksi lintas komoditas dan provinsi.
    """)
    
    # Pilih jumlah provinsi teratas
    n_prov_surface = st.slider(
        "Jumlah provinsi teratas yang ditampilkan:",
        min_value=5, max_value=38, value=20, step=1
    )
    
    # Urutkan berdasarkan total produksi
    df_sorted = df.nlargest(n_prov_surface, 'Total_Produksi').set_index('Provinsi')[KOMODITAS_LIST]
    
    # Matriks Z untuk surface plot
    Z_surface = df_sorted.values
    X_surface = np.arange(len(KOMODITAS_LIST))
    Y_surface = np.arange(len(df_sorted))
    
    # Pilihan colorscale
    colorscale_surface = st.selectbox(
        "Pilih Colorscale:",
        ['Emerald', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo', 'Jet'],
        index=0
    )
    
    # Buat surface plot
    fig_surface = go.Figure(data=[go.Surface(
        z=Z_surface,
        x=X_surface,
        y=Y_surface,
        colorscale=colorscale_surface,
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="#f1c40f",
                project_z=True
            )
        ),
        colorbar=dict(
            title="Produksi (ribu ton)",
            tickfont=dict(color="white"),
            titlefont=dict(color="white"),
            thickness=20
        ),
        opacity=0.95
    )])
    
    # Layout
    fig_surface = apply_plotly_theme(
        fig_surface, 
        title=f"🗻 Topografi Produksi {n_prov_surface} Provinsi Teratas",
        height=700
    )
    
    fig_surface.update_layout(
        scene=dict(
            xaxis=dict(
                title='Komoditas',
                tickvals=X_surface,
                ticktext=[k.replace('_', ' ') for k in KOMODITAS_LIST],
                color="#2ecc71",
                titlefont=dict(size=14)
            ),
            yaxis=dict(
                title='Provinsi',
                tickvals=Y_surface,
                ticktext=[name[:15] for name in df_sorted.index],
                color="#2ecc71",
                titlefont=dict(size=14)
            ),
            zaxis=dict(
                title='Produksi (ribu ton)',
                color="#f1c40f",
                titlefont=dict(size=14)
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                center=dict(x=0, y=0, z=0)
            )
        )
    )
    
    st.plotly_chart(fig_surface, use_container_width=True)
    
    # Insight surface plot
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Insight dari Surface Plot:</strong>
        <p>Gunung tertinggi biasanya didominasi oleh <b>Kelapa Sawit di provinsi seperti Riau, Kalimantan Tengah, 
        dan Sumatera Utara</b>. Komoditas seperti Teh dan Tebu membentuk 'lembah' yang sangat rendah karena 
        hanya diproduksi di beberapa provinsi (Jawa Barat, Jawa Timur, Lampung).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================
    # GRAFIK 3D #3: MESH3D BAR CHART
    # ========================================
    st.subheader("🏗️ Visualisasi 3D #3: Bar Chart 3D")
    st.markdown("""
    Bar chart 3D memberikan representasi volumetrik dari total produksi per provinsi. 
    Setiap balok memiliki volume yang proporsional terhadap total produksi provinsi tersebut.
    """)
    
    # Pilih jumlah provinsi
    n_prov_bar3d = st.slider(
        "Jumlah provinsi untuk Bar 3D:",
        min_value=5, max_value=20, value=10, step=1
    )
    
    # Ambil top N provinsi
    top_n_bar = df.nlargest(n_prov_bar3d, 'Total_Produksi')[['Provinsi', 'Total_Produksi']].reset_index(drop=True)
    
    # Buat figure
    fig_bar3d = go.Figure()
    
    # Warna gradien
    colors_bar = [
        '#2ecc71', '#27ae60', '#16a085', '#1abc9c', '#0e6655',
        '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c',
        '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e'
    ]
    
    # Buat balok 3D untuk setiap provinsi
    for i, row in top_n_bar.iterrows():
        # Koordinat balok
        x0, x1 = i - 0.35, i + 0.35
        y0, y1 = 0, 0.7
        z0, z1 = 0, row['Total_Produksi']
        
        # 8 vertices dari balok
        vertices_x = [x0, x1, x1, x0, x0, x1, x1, x0]
        vertices_y = [y0, y0, y1, y1, y0, y0, y1, y1]
        vertices_z = [z0, z0, z0, z0, z1, z1, z1, z1]
        
        # Indices untuk 12 segitiga (6 faces × 2 triangles)
        i_idx = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
        j_idx = [1, 2, 5, 6, 1, 5, 3, 7, 2, 6, 3, 5]
        k_idx = [2, 3, 6, 7, 5, 4, 7, 6, 3, 7, 5, 4]
        
        # Warna balok
        color = colors_bar[i % len(colors_bar)]
        
        # Tambahkan balok
        fig_bar3d.add_trace(go.Mesh3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            i=i_idx,
            j=j_idx,
            k=k_idx,
            color=color,
            opacity=0.85,
            flatshading=True,
            name=row['Provinsi'],
            hovertext=f"<b>{row['Provinsi']}</b><br>Total: {row['Total_Produksi']:,.0f} ribu ton<br>Rank: #{i+1}",
            hoverinfo='text',
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                fresnel=0.2,
                specular=0.5,
                roughness=0.5
            ),
            lightposition=dict(x=100, y=200, z=300)
        ))
    
    # Tambahkan base/landasan
    fig_bar3d.add_trace(go.Mesh3d(
        x=[-1, n_prov_bar3d, n_prov_bar3d, -1],
        y=[-0.5, -0.5, 1.2, 1.2],
        z=[0, 0, 0, 0],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color='#0a1f14',
        opacity=0.5,
        hoverinfo='skip'
    ))
    
    # Layout
    fig_bar3d = apply_plotly_theme(
        fig_bar3d,
        title=f"🏛️ Bar 3D: Top {n_prov_bar3d} Provinsi dengan Produksi Tertinggi",
        height=650
    )
    
    fig_bar3d.update_layout(
        scene=dict(
            xaxis=dict(
                title='Provinsi',
                tickvals=list(range(n_prov_bar3d)),
                ticktext=[name[:12] for name in top_n_bar['Provinsi']],
                color="#2ecc71",
                titlefont=dict(size=14)
            ),
            yaxis=dict(visible=False),
            zaxis=dict(
                title='Total Produksi (ribu ton)',
                color="#f1c40f",
                titlefont=dict(size=14)
            ),
            camera=dict(
                eye=dict(x=1.8, y=-1.8, z=1.0)
            )
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig_bar3d, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # GRAFIK 3D #4: SCATTER 3D DENGAN TREND LINE
    # ========================================
    st.subheader("🌀 Visualisasi 3D #4: 3D Bubble Chart dengan Wilayah")
    st.markdown("""
    Visualisasi 3D yang menampilkan setiap provinsi sebagai bubble, dengan sumbu yang mencerminkan 
    karakteristik produksi (dominansi, diversifikasi, dan total).
    """)
    
    # Hitung rasio komoditas dominan terhadap total
    def calc_dominance_ratio(row):
        """Hitung rasio produksi komoditas dominan terhadap total."""
        values = {k: row[k] for k in KOMODITAS_LIST}
        max_val = max(values.values())
        total = sum(values.values())
        return (max_val / total * 100) if total > 0 else 0
    
    df['Dominance_Ratio'] = df.apply(calc_dominance_ratio, axis=1)
    
    # Bubble 3D
    fig_bubble3d = px.scatter_3d(
        df,
        x='Dominance_Ratio',
        y='HHI_Index',
        z='Total_Produksi',
        color='Wilayah',
        size='Total_Produksi',
        size_max=50,
        hover_name='Provinsi',
        hover_data={
            'Dominance_Ratio': ':.1f',
            'HHI_Index': ':.3f',
            'Total_Produksi': ':,.0f'
        },
        title='🌀 Bubble 3D: Dominansi × Diversifikasi × Total Produksi',
        color_discrete_map=REGION_COLORS,
        opacity=0.8
    )
    
    fig_bubble3d = apply_plotly_theme(fig_bubble3d, height=650)
    fig_bubble3d.update_layout(
        scene=dict(
            xaxis_title='Rasio Dominansi (%)',
            yaxis_title='HHI Index (Diversifikasi)',
            zaxis_title='Total Produksi (ribu ton)'
        )
    )
    
    st.plotly_chart(fig_bubble3d, use_container_width=True)
    
    # Insight
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Interpretasi:</strong>
        <p>• <b>Atas-kanan</b>: Provinsi dengan produksi besar dan dominansi tinggi (spesialis kuat, misal Riau)<br>
        • <b>Atas-kiri</b>: Provinsi dengan produksi besar tapi diversifikasi baik (misal Sumatera Utara)<br>
        • <b>Bawah-kanan</b>: Provinsi produksi kecil tapi sangat bergantung pada 1 komoditas<br>
        • <b>Bawah-kiri</b>: Provinsi produksi kecil dengan diversifikasi baik</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================
    # GRAFIK 3D #5: STACKED AREA 3D
    # ========================================
    st.subheader("🎨 Visualisasi 3D #5: Stacked Bar Komposisi Komoditas")
    st.markdown("""
    Visualisasi 3D yang menunjukkan komposisi produksi setiap komoditas di setiap provinsi.
    Setiap segmen balok menunjukkan kontribusi satu komoditas terhadap total produksi.
    """)
    
    # Pilih jumlah provinsi
    n_prov_stack = st.slider(
        "Jumlah provinsi (sorted by total):",
        min_value=5, max_value=25, value=15, step=1,
        key="n_stack"
    )
    
    # Ambil top N
    top_stack = df.nlargest(n_prov_stack, 'Total_Produksi').reset_index(drop=True)
    
    # Buat figure
    fig_stacked = go.Figure()
    
    # Stack setiap komoditas
    cumulative_height = np.zeros(n_prov_stack)
    
    for komoditas in KOMODITAS_LIST:
        heights = top_stack[komoditas].values
        
        fig_stacked.add_trace(go.Bar(
            x=top_stack['Provinsi'],
            y=heights,
            name=KOMODITAS_LABELS.get(komoditas, komoditas),
            marker_color=KOMODITAS_COLORS.get(komoditas, '#2ecc71'),
            marker_line=dict(color='white', width=0.5),
            base=cumulative_height,
            hovertemplate=f'<b>%{{x}}</b><br>{komoditas.replace("_", " ")}: %{{y:,.2f}} ribu ton<extra></extra>'
        ))
        
        cumulative_height += heights
    
    fig_stacked = apply_plotly_theme(
        fig_stacked,
        title=f"🎨 Komposisi Produksi per Komoditas - Top {n_prov_stack} Provinsi",
        height=550
    )
    
    fig_stacked.update_layout(
        barmode='relative',
        xaxis_title='Provinsi',
        yaxis_title='Produksi (ribu ton)',
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_stacked, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # ANALISIS PER WILAYAH
    # ========================================
    st.subheader("🗺️ Analisis Produksi per Wilayah")
    st.markdown("Perbandingan total produksi antar wilayah geografis di Indonesia.")
    
    # Agregasi per wilayah
    region_summary = df.groupby('Wilayah').agg({
        **{k: 'sum' for k in KOMODITAS_LIST},
        'Total_Produksi': 'sum',
        'Provinsi': 'count'
    }).reset_index()
    
    region_summary.rename(columns={'Provinsi': 'Jumlah_Provinsi'}, inplace=True)
    
    # Tambahkan rata-rata per provinsi
    for k in KOMODITAS_LIST:
        region_summary[f'{k}_per_prov'] = region_summary[k] / region_summary['Jumlah_Provinsi']
    
    region_summary['Total_per_prov'] = region_summary['Total_Produksi'] / region_summary['Jumlah_Provinsi']
    
    # Sort by total
    region_summary = region_summary.sort_values('Total_Produksi', ascending=False)
    
    # Display
    st.dataframe(
        region_summary[['Wilayah', 'Jumlah_Provinsi', 'Total_Produksi'] + KOMODITAS_LIST].style
        .background_gradient(subset=['Total_Produksi'] + KOMODITAS_LIST, cmap='Greens')
        .format({k: '{:,.0f}' for k in ['Total_Produksi'] + KOMODITAS_LIST}),
        use_container_width=True,
        hide_index=True
    )
    
    # Visualisasi per wilayah
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        # Bar chart total per wilayah
        fig_region_bar = px.bar(
            region_summary,
            x='Wilayah',
            y='Total_Produksi',
            color='Wilayah',
            title='Total Produksi per Wilayah',
            color_discrete_map=REGION_COLORS,
            text='Total_Produksi'
        )
        
        fig_region_bar = apply_plotly_theme(fig_region_bar, height=450)
        fig_region_bar.update_layout(
            xaxis_title='Wilayah',
            yaxis_title='Total Produksi (ribu ton)',
            showlegend=False
        )
        fig_region_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        
        st.plotly_chart(fig_region_bar, use_container_width=True)
    
    with col_w2:
        # Pie chart kontribusi wilayah
        fig_region_pie = px.pie(
            region_summary,
            values='Total_Produksi',
            names='Wilayah',
            title='Kontribusi Wilayah terhadap Total Produksi Nasional',
            color='Wilayah',
            color_discrete_map=REGION_COLORS,
            hole=0.4
        )
        
        fig_region_pie = apply_plotly_theme(fig_region_pie, height=450)
        fig_region_pie.update_traces(
            textposition='outside',
            textinfo='percent+label'
        )
        
        st.plotly_chart(fig_region_pie, use_container_width=True)
    
    # Stacked bar per wilayah
    st.markdown("### 📊 Komposisi Komoditas per Wilayah")
    
    fig_region_stack = go.Figure()
    
    for komoditas in KOMODITAS_LIST:
        fig_region_stack.add_trace(go.Bar(
            x=region_summary['Wilayah'],
            y=region_summary[komoditas],
            name=KOMODITAS_LABELS.get(komoditas, komoditas),
            marker_color=KOMODITAS_COLORS.get(komoditas, '#2ecc71')
        ))
    
    fig_region_stack = apply_plotly_theme(
        fig_region_stack,
        title='Komposisi Produksi Komoditas per Wilayah',
        height=500
    )
    fig_region_stack.update_layout(
        barmode='stack',
        xaxis_title='Wilayah',
        yaxis_title='Produksi (ribu ton)'
    )
    
    st.plotly_chart(fig_region_stack, use_container_width=True)

# ============================================================================
# BAGIAN 13: PAGE 3b - PETA DISTRIBUSI INDONESIA
# ============================================================================
# Halaman khusus peta distribusi produksi di Indonesia

elif "Page 3b: Peta" in page:
    # Header halaman
    st.markdown("""
    <div class="page-header">
        <h1>🗺️ Peta Distribusi Produksi Indonesia</h1>
        <p>Visualisasi geospasial produksi tanaman perkebunan di seluruh provinsi Indonesia</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Peta interaktif ini menunjukkan distribusi spasial produksi komoditas perkebunan di Indonesia. 
    Dengan visualisasi peta, kita dapat mengidentifikasi **kluster produksi**, **wilayah sentra**, 
    dan **kesenjangan regional** dengan lebih intuitif.
    """)
    
    st.markdown("---")
    
    # ========================================
    # PILIHAN KOMODITAS UNTUK PETA
    # ========================================
    st.subheader("🌍 Peta Sebaran Produksi Komoditas")
    
    col_map1, col_map2 = st.columns([2, 1])
    
    with col_map1:
        komoditas_peta = st.selectbox(
            "Pilih Komoditas untuk Ditampilkan:",
            ['Total_Produksi'] + KOMODITAS_LIST,
            format_func=lambda x: "🌾 Total Semua Komoditas" if x == 'Total_Produksi' else KOMODITAS_LABELS.get(x, x),
            key="peta_komoditas"
        )
    
    with col_map2:
        color_scale_map = st.selectbox(
            "Skala Warna:",
            ['Greens', 'YlGn', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis'],
            key="color_scale"
        )
    
    # Siapkan data peta
    df_map = df.copy()
    
    # Filter provinsi dengan koordinat valid
    df_map = df_map.dropna(subset=['Latitude', 'Longitude'])
    
    # ========================================
    # PETA 1: SCATTER GEO (WORLD VIEW)
    # ========================================
    st.markdown("### 🌏 Peta 1: Scatter Map (Asia-Pacific View)")
    
    fig_geo = px.scatter_geo(
        df_map,
        lat='Latitude',
        lon='Longitude',
        color=komoditas_peta,
        size=komoditas_peta,
        size_max=50,
        hover_name='Provinsi',
        hover_data={
            komoditas_peta: ':,.2f',
            'Total_Produksi': ':,.0f',
            'Wilayah': True,
            'Latitude': False,
            'Longitude': False
        },
        scope='asia',
        color_continuous_scale=color_scale_map,
        title=f'🌏 Distribusi {komoditas_peta.replace("_", " ")} di Indonesia',
        projection='natural earth'
    )
    
    fig_geo.update_geos(
        showcountries=True,
        countrycolor="rgba(46, 204, 113, 0.4)",
        showcoastlines=True,
        coastlinecolor="#2ecc71",
        showland=True,
        landcolor="rgba(26, 77, 46, 0.4)",
        showocean=True,
        oceancolor="rgba(10, 31, 20, 0.95)",
        showlakes=False,
        lataxis_range=[-12, 8],
        lonaxis_range=[92, 145]
    )
    
    fig_geo = apply_plotly_theme(fig_geo, height=650)
    fig_geo.update_layout(
        geo=dict(
            bgcolor='rgba(0, 0, 0, 0)',
            lakecolor='rgba(52, 152, 219, 0.3)'
        )
    )
    
    st.plotly_chart(fig_geo, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # PETA 2: HEATMAP DENSITY
    # ========================================
    st.markdown("### 🔥 Peta 2: Density Heatmap")
    st.markdown("Heatmap menunjukkan konsentrasi spasial dari produksi komoditas.")
    
    fig_density = go.Figure(go.Densitymapbox(
        lat=df_map['Latitude'],
        lon=df_map['Longitude'],
        z=df_map[komoditas_peta],
        radius=30,
        colorscale=color_scale_map,
        showscale=True,
        colorbar=dict(
            title=f'{komoditas_peta.replace("_", " ")}<br>(ribu ton)',
            tickfont=dict(color="white"),
            titlefont=dict(color="white")
        ),
        hovertemplate='<b>%{customdata[0]}</b><br>Produksi: %{z:,.2f} ribu ton<extra></extra>',
        customdata=df_map[['Provinsi']].values
    ))
    
    fig_density.update_layout(
        mapbox=dict(
            style='carto-darkmatter',
            center=dict(lat=-2.5, lon=118),
            zoom=4
        ),
        title=f'🔥 Density Heatmap: {komoditas_peta.replace("_", " ")}',
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8f5e9")
    )
    
    st.plotly_chart(fig_density, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # PETA 3: CHOROPLETH-STYLE DENGAN BUBBLES
    # ========================================
    st.markdown("### 🎯 Peta 3: Bubble Map dengan Kategorisasi Wilayah")
    st.markdown("Bubble size menunjukkan total produksi, warna menunjukkan wilayah geografis.")
    
    fig_bubble_map = px.scatter_geo(
        df_map,
        lat='Latitude',
        lon='Longitude',
        color='Wilayah',
        size='Total_Produksi',
        size_max=60,
        hover_name='Provinsi',
        hover_data={
            komoditas_peta: ':,.2f',
            'Total_Produksi': ':,.0f',
            'Latitude': False,
            'Longitude': False
        },
        color_discrete_map=REGION_COLORS,
        title=f'🎯 Peta Wilayah: Distribusi Provinsi Berdasarkan Region',
        scope='asia',
        projection='natural earth'
    )
    
    fig_bubble_map.update_geos(
        showcountries=True,
        countrycolor="rgba(46, 204, 113, 0.4)",
        showcoastlines=True,
        coastlinecolor="#2ecc71",
        showland=True,
        landcolor="rgba(26, 77, 46, 0.4)",
        showocean=True,
        oceancolor="rgba(10, 31, 20, 0.95)",
        lataxis_range=[-12, 8],
        lonaxis_range=[92, 145]
    )
    
    fig_bubble_map = apply_plotly_theme(fig_bubble_map, height=650)
    
    st.plotly_chart(fig_bubble_map, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # PETA 4: KOMODITAS DOMINAN PER PROVINSI
    # ========================================
    st.markdown("### 🏆 Peta 4: Komoditas Dominan per Provinsi")
    st.markdown("Setiap provinsi ditandai dengan warna komoditas dominan (produksi tertinggi).")
    
    # Buat kolom untuk warna komoditas dominan
    df_map['Dominant_Color'] = df_map['Komoditas_Dominan'].apply(
        lambda x: next((v for k, v in KOMODITAS_COLORS.items() if k.replace('_', ' ').title() in x), '#2ecc71')
    )
    
    # Buat peta dengan warna kustom
    fig_dominant = go.Figure()
    
    # Tambahkan scatter untuk setiap komoditas dominan
    for komoditas in KOMODITAS_LIST:
        label = KOMODITAS_LABELS.get(komoditas, komoditas)
        color = KOMODITAS_COLORS.get(komoditas, '#2ecc71')
        
        # Filter provinsi dengan komoditas dominan ini
        mask = df_map['Komoditas_Dominan'] == label
        subset = df_map[mask]
        
        if len(subset) > 0:
            fig_dominant.add_trace(go.Scattergeo(
                lat=subset['Latitude'],
                lon=subset['Longitude'],
                text=subset['Provinsi'],
                mode='markers+text',
                textposition='top center',
                marker=dict(
                    size=15,
                    color=color,
                    line=dict(color='white', width=1),
                    symbol='circle'
                ),
                name=label,
                hovertemplate=f'<b>%{{text}}</b><br>Komoditas Dominan: {label}<extra></extra>'
            ))
    
    fig_dominant.update_layout(
        geo=dict(
            scope='asia',
            projection_type='natural earth',
            showcountries=True,
            countrycolor="rgba(46, 204, 113, 0.4)",
            showcoastlines=True,
            coastlinecolor="#2ecc71",
            showland=True,
            landcolor="rgba(26, 77, 46, 0.4)",
            showocean=True,
            oceancolor="rgba(10, 31, 20, 0.95)",
            lataxis=dict(range=[-12, 8]),
            lonaxis=dict(range=[92, 145]),
            bgcolor='rgba(0,0,0,0)'
        ),
        title='🏆 Komoditas Dominan per Provinsi',
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8f5e9"),
        legend=dict(
            bgcolor="rgba(15, 42, 28, 0.8)",
            bordercolor="#2ecc71",
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig_dominant, use_container_width=True)
    
    # Statistik per komoditas dominan
    st.markdown("### 📊 Statistik Komoditas Dominan")
    
    dominant_stats = df['Komoditas_Dominan'].value_counts().reset_index()
    dominant_stats.columns = ['Komoditas Dominan', 'Jumlah Provinsi']
    dominant_stats['Persentase (%)'] = (dominant_stats['Jumlah Provinsi'] / len(df)) * 100
    
    col_ds1, col_ds2 = st.columns([1, 1])
    
    with col_ds1:
        st.dataframe(
            dominant_stats.style.format({'Persentase (%)': '{:,.1f}'}),
            use_container_width=True,
            hide_index=True
        )
    
    with col_ds2:
        fig_dom_pie = px.pie(
            dominant_stats,
            values='Jumlah Provinsi',
            names='Komoditas Dominan',
            hole=0.4,
            title='Distribusi Komoditas Dominan'
        )
        fig_dom_pie = apply_plotly_theme(fig_dom_pie, height=400)
        st.plotly_chart(fig_dom_pie, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # ANALISIS KLUSTER GEOGRAFIS
    # ========================================
    st.subheader("🎯 Analisis Kluster Geografis")
    st.markdown("Pengelompokan provinsi berdasarkan kedekatan geografis dan profil produksi.")
    
    # K-Means clustering
    n_clusters = st.slider(
        "Jumlah kluster:",
        min_value=2, max_value=8, value=4, step=1,
        key="n_clusters_map"
    )
    
    # Features untuk clustering
    features_cluster = df[KOMODITAS_LIST + ['Latitude', 'Longitude']].fillna(0).values
    
    # Normalisasi
    scaler_cluster = StandardScaler()
    features_scaled = scaler_cluster.fit_transform(features_cluster)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Warna kluster
    cluster_colors = px.colors.qualitative.Set1[:n_clusters]
    
    # Peta kluster
    fig_cluster = px.scatter_geo(
        df,
        lat='Latitude',
        lon='Longitude',
        color='Cluster',
        size='Total_Produksi',
        size_max=40,
        hover_name='Provinsi',
        hover_data={
            'Cluster': True,
            'Total_Produksi': ':,.0f',
            'Latitude': False,
            'Longitude': False
        },
        color_discrete_sequence=cluster_colors,
        title=f'🎯 Kluster Provinsi (K={n_clusters})',
        scope='asia',
        projection='natural earth'
    )
    
    fig_cluster.update_geos(
        showcountries=True,
        countrycolor="rgba(46, 204, 113, 0.4)",
        showcoastlines=True,
        coastlinecolor="#2ecc71",
        showland=True,
        landcolor="rgba(26, 77, 46, 0.4)",
        showocean=True,
        oceancolor="rgba(10, 31, 20, 0.95)",
        lataxis_range=[-12, 8],
        lonaxis_range=[92, 145]
    )
    
    fig_cluster = apply_plotly_theme(fig_cluster, height=650)
    
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Detail per kluster
    st.markdown(f"### 📋 Detail Kluster (K={n_clusters})")
    
    for c in range(n_clusters):
        cluster_data = df[df['Cluster'] == c].sort_values('Total_Produksi', ascending=False)
        
        with st.expander(f"🔵 Kluster {c+1} - {len(cluster_data)} Provinsi"):
            col_c1, col_c2 = st.columns([2, 1])
            
            with col_c1:
                st.dataframe(
                    cluster_data[['Provinsi', 'Wilayah', 'Total_Produksi', 'Komoditas_Dominan']].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col_c2:
                st.markdown(f"""
                **Statistik Kluster:**
                - Total Produksi: **{cluster_data['Total_Produksi'].sum():,.0f}**
                - Rata-rata: **{cluster_data['Total_Produksi'].mean():,.0f}**
                - Median: **{cluster_data['Total_Produksi'].median():,.0f}**
                - Dominan: **{cluster_data['Komoditas_Dominan'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'}**
                """)

# ============================================================================
# BAGIAN 14: PAGE 4 - ANALISIS KORELASI & REGRESI
# ============================================================================
# Halaman keempat: analisis korelasi dan model regresi

elif "Page 4: Korelasi" in page:
    # Header halaman
    st.markdown("""
    <div class="page-header">
        <h1>🔗 Analisis Korelasi & Regresi</h1>
        <p>Menganalisis hubungan antar komoditas dan membangun model prediktif</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Analisis korelasi mengukur kekuatan dan arah hubungan linier antar variabel, 
    sedangkan regresi memungkinkan kita memprediksi nilai satu variabel berdasarkan variabel lainnya.
    """)
    
    st.markdown("---")
    
    # ========================================
    # HEATMAP KORELASI
    # ========================================
    st.subheader("🔥 Heatmap Matriks Korelasi Pearson")
    st.markdown("""
    Matriks korelasi Pearson menunjukkan kekuatan hubungan linier antar komoditas.
    Nilai berkisar dari **-1 (korelasi negatif sempurna)** hingga **+1 (korelasi positif sempurna)**.
    """)
    
    # Hitung matriks korelasi
    corr_matrix = df[KOMODITAS_LIST].corr(method='pearson')
    
    # Pilihan colorscale
    colorscale_corr = st.selectbox(
        "Pilih Colorscale Heatmap:",
        ['RdYlGn', 'RdBu', 'Viridis', 'Plasma', 'Portland'],
        key="colorscale_corr"
    )
    
    # Buat heatmap
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[KOMODITAS_LABELS.get(c, c) for c in corr_matrix.columns],
        y=[KOMODITAS_LABELS.get(c, c) for c in corr_matrix.index],
        colorscale=colorscale_corr,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        hovertemplate='<b>%{x} vs %{y}</b><br>Korelasi: %{z:.3f}<extra></extra>',
        colorbar=dict(
            title="Korelasi Pearson",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1 (Neg Sempurna)', '-0.5', '0 (Netral)', '0.5', '+1 (Pos Sempurna)'],
            tickfont=dict(color="white"),
            titlefont=dict(color="white")
        )
    ))
    
    fig_corr = apply_plotly_theme(fig_corr, title="Matriks Korelasi Pearson Antar Komoditas", height=600)
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Interpretasi korelasi
    st.markdown("""
    <div class="info-box">
        <strong>📖 Interpretasi Heatmap:</strong>
        <ul style="margin: 8px 0 0 0; color: var(--text-secondary);">
            <li>🟢 <b>Hijau (nilai positif)</b>: Kedua komoditas cenderung diproduksi bersama</li>
            <li>🔴 <b>Merah (nilai negatif)</b>: Kedua komoditas cenderung berlawanan</li>
            <li>⚪ <b>Kuning/netral (mendekati 0)</b>: Tidak ada hubungan linier yang signifikan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================
    # KORELASI SPEARMAN (NON-PARAMETRIK)
    # ========================================
    st.subheader("📊 Matriks Korelasi Spearman (Non-Parametrik)")
    st.markdown("""
    Korelasi Spearman mengukur hubungan monotonik (tidak harus linier), lebih robust terhadap outlier 
    dibandingkan Pearson. Cocok untuk data yang tidak terdistribusi normal.
    """)
    
    # Hitung korelasi Spearman
    corr_spearman = df[KOMODITAS_LIST].corr(method='spearman')
    
    # Buat heatmap Spearman
    fig_spearman = go.Figure(data=go.Heatmap(
        z=corr_spearman.values,
        x=[KOMODITAS_LABELS.get(c, c) for c in corr_spearman.columns],
        y=[KOMODITAS_LABELS.get(c, c) for c in corr_spearman.index],
        colorscale=colorscale_corr,
        zmin=-1,
        zmax=1,
        text=corr_spearman.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorbar=dict(
            title="Korelasi Spearman",
            tickfont=dict(color="white"),
            titlefont=dict(color="white")
        )
    ))
    
    fig_spearman = apply_plotly_theme(fig_spearman, title="Matriks Korelasi Spearman Antar Komoditas", height=600)
    
    st.plotly_chart(fig_spearman, use_container_width=True)
    
    # Perbandingan Pearson vs Spearman
    st.markdown("### ⚖️ Perbandingan: Pearson vs Spearman")
    
    # Flatten matrices
    pairs_data = []
    for i, c1 in enumerate(KOMODITAS_LIST):
        for j, c2 in enumerate(KOMODITAS_LIST):
            if i < j:  # Only upper triangle
                pairs_data.append({
                    'Pair': f"{KOMODITAS_LABELS.get(c1, c1)} × {KOMODITAS_LABELS.get(c2, c2)}",
                    'Pearson': corr_matrix.iloc[i, j],
                    'Spearman': corr_spearman.iloc[i, j],
                    'Difference': abs(corr_matrix.iloc[i, j] - corr_spearman.iloc[i, j])
                })
    
    pairs_df = pd.DataFrame(pairs_data).sort_values('Pearson', key=abs, ascending=False)
    
    st.dataframe(
        pairs_df.style
        .format({'Pearson': '{:.3f}', 'Spearman': '{:.3f}', 'Difference': '{:.3f}'})
        .background_gradient(subset=['Pearson', 'Spearman'], cmap='RdYlGn', vmin=-1, vmax=1),
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    st.markdown("---")
    
    # ========================================
    # ANALISIS KORELASI DETAIL
    # ========================================
    st.subheader("🔬 Analisis Korelasi Detail dengan Uji Signifikansi")
    st.markdown("Pilih sepasang komoditas untuk analisis korelasi mendalam dengan uji statistik.")
    
    col_an1, col_an2 = st.columns(2)
    
    with col_an1:
        var1 = st.selectbox(
            "Variabel 1:",
            KOMODITAS_LIST,
            format_func=lambda x: KOMODITAS_LABELS.get(x, x),
            key="corr_var1"
        )
    
    with col_an2:
        var2 = st.selectbox(
            "Variabel 2:",
            KOMODITAS_LIST,
            format_func=lambda x: KOMODITAS_LABELS.get(x, x),
            index=1,
            key="corr_var2"
        )
    
    # Hitung statistik korelasi
    corr_detail = calculate_correlation_significance(df[var1], df[var2])
    
    # Tampilkan hasil
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        st.markdown(create_metric_card(
            icon="📐",
            value=f"{corr_detail['pearson_r']:.4f}",
            label="Pearson r",
            change=corr_detail['pearson_strength'],
            change_type="positive" if abs(corr_detail['pearson_r']) > 0.5 else "negative"
        ), unsafe_allow_html=True)
    
    with col_d2:
        p_val_str = f"{corr_detail['pearson_p']:.4f}" if corr_detail['pearson_p'] > 0.0001 else "< 0.0001"
        st.markdown(create_metric_card(
            icon="🎯",
            value=p_val_str,
            label="P-value (Pearson)",
            change="Signifikan" if corr_detail['pearson_significant'] else "Tidak Signifikan",
            change_type="positive" if corr_detail['pearson_significant'] else "negative"
        ), unsafe_allow_html=True)
    
    with col_d3:
        st.markdown(create_metric_card(
            icon="📊",
            value=f"{corr_detail['spearman_r']:.4f}",
            label="Spearman ρ",
            change=corr_detail['spearman_strength'],
            change_type="positive" if abs(corr_detail['spearman_r']) > 0.5 else "negative"
        ), unsafe_allow_html=True)
    
    with col_d4:
        p_val_sp = f"{corr_detail['spearman_p']:.4f}" if corr_detail['spearman_p'] > 0.0001 else "< 0.0001"
        st.markdown(create_metric_card(
            icon="🔬",
            value=p_val_sp,
            label="P-value (Spearman)",
            change="Signifikan" if corr_detail['spearman_significant'] else "Tidak Signifikan",
            change_type="positive" if corr_detail['spearman_significant'] else "negative"
        ), unsafe_allow_html=True)
    
    # Scatter plot detail
    fig_detail_scatter = px.scatter(
        df,
        x=var1,
        y=var2,
        color='Wilayah',
        size='Total_Produksi',
        hover_name='Provinsi',
        trendline='ols',
        title=f'Scatter Plot: {KOMODITAS_LABELS.get(var1)} vs {KOMODITAS_LABELS.get(var2)}',
        color_discrete_map=REGION_COLORS
    )
    
    fig_detail_scatter = apply_plotly_theme(fig_detail_scatter, height=550)
    
    st.plotly_chart(fig_detail_scatter, use_container_width=True)
    
    # Interpretasi
    if corr_detail['pearson_significant']:
        direction = "positif" if corr_detail['pearson_r'] > 0 else "negatif"
        strength = corr_detail['pearson_strength']
        st.success(f"""
        ✅ **Hubungan Signifikan (α=0.05)**
        
        Terdapat hubungan **{direction}** dengan kekuatan **{strength}** antara 
        {KOMODITAS_LABELS.get(var1)} dan {KOMODITAS_LABELS.get(var2)}.
        
        Koefisien determinasi (R²) = {corr_detail['pearson_r']**2:.4f}, artinya 
        **{corr_detail['pearson_r']**2 * 100:.2f}%** variasi pada {KOMODITAS_LABELS.get(var2)} 
        dapat dijelaskan oleh {KOMODITAS_LABELS.get(var1)}.
        """)
    else:
        st.warning(f"""
        ⚠️ **Hubungan Tidak Signifikan (α=0.05)**
        
        Tidak ada bukti statistik yang cukup untuk menyimpulkan adanya hubungan linier antara 
        {KOMODITAS_LABELS.get(var1)} dan {KOMODITAS_LABELS.get(var2)}.
        """)
    
    st.markdown("---")
    
    # ========================================
    # MODEL REGRESI LINEAR
    # ========================================
    st.subheader("🤖 Model Regresi Linier Berganda")
    st.markdown("""
    Membangun model regresi untuk memprediksi produksi satu komoditas berdasarkan komoditas lainnya.
    Ini berguna untuk memahami bagaimana komoditas saling terkait dalam ekosistem perkebunan.
    """)
    
    # Pilihan variabel target
    target_var = st.selectbox(
        "Variabel Target (yang diprediksi):",
        KOMODITAS_LIST,
        format_func=lambda x: KOMODITAS_LABELS.get(x, x),
        index=0,
        key="reg_target"
    )
    
    # Fitur otomatis: semua komoditas kecuali target
    feature_vars = [k for k in KOMODITAS_LIST if k != target_var]
    
    # Pilihan test size
    col_reg1, col_reg2 = st.columns(2)
    
    with col_reg1:
        test_size = st.slider(
            "Ukuran Test Set (%):",
            min_value=10, max_value=40, value=20, step=5,
            key="test_size"
        ) / 100
    
    with col_reg2:
        random_seed = st.number_input(
            "Random Seed:",
            min_value=0, max_value=1000, value=42,
            key="random_seed"
        )
    
    # Siapkan data
    X_reg = df[feature_vars].values
    y_reg = df[target_var].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg,
        test_size=test_size,
        random_state=random_seed
    )
    
    # Scaling
    scaler_reg = StandardScaler()
    X_train_scaled = scaler_reg.fit_transform(X_train)
    X_test_scaled = scaler_reg.transform(X_test)
    
    # Training model Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    
    # Prediksi
    y_pred_train = model_lr.predict(X_train_scaled)
    y_pred_test = model_lr.predict(X_test_scaled)
    
    # Hitung metrik
    metrics_train = {
        'MAE': mean_absolute_error(y_train, y_pred_train),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'R²': r2_score(y_train, y_pred_train),
        'MAPE': mean_absolute_percentage_error(y_train, y_pred_train + 1e-8) * 100
    }
    
    metrics_test = {
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'R²': r2_score(y_test, y_pred_test),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_test + 1e-8) * 100
    }
    
    # Tampilkan metrik
    st.markdown(f"### 📊 Metrik Evaluasi Model (Target: {KOMODITAS_LABELS.get(target_var)})")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(create_metric_card(
            icon="📏",
            value=f"{metrics_test['MAE']:.2f}",
            label="MAE (Test)",
            change=f"Train: {metrics_train['MAE']:.2f}",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(create_metric_card(
            icon="📐",
            value=f"{metrics_test['RMSE']:.2f}",
            label="RMSE (Test)",
            change=f"Train: {metrics_train['RMSE']:.2f}",
            change_type="positive"
        ), unsafe_allow_html=True)
    
    with col_m3:
        st.markdown(create_metric_card(
            icon="🎯",
            value=f"{metrics_test['R²']:.4f}",
            label="R² Score (Test)",
            change=f"Train: {metrics_train['R²']:.4f}",
            change_type="positive" if metrics_test['R²'] > 0.5 else "negative"
        ), unsafe_allow_html=True)
    
    with col_m4:
        st.markdown(create_metric_card(
            icon="📊",
            value=f"{metrics_test['MAPE']:.1f}%",
            label="MAPE (Test)",
            change=f"Train: {metrics_train['MAPE']:.1f}%",
            change_type="positive" if metrics_test['MAPE'] < 50 else "negative"
        ), unsafe_allow_html=True)
    
    # Interpretasi R²
    r2 = metrics_test['R²']
    if r2 > 0.7:
        r2_interpretation = "🌟 Model memiliki kemampuan prediksi yang sangat baik"
    elif r2 > 0.5:
        r2_interpretation = "✅ Model memiliki kemampuan prediksi yang baik"
    elif r2 > 0.3:
        r2_interpretation = "⚠️ Model memiliki kemampuan prediksi yang cukup"
    else:
        r2_interpretation = "❌ Model memiliki kemampuan prediksi yang lemah - perlu perbaikan"
    
    st.info(f"""
    **Interpretasi R² = {r2:.4f}:**
    
    {r2_interpretation}
    
    Artinya **{r2*100:.2f}%** variasi pada {KOMODITAS_LABELS.get(target_var)} dapat dijelaskan 
    oleh variabel-variabel prediktor dalam model.
    """)
    
    # Visualisasi Actual vs Predicted
    st.markdown("### 📈 Visualisasi Actual vs Predicted")
    
    fig_avp = go.Figure()
    
    # Training data
    fig_avp.add_trace(go.Scatter(
        x=y_train,
        y=y_pred_train,
        mode='markers',
        name='Training Data',
        marker=dict(size=10, color='#2ecc71', opacity=0.6, symbol='circle')
    ))
    
    # Test data
    fig_avp.add_trace(go.Scatter(
        x=y_test,
        y=y_pred_test,
        mode='markers',
        name='Test Data',
        marker=dict(size=12, color='#f1c40f', opacity=0.9, symbol='diamond', 
                   line=dict(color='white', width=1))
    ))
    
    # Perfect prediction line
    all_vals = np.concatenate([y_train, y_test, y_pred_train, y_pred_test])
    min_val, max_val = min(all_vals), max(all_vals)
    
    fig_avp.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction (y=x)',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))
    
    fig_avp = apply_plotly_theme(
        fig_avp,
        title=f"Actual vs Predicted: {KOMODITAS_LABELS.get(target_var)}",
        height=550
    )
    fig_avp.update_layout(
        xaxis_title=f"Nilai Aktual ({KOMODITAS_LABELS.get(target_var)})",
        yaxis_title=f"Nilai Prediksi ({KOMODITAS_LABELS.get(target_var)})"
    )
    
    st.plotly_chart(fig_avp, use_container_width=True)
    
    # Residual plot
    st.markdown("### 📉 Analisis Residual")
    
    residuals = y_test - y_pred_test
    
    fig_residual = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals vs Predicted', 'Distribution of Residuals'),
        horizontal_spacing=0.1
    )
    
    # Residual vs Predicted
    fig_residual.add_trace(
        go.Scatter(
            x=y_pred_test,
            y=residuals,
            mode='markers',
            marker=dict(color='#2ecc71', size=10),
            name='Residuals'
        ),
        row=1, col=1
    )
    
    fig_residual.add_hline(
        y=0,
        line_dash="dash",
        line_color="#e74c3c",
        row=1, col=1
    )
    
    # Distribution of residuals
    fig_residual.add_trace(
        go.Histogram(
            x=residuals,
            marker_color='#f1c40f',
            name='Residual Distribution',
            nbinsx=15
        ),
        row=1, col=2
    )
    
    fig_residual = apply_plotly_theme(fig_residual, height=400)
    fig_residual.update_layout(showlegend=False)
    fig_residual.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig_residual.update_yaxes(title_text="Residuals", row=1, col=1)
    fig_residual.update_xaxes(title_text="Residual Value", row=1, col=2)
    fig_residual.update_yaxes(title_text="Frequency", row=1, col=2)
    
    st.plotly_chart(fig_residual, use_container_width=True)
    
    # Koefisien regresi
    st.markdown("### ⚖️ Koefisien Regresi (Feature Importance)")
    
    coef_df = pd.DataFrame({
        'Fitur': [KOMODITAS_LABELS.get(f, f) for f in feature_vars],
        'Koefisien': model_lr.coef_,
        'Abs_Koefisien': np.abs(model_lr.coef_)
    }).sort_values('Abs_Koefisien', ascending=False)
    
    # Interpretasi koefisien
    coef_df['Interpretasi'] = coef_df['Koefisien'].apply(
        lambda x: '➕ Positif' if x > 0 else '➖ Negatif'
    )
    
    st.dataframe(
        coef_df[['Fitur', 'Koefisien', 'Interpretasi']].style
        .format({'Koefisien': '{:.4f}'}),
        use_container_width=True,
        hide_index=True
    )
    
    # Visualisasi koefisien
    fig_coef = px.bar(
        coef_df,
        x='Fitur',
        y='Koefisien',
        color='Koefisien',
        color_continuous_scale='RdYlGn',
        title='Koefisien Regresi (Feature Importance)',
        text='Koefisien'
    )
    
    fig_coef = apply_plotly_theme(fig_coef, height=450)
    fig_coef.update_traces(texttemplate='%{y:.3f}', textposition='outside')
    
    st.plotly_chart(fig_coef, use_container_width=True)
    
    # Persamaan regresi
    st.markdown("### 📝 Persamaan Regresi")
    
    intercept = model_lr.intercept_
    equation_parts = [f"ŷ = {intercept:.4f}"]
    
    for feat, coef in zip(feature_vars, model_lr.coef_):
        sign = "+" if coef > 0 else "-"
        equation_parts.append(f" {sign} {abs(coef):.4f} × {KOMODITAS_LABELS.get(feat, feat).split()[-1]}")
    
    equation = "".join(equation_parts)
    
    st.code(equation, language='text')

# ============================================================================
# BAGIAN 15: PAGE 4b - MACHINE LEARNING LANJUTAN
# ============================================================================
# Halaman lanjutan dengan berbagai model ML

elif "Page 4b: Machine Learning" in page:
    # Header halaman
    st.markdown("""
    <div class="page-header">
        <h1>🎯 Machine Learning Lanjutan</h1>
        <p>Perbandingan berbagai algoritma regresi untuk prediksi produksi komoditas</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Halaman ini membandingkan kinerja berbagai algoritma machine learning untuk prediksi 
    produksi komoditas perkebunan. Perbandingan model membantu memilih algoritma terbaik 
    untuk kasus spesifik ini.
    """)
    
    st.markdown("---")
    
    # ========================================
    # KONFIGURASI MODEL
    # ========================================
    st.subheader("⚙️ Konfigurasi Eksperimen")
    
    col_ml1, col_ml2, col_ml3 = st.columns(3)
    
    with col_ml1:
        target_ml = st.selectbox(
            "Variabel Target:",
            KOMODITAS_LIST,
            format_func=lambda x: KOMODITAS_LABELS.get(x, x),
            key="ml_target"
        )
    
    with col_ml2:
        test_size_ml = st.slider(
            "Ukuran Test Set (%):",
            min_value=10, max_value=40, value=20, step=5,
            key="ml_test_size"
        ) / 100
    
    with col_ml3:
        n_folds = st.slider(
            "Jumlah Fold (Cross-Validation):",
            min_value=3, max_value=10, value=5, step=1,
            key="ml_folds"
        )
    
    # Pilihan model
    st.markdown("### 🤖 Pilih Model untuk Dibandingkan")
    
    selected_models = st.multiselect(
        "Model yang akan dievaluasi:",
        [
            "Linear Regression",
            "Ridge Regression (L2)",
            "Lasso Regression (L1)",
            "ElasticNet",
            "Random Forest",
            "Gradient Boosting"
        ],
        default=["Linear Regression", "Ridge Regression (L2)", "Random Forest", "Gradient Boosting"]
    )
    
    if not selected_models:
        st.warning("⚠️ Silakan pilih minimal satu model.")
        st.stop()
    
    # Siapkan data
    feature_vars_ml = [k for k in KOMODITAS_LIST if k != target_ml]
    X_ml = df[feature_vars_ml].values
    y_ml = df[target_ml].values
    
    # Split data
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
        X_ml, y_ml,
        test_size=test_size_ml,
        random_state=42
    )
    
    # Scaling
    scaler_ml = StandardScaler()
    X_train_ml_sc = scaler_ml.fit_transform(X_train_ml)
    X_test_ml_sc = scaler_ml.transform(X_test_ml)
    
    # ========================================
    # TRAINING MODEL
    # ========================================
    st.markdown("---")
    st.subheader("🏋️ Hasil Training Model")
    
    # Dictionary untuk menyimpan hasil
    results = []
    models_dict = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model_name in enumerate(selected_models):
        status_text.text(f"🔄 Training model: {model_name}...")
        
        # Inisialisasi model
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Ridge Regression (L2)":
            model = Ridge(alpha=1.0, random_state=42)
        elif model_name == "Lasso Regression (L1)":
            model = Lasso(alpha=1.0, random_state=42, max_iter=10000)
        elif model_name == "ElasticNet":
            model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
        elif model_name == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == "Gradient Boosting":
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        
        # Training
        if model_name in ["Random Forest", "Gradient Boosting"]:
            # Tree-based models tidak perlu scaling
            model.fit(X_train_ml, y_train_ml)
            y_pred_train = model.predict(X_train_ml)
            y_pred_test = model.predict(X_test_ml)
        else:
            # Linear models menggunakan scaled data
            model.fit(X_train_ml_sc, y_train_ml)
            y_pred_train = model.predict(X_train_ml_sc)
            y_pred_test = model.predict(X_test_ml_sc)
        
        # Cross-validation
        try:
            if model_name in ["Random Forest", "Gradient Boosting"]:
                cv_scores = cross_val_score(
                    model, X_ml, y_ml,
                    cv=KFold(n_splits=n_folds, shuffle=True, random_state=42),
                    scoring='r2'
                )
            else:
                # Untuk linear models, perlu custom CV dengan scaling
                cv_scores = []
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                for train_idx, val_idx in kf.split(X_ml):
                    X_tr, X_val = X_ml[train_idx], X_ml[val_idx]
                    y_tr, y_val = y_ml[train_idx], y_ml[val_idx]
                    
                    sc = StandardScaler()
                    X_tr_sc = sc.fit_transform(X_tr)
                    X_val_sc = sc.transform(X_val)
                    
                    m = type(model)(**model.get_params())
                    m.fit(X_tr_sc, y_tr)
                    y_val_pred = m.predict(X_val_sc)
                    
                    cv_scores.append(r2_score(y_val, y_val_pred))
                cv_scores = np.array(cv_scores)
        except Exception as e:
            cv_scores = np.array([np.nan])
        
        # Hitung metrik
        train_metrics = {
            'MAE': mean_absolute_error(y_train_ml, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(y_train_ml, y_pred_train)),
            'R²': r2_score(y_train_ml, y_pred_train),
            'MAPE': mean_absolute_percentage_error(y_train_ml, y_pred_train + 1e-8) * 100
        }
        
        test_metrics = {
            'MAE': mean_absolute_error(y_test_ml, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test_ml, y_pred_test)),
            'R²': r2_score(y_test_ml, y_pred_test),
            'MAPE': mean_absolute_percentage_error(y_test_ml, y_pred_test + 1e-8) * 100
        }
        
        # Simpan hasil
        results.append({
            'Model': model_name,
            'Train MAE': train_metrics['MAE'],
            'Test MAE': test_metrics['MAE'],
            'Train RMSE': train_metrics['RMSE'],
            'Test RMSE': test_metrics['RMSE'],
            'Train R²': train_metrics['R²'],
            'Test R²': test_metrics['R²'],
            'CV R² Mean': np.mean(cv_scores),
            'CV R² Std': np.std(cv_scores),
            'Train MAPE': train_metrics['MAPE'],
            'Test MAPE': test_metrics['MAPE']
        })
        
        models_dict[model_name] = {
            'model': model,
            'y_pred_test': y_pred_test
        }
        
        # Update progress
        progress_bar.progress((idx + 1) / len(selected_models))
    
    progress_bar.empty()
    status_text.empty()
    
    # ========================================
    # TAMPILKAN HASIL
    # ========================================
    results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
    
    st.markdown("### 📊 Tabel Perbandingan Model")
    
    st.dataframe(
        results_df.style
        .highlight_max(subset=['Test R²', 'CV R² Mean'], color='#2ecc71')
        .highlight_min(subset=['Test MAE', 'Test RMSE', 'Test MAPE'], color='#2ecc71')
        .format({
            'Train MAE': '{:.3f}', 'Test MAE': '{:.3f}',
            'Train RMSE': '{:.3f}', 'Test RMSE': '{:.3f}',
            'Train R²': '{:.4f}', 'Test R²': '{:.4f}',
            'CV R² Mean': '{:.4f}', 'CV R² Std': '{:.4f}',
            'Train MAPE': '{:.2f}', 'Test MAPE': '{:.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Model terbaik
    best_model = results_df.iloc[0]
    
    st.success(f"""
    ### 🏆 Model Terbaik: {best_model['Model']}
    
    - **Test R²:** {best_model['Test R²']:.4f}
    - **Test MAE:** {best_model['Test MAE']:.3f}
    - **Test RMSE:** {best_model['Test RMSE']:.3f}
    - **CV R² (Mean ± Std):** {best_model['CV R² Mean']:.4f} ± {best_model['CV R² Std']:.4f}
    """)
    
    st.markdown("---")
    
    # ========================================
    # VISUALISASI PERBANDINGAN
    # ========================================
    st.subheader("📈 Visualisasi Perbandingan Model")
    
    # Bar chart R² Score
    fig_r2_compare = go.Figure()
    
    fig_r2_compare.add_trace(go.Bar(
        x=results_df['Model'],
        y=results_df['Train R²'],
        name='Train R²',
        marker_color='#2ecc71',
        text=results_df['Train R²'].round(3),
        textposition='outside'
    ))
    
    fig_r2_compare.add_trace(go.Bar(
        x=results_df['Model'],
        y=results_df['Test R²'],
        name='Test R²',
        marker_color='#f1c40f',
        text=results_df['Test R²'].round(3),
        textposition='outside'
    ))
    
    fig_r2_compare = apply_plotly_theme(
        fig_r2_compare,
        title="Perbandingan R² Score: Train vs Test",
        height=500
    )
    fig_r2_compare.update_layout(
        barmode='group',
        xaxis_title='Model',
        yaxis_title='R² Score'
    )
    
    st.plotly_chart(fig_r2_compare, use_container_width=True)
    
    # Radar chart perbandingan metrik
    st.markdown("### 🎯 Radar Chart Perbandingan Metrik")
    
    # Normalisasi metrik untuk radar
    radar_data = []
    for _, row in results_df.iterrows():
        radar_data.append({
            'Model': row['Model'],
            'R² Score': row['Test R²'],
            '1 - Normalized MAE': 1 - (row['Test MAE'] / results_df['Test MAE'].max()),
            '1 - Normalized RMSE': 1 - (row['Test RMSE'] / results_df['Test RMSE'].max()),
            '1 - Normalized MAPE': 1 - (row['Test MAPE'] / max(results_df['Test MAPE'].max(), 1)),
            'CV R²': row['CV R² Mean']
        })
    
    radar_df = pd.DataFrame(radar_data)
    
    fig_radar = go.Figure()
    
    for idx, row in radar_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['R² Score'], row['1 - Normalized MAE'], row['1 - Normalized RMSE'], 
               row['1 - Normalized MAPE'], row['CV R²'], row['R² Score']],
            theta=['R² Score', '1-Norm MAE', '1-Norm RMSE', '1-Norm MAPE', 'CV R²', 'R² Score'],
            fill='toself',
            name=row['Model'],
            opacity=0.6
        ))
    
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(10, 31, 20, 0.5)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                color="#e8f5e9"
            ),
            angularaxis=dict(color="#e8f5e9")
        ),
        showlegend=True,
        title="🎯 Radar Chart: Perbandingan Performa Model",
        height=550,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8f5e9")
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # FEATURE IMPORTANCE
    # ========================================
    st.subheader("🔍 Feature Importance Analysis")
    st.markdown("Analisis pentingnya fitur untuk setiap model.")
    
    model_for_fi = st.selectbox(
        "Pilih model untuk analisis feature importance:",
        selected_models,
        key="fi_model"
    )
    
    selected_model_obj = models_dict[model_for_fi]['model']
    
    # Hitung feature importance
    if hasattr(selected_model_obj, 'coef_'):
        # Linear models
        importance = np.abs(selected_model_obj.coef_)
        importance_type = "Absolute Coefficient"
    elif hasattr(selected_model_obj, 'feature_importances_'):
        # Tree-based models
        importance = selected_model_obj.feature_importances_
        importance_type = "Feature Importance (Gini)"
    else:
        st.warning("Model ini tidak mendukung analisis feature importance.")
        importance = None
        importance_type = None
    
    if importance is not None:
        fi_df = pd.DataFrame({
            'Fitur': [KOMODITAS_LABELS.get(f, f) for f in feature_vars_ml],
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        fi_df['Rank'] = range(1, len(fi_df) + 1)
        
        # Tampilkan tabel
        st.dataframe(fi_df, use_container_width=True, hide_index=True)
        
        # Visualisasi
        fig_fi = px.bar(
            fi_df,
            x='Importance',
            y='Fitur',
            orientation='h',
            title=f'Feature Importance - {model_for_fi} ({importance_type})',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig_fi = apply_plotly_theme(fig_fi, height=400)
        fig_fi.update_layout(
            xaxis_title=importance_type,
            yaxis_title='Fitur'
        )
        
        st.plotly_chart(fig_fi, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # PREDIKSI MODEL TERBAIK
    # ========================================
    st.subheader("🎯 Prediksi Model Terbaik")
    
    best_model_name = best_model['Model']
    y_pred_best = models_dict[best_model_name]['y_pred_test']
    
    # Actual vs Predicted plot
    fig_best_avp = go.Figure()
    
    fig_best_avp.add_trace(go.Scatter(
        x=y_test_ml,
        y=y_pred_best,
        mode='markers',
        marker=dict(
            size=12,
            color='#2ecc71',
            line=dict(color='#f1c40f', width=2)
        ),
        name=f'{best_model_name} Predictions',
        hovertemplate='Aktual: %{x:.2f}<br>Prediksi: %{y:.2f}<extra></extra>'
    ))
    
    # Perfect prediction line
    min_v, max_v = min(y_test_ml.min(), y_pred_best.min()), max(y_test_ml.max(), y_pred_best.max())
    fig_best_avp.add_trace(go.Scatter(
        x=[min_v, max_v],
        y=[min_v, max_v],
        mode='lines',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig_best_avp = apply_plotly_theme(
        fig_best_avp,
        title=f"🏆 Actual vs Predicted - {best_model_name}",
        height=550
    )
    fig_best_avp.update_layout(
        xaxis_title='Nilai Aktual',
        yaxis_title='Nilai Prediksi'
    )
    
    st.plotly_chart(fig_best_avp, use_container_width=True)
    
    # ========================================
    # KESIMPULAN
    # ========================================
    st.subheader("💡 Kesimpulan Analisis")
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>🔍 Temuan Utama:</strong>
        <ul>
            <li>Model <b>{best_model['Model']}</b> menunjukkan performa terbaik dengan R² test sebesar 
                <b>{best_model['Test R²']:.4f}</b>.</li>
            <li>Cross-validation menunjukkan stabilitas model dengan mean R² = <b>{best_model['CV R² Mean']:.4f}</b> 
                (± {best_model['CV R² Std']:.4f}).</li>
            <li>Perbandingan model membantu memahami trade-off antara bias dan variance.</li>
            <li>Feature importance memberikan insight tentang komoditas mana yang paling 
                berpengaruh dalam prediksi.</li>
        </ul>
    </div>
    
    <div class="recommend-box">
        <strong>🎯 Rekomendasi:</strong>
        <ul>
            <li>Gunakan model <b>{best_model['Model']}</b> untuk prediksi di masa depan.</li>
            <li>Pertimbangkan untuk menambah fitur tambahan (luas lahan, iklim) untuk meningkatkan R².</li>
            <li>Lakukan hyperparameter tuning untuk mengoptimalkan model lebih lanjut.</li>
            <li>Validasi dengan data dari tahun yang berbeda untuk memastikan generalisasi model.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# BAGIAN 16: PAGE 5 - INSIGHTS & REKOMENDASI
# ============================================================================
# Halaman terakhir: insight dan rekomendasi strategis

elif "Page 5: Insights" in page:
    # Header halaman
    st.markdown("""
    <div class="page-header">
        <h1>💡 Insights & Rekomendasi Strategis</h1>
        <p>Kesimpulan mendalam dari analisis data dan rekomendasi implementatif untuk pemangku kepentingan</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Halaman ini menyajikan sintesis dari seluruh analisis yang telah dilakukan, dengan fokus pada 
    insight yang actionable dan rekomendasi yang dapat diimplementasikan oleh pemerintah, pelaku usaha, 
    dan masyarakat untuk pengembangan sektor perkebunan Indonesia.
    """)
    
    st.markdown("---")
    
    # ========================================
    # RINGKASAN EKSEKUTIF
    # ========================================
    st.subheader("📋 Ringkasan Eksekutif")
    
    # Hitung metrik kunci untuk ringkasan
    top_prov_sawit = df.loc[df['Kelapa_Sawit'].idxmax(), 'Provinsi']
    top_val_sawit = df['Kelapa_Sawit'].max()
    pct_sawit_total = (df['Kelapa_Sawit'].sum() / df['Total_Produksi'].sum()) * 100
    
    top_wilayah = df.groupby('Wilayah')['Total_Produksi'].sum().idxmax()
    top_wilayah_val = df.groupby('Wilayah')['Total_Produksi'].sum().max()
    
    # Gini coefficient untuk ketimpangan
    gini_total = calculate_gini_coefficient(df['Total_Produksi'])
    
    st.markdown(f"""
    <div class="info-box" style="font-size: 1.05em;">
        <p>Analisis terhadap data produksi tanaman perkebunan di <b>{len(df)} provinsi Indonesia</b> 
        menunjukkan beberapa temuan kunci:</p>
        <ul>
            <li>🌴 <b>Kelapa Sawit</b> mendominasi dengan kontribusi <b>{pct_sawit_total:.1f}%</b> 
                dari total produksi nasional</li>
            <li>🏆 <b>{top_prov_sawit}</b> adalah produsen sawit terbesar ({top_val_sawit:,.0f} ribu ton)</li>
            <li>🗺️ <b>Wilayah {top_wilayah}</b> adalah sentra produksi terbesar ({top_wilayah_val:,.0f} ribu ton)</li>
            <li>⚖️ Koefisien Gini = <b>{gini_total:.3f}</b> menunjukkan ketimpangan distribusi produksi 
                {"tinggi" if gini_total > 0.5 else "sedang" if gini_total > 0.3 else "rendah"}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================
    # 5 INSIGHT MENDALAM
    # ========================================
    st.subheader("🔍 5 Insight Mendalam dari Analisis Data")
    
    # Insight 1: Dominasi Kelapa Sawit
    st.markdown("""
    <div class="insight-box">
        <h3 style="color: var(--emerald-light) !important; margin-top: 0;">
            💡 Insight #1: Dominasi Mutlak Kelapa Sawit dalam Produksi Nasional
        </h3>
        <p>Kelapa Sawit mendominasi <b>{:.1f}%</b> dari total produksi perkebunan nasional, dengan 
        <b>{}</b> sebagai produsen utama ({:,.0f} ribu ton). Konsentrasi ini menunjukkan bahwa Indonesia 
        sangat bergantung pada satu komoditas untuk pendapatan devisa sektor perkebunan. Dominasi ini 
        membawa risiko ekonomi karena ketergantungan pada fluktuasi harga CPO di pasar global.</p>
        <p><b>Implikasi:</b> Ketergantungan pada satu komoditas membuat ekonomi perkebunan rentan 
        terhadap shock eksternal seperti kebijakan EUDR (EU Deforestation Regulation) atau kampanye 
        anti-sawit internasional.</p>
    </div>
    """.format(pct_sawit_total, top_prov_sawit, top_val_sawit), unsafe_allow_html=True)
    
    # Insight 2: Ketimpangan Regional
    st.markdown(f"""
    <div class="insight-box">
        <h3 style="color: var(--emerald-light) !important; margin-top: 0;">
            💡 Insight #2: Ketimpangan Produksi yang Tinggi Antar Provinsi
        </h3>
        <p>Koefisien Gini sebesar <b>{gini_total:.3f}</b> menunjukkan ketimpangan produksi yang signifikan 
        antar provinsi. Provinsi-provinsi di <b>Sumatera dan Kalimantan</b> mendominasi produksi, sementara 
        provinsi di Indonesia Timur (Maluku, Papua) memiliki produksi yang sangat minim.</p>
        <p><b>Data pendukung:</b> 5 provinsi teratas menguasai lebih dari 60% total produksi nasional, 
        sementara 10 provinsi terbawah hanya berkontribusi kurang dari 2%.</p>
        <p><b>Implikasi:</b> Ketimpangan ini mencerminkan kesenjangan ekonomi antar wilayah dan perlunya 
        redistribusi investasi infrastruktur perkebunan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Insight 3: Spesialisasi Regional
    st.markdown("""
    <div class="insight-box">
        <h3 style="color: var(--emerald-light) !important; margin-top: 0;">
            💡 Insight #3: Spesialisasi Regional yang Jelas per Komoditas
        </h3>
        <p>Analisis komoditas dominan menunjukkan pola spesialisasi regional yang konsisten dengan 
        kondisi agroklimat:</p>
        <ul>
            <li>🌴 <b>Kelapa Sawit</b>: Sumatera (Riau, Sumut) dan Kalimantan (Kalteng, Kalbar)</li>
            <li>☕ <b>Kopi</b>: Dataran tinggi Sumatera (Aceh, Sumut) dan Sulawesi</li>
            <li>🌳 <b>Karet</b>: Sumatera Selatan dan Jambi</li>
            <li>🎋 <b>Tebu</b>: Jawa Timur (dominan), Lampung, Jawa Tengah</li>
            <li>🍵 <b>Teh</b>: Jawa Barat (sentra utama)</li>
            <li>🍫 <b>Kakao</b>: Sulawesi Tengah, Sulawesi Tenggara, Sulawesi Selatan</li>
        </ul>
        <p><b>Implikasi:</b> Spesialisasi ini harus dipertahankan dan ditingkatkan melalui 
        program pengembangan klaster komoditas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Insight 4: Diversifikasi rendah
    avg_hhi = df['HHI_Index'].mean()
    high_specialization = (df['HHI_Index'] > 0.40).sum()
    
    st.markdown(f"""
    <div class="insight-box">
        <h3 style="color: var(--emerald-light) !important; margin-top: 0;">
            💡 Insight #4: Tingkat Diversifikasi yang Rendah
        </h3>
        <p>Rata-rata HHI Index sebesar <b>{avg_hhi:.3f}</b> menunjukkan bahwa sebagian besar provinsi 
        memiliki diversifikasi komoditas yang rendah. Sebanyak <b>{high_specialization} provinsi</b> 
        (dari {len(df)}) memiliki HHI > 0.40 yang menandakan spesialisasi sangat tinggi pada 1-2 komoditas saja.</p>
        <p><b>Risiko:</b> Spesialisasi tinggi membuat provinsi rentan terhadap fluktuasi harga komoditas 
        tertentu. Jika harga sawit jatuh, provinsi seperti Riau akan terdampak signifikan.</p>
        <p><b>Contoh positif:</b> Provinsi seperti Jawa Barat dan Jawa Timur memiliki HHI lebih rendah 
        karena diversifikasi yang lebih baik (tebu, teh, kopi, dll).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Insight 5: Potensi Indonesia Timur
    indonesia_timur = df[df['Wilayah'].isin(['Maluku', 'Papua', 'Bali & Nusa Tenggara', 'Sulawesi'])]
    pct_timur = (indonesia_timur['Total_Produksi'].sum() / df['Total_Produksi'].sum()) * 100
    
    st.markdown(f"""
    <div class="insight-box">
        <h3 style="color: var(--emerald-light) !important; margin-top: 0;">
            💡 Insight #5: Potensi Belum Tergarap di Indonesia Timur
        </h3>
        <p>Wilayah Indonesia Timur (Maluku, Papua, Bali-NTT, Sulawesi) baru berkontribusi <b>{pct_timur:.1f}%</b> 
        terhadap total produksi nasional, padahal memiliki potensi agroklimat yang baik untuk berbagai komoditas.</p>
        <p><b>Contoh potensi:</b></p>
        <ul>
            <li>Papua: Cocok untuk kakao, kelapa, dan kopi robusta</li>
            <li>Maluku: Potensi pala, cengkeh, dan kelapa</li>
            <li>Sulawesi: Sudah menjadi sentra kakao nasional</li>
        </ul>
        <p><b>Implikasi:</b> Pengembangan infrastruktur (jalan, pelabuhan, listrik) di Indonesia Timur 
        akan membuka potensi perkebunan yang sangat besar.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================
    # 5 REKOMENDASI IMPLEMENTATIF
    # ========================================
    st.subheader("🎯 5 Rekomendasi Implementatif")
    
    st.markdown("""
    <div class="recommend-box">
        <h3 style="color: var(--gold) !important; margin-top: 0;">
            🎯 Rekomendasi #1: Program Diversifikasi Komoditas Strategis
        </h3>
        <p><b>Latar Belakang:</b> Dominasi kelapa sawit (65%+ produksi) dan rendahnya diversifikasi 
        (HHI rata-rata 0.43) menciptakan kerentanan ekonomi.</p>
        <p><b>Aksi yang Diusulkan:</b></p>
        <ul>
            <li>Program tumpang sari (intercropping) sawit dengan kakao atau kopi di lahan yang sesuai</li>
            <li>Insentif fiskal (subsidi, tax break) untuk petani yang menanam komoditas selain sawit</li>
            <li>Pengembangan klaster komoditas alternatif di setiap provinsi</li>
            <li>Program "Satu Desa Satu Komoditas Unggulan" berbasis potensi agroklimat</li>
        </ul>
        <p><b>Target:</b> Menurunkan HHI rata-rata nasional dari 0.43 menjadi < 0.35 dalam 5 tahun.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommend-box">
        <h3 style="color: var(--gold) !important; margin-top: 0;">
            🎯 Rekomendasi #2: Percepatan Pembangunan Infrastruktur di Indonesia Timur
        </h3>
        <p><b>Latar Belakang:</b> Indonesia Timur baru menyumbang < 15% produksi nasional padahal 
        memiliki potensi lahan dan agroklimat yang besar.</p>
        <p><b>Aksi yang Diusulkan:</b></p>
        <ul>
            <li>Pembangunan jalan trans-Papua dan konektivitas antar pulau di Maluku</li>
            <li>Pembangunan pelabuhan khusus komoditas pertanian di Sorong, Ambon, dan Kupang</li>
            <li>Investasi cold storage dan fasilitas pasca panen</li>
            <li>Pengembangan bandara kargo untuk komoditas bernilai tinggi (kopi specialty, vanili)</li>
        </ul>
        <p><b>Target:</b> Meningkatkan kontribusi Indonesia Timur menjadi > 20% dalam 10 tahun.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommend-box">
        <h3 style="color: var(--gold) !important; margin-top: 0;">
            🎯 Rekomendasi #3: Sertifikasi Sustainability dan Traceability
        </h3>
        <p><b>Latar Belakang:</b> Tekanan regulasi internasional (EUDR, US UFLPA) menuntut produk 
        perkebunan Indonesia memiliki sertifikasi sustainability dan traceability yang kuat.</p>
        <p><b>Aksi yang Diusulkan:</b></p>
        <ul>
            <li>Percepatan sertifikasi ISPO (Indonesia Sustainable Palm Oil) untuk 100% produsen sawit</li>
            <li>Pengembangan sistem traceability digital berbasis blockchain</li>
            <li>Pelatihan dan pendampingan untuk petani rakyat agar memenuhi standar sustainability</li>
            <li>Kolaborasi dengan buyer internasional untuk memfasilitasi sertifikasi</li>
        </ul>
        <p><b>Target:</b> 100% ekspor perkebunan tersertifikasi ISPO/RSPO pada 2030.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommend-box">
        <h3 style="color: var(--gold) !important; margin-top: 0;">
            🎯 Rekomendasi #4: Digitalisasi dan Sistem Informasi Terintegrasi
        </h3>
        <p><b>Latar Belakang:</b> Saat ini belum ada sistem informasi nasional yang terintegrasi 
        untuk monitoring produksi perkebunan secara real-time.</p>
        <p><b>Aksi yang Diusulkan:</b></p>
        <ul>
            <li>Pengembangan dashboard nasional berbasis data real-time (seperti dasbor ini tapi skala nasional)</li>
            <li>Implementasi IoT sensor untuk monitoring kondisi kebun</li>
            <li>Sistem early warning untuk cuaca, hama, dan penyakit</li>
            <li>Mobile app untuk petani dengan informasi harga pasar dan best practices</li>
            <li>Integrasi dengan sistem satelit untuk monitoring deforestasi</li>
        </ul>
        <p><b>Target:</b> Seluruh provinsi terhubung dalam sistem informasi terintegrasi pada 2028.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommend-box">
        <h3 style="color: var(--gold) !important; margin-top: 0;">
            🎯 Rekomendasi #5: Hilirisasi dan Peningkatan Nilai Tambah
        </h3>
        <p><b>Latar Belakang:</b> Sebagian besar produk perkebunan Indonesia diekspor dalam bentuk 
        bahan mentah, sehingga nilai tambahnya dinikmati negara lain.</p>
        <p><b>Aksi yang Diusulkan:</b></p>
        <ul>
            <li>Pembangunan industri pengolahan di sentra produksi (CPO → minyak goreng, oleochemical)</li>
            <li>Pengembangan industri cokelat dari hulu ke hilir (kakao → produk akhir)</li>
            <li>Incentive untuk investasi industri downstreaming</li>
            <li>Branding dan positioning produk perkebunan Indonesia di pasar premium global</li>
            <li>Pengembangan kopi specialty dan teh premium dengan sertifikasi geografis (IG)</li>
        </ul>
        <p><b>Target:</b> Meningkatkan rasio ekspor produk olahan dari 30% menjadi > 60% pada 2030.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================
    # ROADMAP IMPLEMENTASI
    # ========================================
    st.subheader("🗺️ Roadmap Implementasi 5 Tahun")
    st.markdown("Timeline implementasi rekomendasi dalam horizon 5 tahun ke depan.")
    
    # Buat roadmap visual
    roadmap_data = {
        'Tahun': ['2026', '2027', '2028', '2029', '2030'],
        'Diversifikasi': [
            'Pilot project 10 provinsi',
            'Expand ke 20 provinsi',
            'Program nasional tumpang sari',
            'Evaluasi dan optimasi',
            'Target HHI < 0.35'
        ],
        'Infrastruktur': [
            'Studi kelayakan',
            'Pembangunan tahap 1',
            'Pembangunan tahap 2',
            'Operasionalisasi',
            'Kontribusi > 20%'
        ],
        'Sustainability': [
            'Baseline & audit',
            'Sertifikasi 50% produsen',
            'Sertifikasi 75% produsen',
            'Traceability system',
            '100% tersertifikasi'
        ],
        'Digitalisasi': [
            'Pilot dashboard',
            'IoT deployment',
            'Mobile app launch',
            'Integrasi sistem',
            'Full integration'
        ],
        'Hilirisasi': [
            'Masterplan industri',
            'Incentive & FDI',
            'Pabrik mulai beroperasi',
            'Scale up produksi',
            'Rasio olahan > 60%'
        ]
    }
    
    roadmap_df = pd.DataFrame(roadmap_data)
    
    st.dataframe(
        roadmap_df.style.background_gradient(subset=['Diversifikasi', 'Infrastruktur', 'Sustainability', 'Digitalisasi', 'Hilirisasi'], cmap='YlGn'),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # ========================================
    # KESIMPULAN AKHIR
    # ========================================
    st.subheader("📝 Kesimpulan Akhir")
    
    st.markdown("""
    <div class="info-box" style="font-size: 1.05em;">
        <p>Analisis komprehensif terhadap data produksi perkebunan Indonesia mengungkap bahwa 
        sektor ini memiliki <b>potensi besar namun menghadapi tantangan struktural</b> berupa:</p>
        <ol>
            <li><b>Ketergantungan berlebihan</b> pada satu komoditas (kelapa sawit)</li>
            <li><b>Ketimpangan regional</b> yang signifikan antara Indonesia Barat dan Timur</li>
            <li><b>Rendahnya diversifikasi</b> yang meningkatkan kerentanan ekonomi</li>
            <li><b>Belum optimalnya hilirisasi</b> sehingga nilai tambah belum maksimal</li>
        </ol>
        <p>Dengan implementasi rekomendasi strategis yang tepat, sektor perkebunan Indonesia 
        berpotensi menjadi <b>motor penggerak ekonomi berkelanjutan</b> yang memberikan manfaat 
        luas bagi seluruh stakeholder: petani, industri, pemerintah, dan masyarakat.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quote penutup
    st.markdown("""
    <div style="text-align: center; padding: 30px; margin: 20px 0; 
                background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(241, 196, 15, 0.1));
                border-radius: 15px; border: 1px solid var(--border);">
        <p style="font-size: 1.3em; color: var(--emerald-light); font-style: italic; margin: 0;">
            "Pertanian adalah sektor yang paling mulia, karena dari situlah kehidupan dimulai 
            dan keberlangsungan peradaban dipertahankan."
        </p>
        <p style="color: var(--gold); margin-top: 15px; font-weight: 600;">
            — Filosofi Agrikultur Indonesia 🌾
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# BAGIAN 17: TENTANG APLIKASI
# ============================================================================
# Halaman informasi tentang aplikasi

elif "Tentang Aplikasi" in page:
    # Header halaman
    st.markdown("""
    <div class="page-header">
        <h1>📚 Tentang Aplikasi</h1>
        <p>Informasi lengkap tentang dasbor ini, teknologi yang digunakan, dan cara deployment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Deskripsi aplikasi
    st.subheader("🌿 Deskripsi Aplikasi")
    st.markdown("""
    **Dasbor Produksi Perkebunan Indonesia** adalah aplikasi web interaktif berbasis Streamlit 
    yang dirancang untuk menganalisis dan memvisualisasikan data produksi tanaman perkebunan 
    di seluruh provinsi Indonesia. Aplikasi ini dibangun sebagai tugas **UAS Visualisasi Data** 
    dengan tema *Sains Data & Pertanian Indonesia*.
    
    ### ✨ Fitur Utama
    
    - 📊 **5+ Halaman Terstruktur**: Overview, Data Cleaning, EDA 3D, Peta, Analisis Korelasi & Regresi, ML Lanjutan, Insights
    - 🧊 **5 Visualisasi 3D Interaktif** menggunakan Plotly:
      - Scatter 3D (3 komoditas)
      - Surface Plot (Topografi Produksi)
      - Mesh3D Bar Chart (Top N Provinsi)
      - Bubble 3D Chart
      - Stacked Bar Komposisi
    - 🗺️ **4 Jenis Peta Interaktif**:
      - Scatter Geo (Asia-Pacific view)
      - Density Heatmap
      - Bubble Map Wilayah
      - Komoditas Dominan
    - 🔥 **Heatmap Korelasi** dengan analisis signifikansi statistik
    - 🤖 **Multiple ML Models**: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting
    - 🎯 **Clustering Geografis** dengan K-Means
    - 🎨 **Tema Modern Dark Mode** dengan aksen emerald & emas
    
    ### 🎨 Desain Visual
    
    - **Palet Warna**: Emerald green (#2ecc71) + Gold (#f1c40f) + Dark background
    - **Tipografi**: Poppins (headings), Inter (body), JetBrains Mono (numbers)
    - **Animasi**: Hover effects, gradient animations, smooth transitions
    - **Layout**: Responsive design, modern cards, glass-morphism
    """)
    
    st.markdown("---")
    
    # Teknologi yang digunakan
    st.subheader("🛠️ Teknologi yang Digunakan")
    
    tech_data = {
        'Teknologi': [
            'Streamlit', 'Pandas', 'NumPy', 'Plotly', 
            'scikit-learn', 'SciPy', 'Python'
        ],
        'Versi': ['1.36.0', '2.2.2', '1.26.4', '5.22.0', '1.5.0', '1.12.0', '3.10+'],
        'Fungsi': [
            'Framework web app interaktif',
            'Manipulasi dan analisis data tabular',
            'Komputasi numerik dan array',
            'Visualisasi interaktif 2D & 3D',
            'Machine Learning (regression, clustering)',
            'Statistik ilmiah (korelasi, testing)',
            'Bahasa pemrograman utama'
        ],
        'Kategori': [
            'Framework', 'Data', 'Data', 'Visualization',
            'ML/AI', 'Statistics', 'Core'
        ]
    }
    
    tech_df = pd.DataFrame(tech_data)
    
    st.dataframe(
        tech_df.style.background_gradient(subset=['Kategori'], cmap='Greens'),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Struktur repositori
    st.subheader("📁 Struktur Repositori GitHub")
    
    st.code("""
dasbor-perkebunan/
│
├── 📄 app.py                    # Kode utama aplikasi Streamlit
├── 📄 produksi_tanaman.csv      # Dataset produksi perkebunan
├── 📄 requirements.txt          # Daftar dependensi Python
├── 📄 README.md                 # Dokumentasi proyek
│
├── 📁 .streamlit/
│   └── 📄 config.toml           # Konfigurasi tema Streamlit (opsional)
│
├── 📁 assets/                   # Folder untuk gambar/logo (opsional)
│   └── 🖼️ logo.png
│
└── 📁 notebooks/                # Jupyter notebooks (eksplorasi)
    └── 📓 exploration.ipynb
    """, language='text')
    
    st.markdown("---")
    
    # Cara menjalankan
    st.subheader("🚀 Cara Menjalankan Lokal")
    
    st.markdown("""
    ### 1️⃣ Clone Repositori
    
    ```bash
    git clone https://github.com/username/dasbor-perkebunan.git
    cd dasbor-perkebunan
    ```
    
    ### 2️⃣ Buat Virtual Environment (Direkomendasikan)
    
    **Linux/macOS:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    
    **Windows:**
    ```cmd
    python -m venv venv
    venv\\Scripts\\activate
    ```
    
    ### 3️⃣ Install Dependensi
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ### 4️⃣ Jalankan Aplikasi
    
    ```bash
    streamlit run app.py
    ```
    
    Aplikasi akan terbuka di `http://localhost:8501` 🎉
    """)
    
    st.markdown("---")
    
    # Deployment
    st.subheader("☁️ Deployment ke Streamlit Cloud")
    
    st.markdown("""
    ### Langkah-langkah Deployment:
    
    1. **Push kode ke GitHub**
       ```bash
       git add .
       git commit -m "Initial commit: Dasbor Produksi Perkebunan"
       git push origin main
       ```
    
    2. **Buka Streamlit Cloud**
       - Kunjungi [share.streamlit.io](https://share.streamlit.io)
       - Login dengan akun GitHub
    
    3. **Buat New App**
       - Klik **"New App"**
       - Pilih:
         - **Repository**: `username/dasbor-perkebunan`
         - **Branch**: `main`
         - **Main file path**: `app.py`
       - Klik **"Advanced settings"** untuk konfigurasi Python version
    
    4. **Deploy**
       - Klik **"Deploy"**
       - Tunggu proses build (biasanya 3-5 menit)
       - Aplikasi akan tersedia di `https://your-app.streamlit.app`
    
    ### 📝 File `config.toml` (Opsional)
    
    Buat folder `.streamlit` dengan file `config.toml` untuk kustomisasi tema:
    
    ```toml
    [theme]
    primaryColor = "#2ecc71"
    backgroundColor = "#0a1f14"
    secondaryBackgroundColor = "#0f2a1c"
    textColor = "#e8f5e9"
    font = "sans serif"
    
    [server]
    enableCORS = false
    enableXsrfProtection = true
    ```
    """)
    
    st.markdown("---")
    
    # Dataset
    st.subheader("📊 Informasi Dataset")
    
    st.markdown(f"""
    ### `produksi_tanaman.csv`
    
    Dataset ini berisi informasi produksi **7 komoditas perkebunan utama** di **{len(df)} provinsi** Indonesia.
    
    **Kolom-kolom:**
    - `Provinsi`: Nama provinsi
    - `Kelapa_Sawit`: Produksi kelapa sawit (ribu ton)
    - `Kelapa`: Produksi kelapa (ribu ton)
    - `Karet`: Produksi karet (ribu ton)
    - `Kopi`: Produksi kopi (ribu ton)
    - `Kakao`: Produksi kakao (ribu ton)
    - `Teh`: Produksi teh (ribu ton)
    - `Tebu`: Produksi tebu (ribu ton)
    
    **Statistik Dataset:**
    - Total baris: **{len(df)}**
    - Total kolom: **{df.shape[1]}**
    - Ukuran file: **{df.memory_usage(deep=True).sum() / 1024:.2f} KB**
    
    **Sumber Data:** Dataset produksi tanaman perkebunan Indonesia (disesuaikan untuk keperluan edukasi)
    """)
    
    st.markdown("---")
    
    # Lisensi dan kontak
    st.subheader("📜 Lisensi & Kontak")
    
    st.markdown("""
    ### 📜 Lisensi
    
    Proyek ini dibuat untuk keperluan akademik (Tugas UAS Visualisasi Data).
    Dataset yang digunakan bersifat edukatif.
    
    ### 👨‍💻 Developer
    
    - **Nama:** [Nama Mahasiswa]
    - **NIM:** [NIM Mahasiswa]
    - **Program Studi:** Sains Data
    - **Mata Kuliah:** Visualisasi Data
    - **Dosen Pengampu:** [Nama Dosen]
    
    ### 🤝 Kontribusi
    
    Kontribusi, saran, dan feedback sangat dihargai! Silakan:
    - ⭐ Star repositori jika bermanfaat
    - 🐛 Laporkan bug melalui GitHub Issues
    - 💡 Usulkan fitur baru melalui Pull Request
    
    ### 🔗 Link Penting
    
    - 📖 [Dokumentasi Streamlit](https://docs.streamlit.io)
    - 📊 [Plotly Python Documentation](https://plotly.com/python/)
    - 🤖 [scikit-learn Documentation](https://scikit-learn.org)
    - ☁️ [Streamlit Cloud](https://streamlit.io/cloud)
    """)

# ============================================================================
# BAGIAN 18: FOOTER APLIKASI (DITAMPILKAN DI SEMUA HALAMAN)
# ============================================================================
# Footer konsisten di seluruh halaman

st.markdown("---")

# Footer dengan info aplikasi
st.markdown("""
<div class="app-footer">
    <p class="footer-title">🌿 Dasbor Produksi Perkebunan Indonesia 🌾</p>
    <p>Dibuat dengan ❤️ menggunakan Streamlit, Plotly, dan scikit-learn</p>
    <p style="margin-top: 10px; font-size: 0.9em;">
        Tugas UAS Visualisasi Data | Sains Data & Pertanian Indonesia | © 2026
    </p>
    <p style="margin-top: 5px; font-size: 0.8em; color: var(--text-muted);">
        Versi {version} | Last Updated: July 2026
    </p>
</div>
""".format(version=APP_VERSION), unsafe_allow_html=True)

# ============================================================================
# AKHIR DARI FILE app.py
# ============================================================================
# Total baris kode: ~3000+ baris (termasuk komentar dan docstring)
# ============================================================================
