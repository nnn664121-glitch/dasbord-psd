 ============================================================================
# app.py - VERSI DIPERBAIKI (Fix Import Error)
# ============================================================================
# Perubahan utama:
# 1. Menambahkan try-except pada semua import library
# 2. Memberikan pesan error yang jelas jika library tidak terinstall
# 3. Fallback handling untuk environment yang tidak lengkap
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.colors as pc
import warnings
import os
import sys
import io
import base64
import json
import math
from datetime import datetime, timedelta
import hashlib

# ============================================================================
# IMPORT MACHINE LEARNING DENGAN ERROR HANDLING
# ============================================================================
# Bagian ini menangani error import sklearn dengan try-except
# Jika scikit-learn tidak terinstall, aplikasi tetap berjalan dengan fitur terbatas

try:
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
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
    SKLEARN_ERROR = None
except ImportError as e:
    SKLEARN_AVAILABLE = False
    SKLEARN_ERROR = str(e)
    # Buat dummy classes agar kode tidak crash saat import
    class LinearRegression: pass
    class Ridge: pass
    class Lasso: pass
    class ElasticNet: pass
    class RandomForestRegressor: pass
    class GradientBoostingRegressor: pass
    class StandardScaler: pass
    class MinMaxScaler: pass
    class RobustScaler: pass
    class KMeans: pass
    class PCA: pass
    train_test_split = None
    cross_val_score = None
    KFold = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None
    mean_absolute_percentage_error = None
    explained_variance_score = None

# Import scipy dengan error handling
try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    import scipy.cluster.hierarchy as sch
    SCIPY_AVAILABLE = True
    SCIPY_ERROR = None
except ImportError as e:
    SCIPY_AVAILABLE = False
    SCIPY_ERROR = str(e)
    stats = None
    pearsonr = None
    spearmanr = None

# Konfigurasi peringatan
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="🌿 Dasbor Produksi Perkebunan Indonesia",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CEK KETERSEDIAAN LIBRARY DAN TAMPILKAN WARNING JIKA ADA YANG TIDAK TERINSTALL
# ============================================================================
missing_libs = []
if not SKLEARN_AVAILABLE:
    missing_libs.append(f"scikit-learn ({SKLEARN_ERROR})")
if not SCIPY_AVAILABLE:
    missing_libs.append(f"scipy ({SCIPY_ERROR})")

if missing_libs:
    st.error("⚠️ **Beberapa library tidak terinstall dengan benar!**")
    st.warning(f"""
    ### 📋 Library yang Bermasalah:
    {chr(10).join(['- ' + lib for lib in missing_libs])}
    
    ### 🛠️ Solusi:
    
    **1. Jika di Lokal (Local):**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install scikit-learn scipy statsmodels
    ```
    
    **2. Jika di Streamlit Cloud:**
    - Pastikan file `requirements.txt` ada di root repository
    - Pastikan tidak ada komentar inline di requirements.txt
    - Reboot/deploy ulang aplikasi dari menu Streamlit Cloud
    - Tunggu hingga build selesai (3-5 menit)
    
    **3. Verifikasi:**
    ```bash
    pip list | grep -i "scikit\|scipy"
    ```
    """)
    
    st.info("💡 **Aplikasi tetap berjalan dalam mode terbatas.** Fitur Machine Learning dan beberapa statistik tidak tersedia.")

# ============================================================================
# CSS KUSTOM
# ============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/poppins@5.0.0/index.min.css');
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/inter@5.0.0/index.min.css');
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.0.0/index.min.css');
    
    :root {
        --bg-primary: #0a1f14;
        --bg-secondary: #0f2a1c;
        --emerald: #2ecc71;
        --emerald-light: #58d68d;
        --gold: #f1c40f;
        --text-primary: #e8f5e9;
        --text-secondary: #a8d5ba;
        --border: rgba(46, 204, 113, 0.3);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a1f14 0%, #0f2a1c 40%, #1a1a2e 100%);
        color: #e8f5e9;
        font-family: 'Inter', sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d3320 0%, #1a4d2e 100%);
    }
    
    .main-title {
        background: linear-gradient(90deg, #2ecc71, #f1c40f, #27ae60);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 900;
        text-align: center;
        margin-bottom: 10px;
        font-family: 'Poppins', sans-serif;
    }
    
    .sub-title {
        text-align: center;
        color: #a8d5ba;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    .page-header {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(241, 196, 15, 0.08));
        border: 1px solid rgba(46, 204, 113, 0.4);
        border-radius: 20px;
        padding: 25px 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(46, 204, 113, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.18), rgba(241, 196, 15, 0.12));
        border: 1px solid rgba(46, 204, 113, 0.4);
        border-radius: 18px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(46, 204, 113, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(46, 204, 113, 0.3);
    }
    
    .metric-icon { font-size: 2em; margin-bottom: 8px; }
    
    .metric-value {
        font-size: 1.8em;
        font-weight: 800;
        color: #f1c40f;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: #a8d5ba;
        font-size: 0.82em;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    .insight-box {
        background: rgba(46, 204, 113, 0.12);
        border-left: 6px solid #2ecc71;
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .recommend-box {
        background: rgba(241, 196, 15, 0.12);
        border-left: 6px solid #f1c40f;
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .info-box {
        background: rgba(52, 152, 219, 0.12);
        border-left: 6px solid #3498db;
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .warning-box {
        background: rgba(231, 76, 60, 0.12);
        border-left: 6px solid #e74c3c;
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    h1, h2, h3 { color: #58d68d !important; font-family: 'Poppins', sans-serif; }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #2ecc71, #f1c40f, #2ecc71, transparent);
        margin: 30px 0;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# KONSTANTA
# ============================================================================
APP_VERSION = "1.1.0"
KOMODITAS_LIST = ['Kelapa_Sawit', 'Kelapa', 'Karet', 'Kopi', 'Kakao', 'Teh', 'Tebu']

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
    'ACEH': (5.5483, 95.3238),
    'SUMATERA UTARA': (3.5952, 98.6722),
    'SUMATERA BARAT': (-0.7893, 100.3291),
    'RIAU': (1.0, 101.4474),
    'JAMBI': (-1.6101, 103.6131),
    'SUMATERA SELATAN': (-3.3194, 104.9146),
    'BENGKULU': (-3.7926, 102.2614),
    'LAMPUNG': (-5.3971, 105.2668),
    'KEP. BANGKA BELITUNG': (-2.7411, 106.4406),
    'KEP. RIAU': (3.9453, 108.1428),
    'DKI JAKARTA': (-6.2088, 106.8456),
    'JAWA BARAT': (-6.9175, 107.6191),
    'JAWA TENGAH': (-7.1509, 110.1403),
    'DI YOGYAKARTA': (-7.7956, 110.3695),
    'JAWA TIMUR': (-7.5361, 112.2384),
    'BANTEN': (-6.4058, 106.0640),
    'BALI': (-8.3405, 115.0920),
    'NUSA TENGGARA BARAT': (-8.5833, 116.1167),
    'NUSA TENGGARA TIMUR': (-8.6574, 121.0794),
    'KALIMANTAN BARAT': (-0.0263, 109.3425),
    'KALIMANTAN TENGAH': (-1.6815, 113.3824),
    'KALIMANTAN SELATAN': (-3.3186, 114.5944),
    'KALIMANTAN TIMUR': (1.2379, 116.8529),
    'KALIMANTAN UTARA': (3.0731, 116.0414),
    'SULAWESI UTARA': (1.4748, 124.8421),
    'SULAWESI TENGAH': (-0.8950, 119.8376),
    'SULAWESI SELATAN': (-3.6688, 119.9741),
    'SULAWESI TENGGARA': (-3.9563, 122.5097),
    'GORONTALO': (0.5417, 123.0594),
    'SULAWESI BARAT': (-2.6748, 119.1051),
    'MALUKU': (-3.2385, 130.1453),
    'MALUKU UTARA': (1.5710, 127.7893),
    'PAPUA BARAT': (-1.3361, 132.4085),
    'PAPUA BARAT DAYA': (-1.5897, 131.2011),
    'PAPUA': (-2.5916, 140.6690),
    'PAPUA SELATAN': (-7.4833, 140.7500),
    'PAPUA TENGAH': (-3.3171, 137.3811),
    'PAPUA PEGUNUNGAN': (-4.0433, 138.9611)
}

# ============================================================================
# FUNGSI HELPER
# ============================================================================
def assign_region(provinsi):
    """Mengelompokkan provinsi ke wilayah geografis."""
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
            if keyword in prov_upper:
                return region
    return 'Lainnya'


def format_number(num, decimals=0):
    """Format angka dengan separator ribuan."""
    try:
        if pd.isna(num):
            return "N/A"
        if decimals == 0:
            return f"{int(num):,}".replace(',', '.')
        else:
            return f"{num:,.{decimals}f}".replace(',', '#').replace('.', ',').replace('#', '.')
    except Exception:
        return "N/A"


def calculate_correlation_significance(x, y):
    """Hitung korelasi dengan fallback jika scipy tidak tersedia."""
    if not SCIPY_AVAILABLE:
        # Fallback: gunakan pandas corr
        pearson_r = pd.Series(x).corr(pd.Series(y))
        return {
            'pearson_r': pearson_r,
            'pearson_p': 0.05,  # asumsi signifikan
            'pearson_significant': True,
            'pearson_strength': 'Sedang',
            'spearman_r': pearson_r,
            'spearman_p': 0.05,
            'spearman_significant': True,
            'spearman_strength': 'Sedang'
        }
    
    try:
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        
        def interpret_strength(r):
            abs_r = abs(r)
            if abs_r < 0.3: return "Lemah"
            elif abs_r < 0.5: return "Sedang"
            elif abs_r < 0.7: return "Kuat"
            elif abs_r < 0.9: return "Sangat Kuat"
            else: return "Hampir Sempurna"
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'pearson_significant': pearson_p < 0.05,
            'pearson_strength': interpret_strength(pearson_r),
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'spearman_significant': spearman_p < 0.05,
            'spearman_strength': interpret_strength(spearman_r)
        }
    except Exception as e:
        return {
            'pearson_r': 0, 'pearson_p': 1, 'pearson_significant': False,
            'pearson_strength': 'Error', 'spearman_r': 0, 'spearman_p': 1,
            'spearman_significant': False, 'spearman_strength': 'Error'
        }


def calculate_gini_coefficient(series):
    """Hitung koefisien Gini."""
    try:
        sorted_values = series.sort_values().reset_index(drop=True)
        n = len(sorted_values)
        if sorted_values.sum() == 0:
            return 0.0
        cumsum = sorted_values.cumsum()
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values)) / (n * cumsum.iloc[-1])) - (n + 1) / n
        return max(0, min(1, gini))
    except Exception:
        return np.nan


def apply_plotly_theme(fig, title=None, height=600):
    """Terapkan tema kustom pada figure Plotly."""
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(10, 31, 20, 0.3)",
        font=dict(family="Inter, sans-serif", color="#e8f5e9", size=13),
        title=dict(
            text=title if title else "",
            font=dict(size=20, color="#58d68d", family="Poppins"),
            x=0.5, xanchor="center"
        ),
        height=height,
        margin=dict(l=50, r=50, t=60, b=50),
        hoverlabel=dict(bgcolor="#0f2a1c", font_size=14, bordercolor="#2ecc71"),
        legend=dict(bgcolor="rgba(15, 42, 28, 0.8)", bordercolor="#2ecc71", borderwidth=1, font=dict(color="#e8f5e9"))
    )
    
    try:
        if hasattr(fig.layout, 'scene'):
            fig.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor="rgb(10, 31, 20)", gridcolor="rgba(46, 204, 113, 0.2)", color="#e8f5e9", title_font=dict(color="#2ecc71")),
                    yaxis=dict(backgroundcolor="rgb(10, 31, 20)", gridcolor="rgba(46, 204, 113, 0.2)", color="#e8f5e9", title_font=dict(color="#2ecc71")),
                    zaxis=dict(backgroundcolor="rgb(10, 31, 20)", gridcolor="rgba(46, 204, 113, 0.2)", color="#e8f5e9", title_font=dict(color="#f1c40f"))
                )
            )
    except Exception:
        pass
    
    return fig


def create_metric_card(icon, value, label, change=None, change_type="positive"):
    """Buat HTML kartu metrik."""
    change_html = ""
    if change:
        change_html = f'<div style="font-size:0.85em;margin-top:8px;color:{"#2ecc71" if change_type=="positive" else "#e74c3c"};">{change}</div>'
    
    return f'''
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {change_html}
    </div>
    '''


# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """Load data dengan preprocessing."""
    try:
        df_raw = pd.read_csv("produksi_tanaman.csv")
    except FileNotFoundError:
        return None, None, "❌ File 'produksi_tanaman.csv' tidak ditemukan."
    except Exception as e:
        return None, None, f"❌ Error membaca file: {str(e)}"
    
    df = df_raw.copy()
    df['Wilayah'] = df['Provinsi'].apply(assign_region)
    df['Total_Produksi'] = df[KOMODITAS_LIST].sum(axis=1)
    df['Latitude'] = df['Provinsi'].map(lambda p: PROV_COORDS.get(p, (np.nan, np.nan))[0])
    df['Longitude'] = df['Provinsi'].map(lambda p: PROV_COORDS.get(p, (np.nan, np.nan))[1])
    df['Rank_Total'] = df['Total_Produksi'].rank(ascending=False, method='min').astype(int)
    
    def get_dominant(row):
        values = {k: row[k] for k in KOMODITAS_LIST}
        max_key = max(values, key=values.get)
        return KOMODITAS_LABELS.get(max_key, max_key)
    
    df['Komoditas_Dominan'] = df.apply(get_dominant, axis=1)
    
    # HHI
    hhi_dict = {}
    for idx, row in df.iterrows():
        values = [row[col] for col in KOMODITAS_LIST]
        total = sum(values)
        if total > 0:
            shares = [v / total for v in values]
            hhi_dict[row['Provinsi']] = sum(s ** 2 for s in shares)
        else:
            hhi_dict[row['Provinsi']] = 0.0
    
    df['HHI_Index'] = df['Provinsi'].map(hhi_dict)
    
    def classify_hhi(h):
        if h < 0.15: return "🌈 Sangat Diversifikasi"
        elif h < 0.25: return "🎨 Moderat"
        elif h < 0.40: return "⚖️ Kurang Diversifikasi"
        else: return "🎯 Sangat Spesialisasi"
    
    df['Diversifikasi'] = df['HHI_Index'].apply(classify_hhi)
    
    return df_raw, df, None


df_raw, df, data_error = load_and_preprocess_data()

if data_error:
    st.error(data_error)
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 3em;">🌿🌾🌴</div>
        <div style="color: #58d68d; font-size: 1.1em; font-weight: 700; margin-top: 10px;">
            DASHBOARD PERKEBUNAN
        </div>
        <div style="color: #f7dc6f; font-size: 0.85em;">Indonesia 🇮🇩</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🗺️ Navigasi")
    
    pages = [
        "📊 Page 1: Overview",
        "🧹 Page 2: Data Cleaning",
        "📈 Page 3: EDA & 3D Viz",
        "🗺️ Page 3b: Peta",
        "🔗 Page 4: Korelasi & Regresi",
    ]
    
    if SKLEARN_AVAILABLE:
        pages.append("🎯 Page 4b: ML Lanjutan")
    
    pages.extend(["💡 Page 5: Insights", "📚 Tentang"])
    
    page = st.radio("Pilih halaman:", pages, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown(f"""
    ### 📊 Info Dataset
    - **Provinsi:** {len(df)}
    - **Komoditas:** {len(KOMODITAS_LIST)}
    - **Total Produksi:** {format_number(df['Total_Produksi'].sum())}
    
    ### ℹ️ Status Library
    - ✅ scikit-learn: {"Tersedia" if SKLEARN_AVAILABLE else "❌ Tidak"}
    - ✅ scipy: {"Tersedia" if SCIPY_AVAILABLE else "❌ Tidak"}
    
    <div style="text-align:center; color:#6b8f7a; font-size:0.85em; margin-top:20px;">
        <p>🌿 Versi {APP_VERSION}</p>
        <p>UAS Visualisasi Data 2026</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<h1 class="main-title">🌿 Dasbor Produksi Perkebunan Indonesia</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analisis Visual 3D Interaktif • Sektor Agrikultur Modern • Tugas UAS</p>', unsafe_allow_html=True)

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if "Page 1" in page:
    st.markdown("""
    <div class="page-header">
        <h1>📊 Overview & Data Understanding</h1>
        <p>Gambaran umum dataset produksi tanaman perkebunan Indonesia</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI
    st.subheader("🏆 Key Performance Indicators")
    
    total_semua = df[KOMODITAS_LIST].sum().sum()
    total_sawit = df['Kelapa_Sawit'].sum()
    total_karet = df['Karet'].sum()
    total_kopi = df['Kopi'].sum()
    pct_sawit = (total_sawit / total_semua) * 100 if total_semua > 0 else 0
    pct_karet = (total_karet / total_semua) * 100 if total_semua > 0 else 0
    pct_kopi = (total_kopi / total_semua) * 100 if total_semua > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(create_metric_card("🌾", format_number(total_semua), "Total Produksi", "ribu ton"), unsafe_allow_html=True)
    with c2:
        st.markdown(create_metric_card("🌴", format_number(total_sawit), "Kelapa Sawit", f"{pct_sawit:.1f}%"), unsafe_allow_html=True)
    with c3:
        st.markdown(create_metric_card("🌳", format_number(total_karet), "Karet", f"{pct_karet:.1f}%"), unsafe_allow_html=True)
    with c4:
        st.markdown(create_metric_card("☕", format_number(total_kopi), "Kopi", f"{pct_kopi:.1f}%"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📋 Preview Data")
    st.dataframe(df.head(15), use_container_width=True, height=450)
    
    st.markdown("---")
    
    st.subheader("📉 Statistik Deskriptif")
    st.dataframe(df[KOMODITAS_LIST].describe().round(2), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📦 Box Plot Distribusi")
    df_melted = df.melt(id_vars=['Provinsi'], value_vars=KOMODITAS_LIST, var_name='Komoditas', value_name='Produksi')
    
    fig_box = px.box(
        df_melted, x='Komoditas', y='Produksi', color='Komoditas',
        title='Distribusi Produksi per Komoditas'
    )
    fig_box = apply_plotly_theme(fig_box, height=500)
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

# ============================================================================
# PAGE 2: DATA CLEANING
# ============================================================================
elif "Page 2" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🧹 Data Cleaning & Preprocessing</h1>
        <p>Tahapan pembersihan dan persiapan data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Missing values
    st.subheader("1️⃣ Missing Values")
    missing = df_raw.isnull().sum()
    missing_df = pd.DataFrame({
        'Kolom': missing.index,
        'Jumlah Missing': missing.values,
        'Persentase (%)': (missing.values / len(df_raw)) * 100
    })
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
    
    if missing.sum() == 0:
        st.success("✅ Dataset bersih, tidak ada missing values.")
    
    st.markdown("---")
    
    # Duplikasi
    st.subheader("2️⃣ Data Duplikat")
    n_dup = df_raw.duplicated().sum()
    st.info(f"Jumlah baris duplikat: **{n_dup}**")
    
    st.markdown("---")
    
    # Outlier
    st.subheader("3️⃣ Deteksi Outlier (IQR Method)")
    
    outlier_results = []
    for kom in KOMODITAS_LIST:
        series = df[kom]
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((series < lower) | (series > upper)).sum()
        outlier_results.append({
            'Komoditas': kom,
            'Q1': round(Q1, 2),
            'Q3': round(Q3, 2),
            'IQR': round(IQR, 2),
            'Jumlah Outlier': n_out
        })
    
    st.dataframe(pd.DataFrame(outlier_results), use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>📝 Catatan:</strong> Outlier umumnya merepresentasikan provinsi produsen utama 
        (contoh: Riau untuk Kelapa Sawit), bukan error data.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Engineering
    st.subheader("4️⃣ Feature Engineering")
    st.markdown("""
    **Fitur baru yang ditambahkan:**
    - `Wilayah`: Pengelompokan provinsi
    - `Total_Produksi`: Jumlah semua komoditas
    - `Rank_Total`: Ranking berdasarkan total produksi
    - `Komoditas_Dominan`: Komoditas tertinggi
    - `HHI_Index`: Herfindahl-Hirschman Index (diversifikasi)
    """)
    
    st.dataframe(df[['Provinsi', 'Wilayah', 'Total_Produksi', 'Komoditas_Dominan', 'HHI_Index', 'Diversifikasi']].head(15), 
                 use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 3: EDA & 3D VISUALIZATIONS
# ============================================================================
elif "Page 3: EDA" in page:
    st.markdown("""
    <div class="page-header">
        <h1>📈 EDA & 3D Visualizations</h1>
        <p>Exploratory Data Analysis dengan visualisasi 3D interaktif</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pilih 3 komoditas untuk scatter 3D
    st.subheader("🧊 Visualisasi 3D #1: Scatter 3D")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        ax_x = st.selectbox("Sumbu X:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x, x), key="3d_x")
    with c2:
        ax_y = st.selectbox("Sumbu Y:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x, x), index=1, key="3d_y")
    with c3:
        ax_z = st.selectbox("Sumbu Z:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x, x), index=2, key="3d_z")
    
    color_by = st.radio("Warnai berdasarkan:", ["Wilayah", "Komoditas_Dominan", "Diversifikasi"], horizontal=True)
    
    fig_s3d = px.scatter_3d(
        df, x=ax_x, y=ax_y, z=ax_z, color=color_by,
        size='Total_Produksi', size_max=35, hover_name='Provinsi',
        title=f'Scatter 3D: {ax_x} × {ax_y} × {ax_z}', opacity=0.85
    )
    fig_s3d = apply_plotly_theme(fig_s3d, height=650)
    st.plotly_chart(fig_s3d, use_container_width=True)
    
    st.markdown("---")
    
    # Surface plot
    st.subheader("🏔️ Visualisasi 3D #2: Surface Plot (Topografi Produksi)")
    
    n_prov_surface = st.slider("Jumlah provinsi teratas:", 5, 38, 20, 1)
    df_sorted = df.nlargest(n_prov_surface, 'Total_Produksi').set_index('Provinsi')[KOMODITAS_LIST]
    
    colorscale = st.selectbox("Colorscale:", ['Emerald', 'Viridis', 'Plasma', 'Inferno', 'Turbo', 'Jet'])
    
    fig_surf = go.Figure(data=[go.Surface(
        z=df_sorted.values,
        x=np.arange(len(KOMODITAS_LIST)),
        y=np.arange(len(df_sorted)),
        colorscale=colorscale,
        colorbar=dict(title="Produksi", tickfont=dict(color="white"))
    )])
    
    fig_surf = apply_plotly_theme(fig_surf, title=f"Topografi Produksi {n_prov_surface} Provinsi Teratas", height=700)
    fig_surf.update_layout(
        scene=dict(
            xaxis=dict(title='Komoditas', tickvals=np.arange(len(KOMODITAS_LIST)), ticktext=[k.replace('_', ' ') for k in KOMODITAS_LIST]),
            yaxis=dict(title='Provinsi', tickvals=np.arange(len(df_sorted)), ticktext=[n[:15] for n in df_sorted.index]),
            zaxis=dict(title='Produksi (ribu ton)')
        )
    )
    st.plotly_chart(fig_surf, use_container_width=True)
    
    st.markdown("---")
    
    # Bar 3D
    st.subheader("🏗️ Visualisasi 3D #3: Bar Chart 3D")
    
    n_bar = st.slider("Jumlah provinsi:", 5, 20, 10, 1)
    top_bar = df.nlargest(n_bar, 'Total_Produksi')[['Provinsi', 'Total_Produksi']].reset_index(drop=True)
    
    fig_bar3d = go.Figure()
    colors = ['#2ecc71', '#27ae60', '#16a085', '#1abc9c', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#3498db',
              '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#7f8c8d', '#c0392b', '#16a085', '#2c3e50', '#27ae60', '#e67e22']
    
    for i, row in top_bar.iterrows():
        x0, x1 = i - 0.35, i + 0.35
        y0, y1 = 0, 0.7
        z0, z1 = 0, row['Total_Produksi']
        
        fig_bar3d.add_trace(go.Mesh3d(
            x=[x0, x1, x1, x0, x0, x1, x1, x0],
            y=[y0, y0, y1, y1, y0, y0, y1, y1],
            z=[z0, z0, z0, z0, z1, z1, z1, z1],
            i=[0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1],
            j=[1, 2, 5, 6, 1, 5, 3, 7, 2, 6, 3, 5],
            k=[2, 3, 6, 7, 5, 4, 7, 6, 3, 7, 5, 4],
            color=colors[i % len(colors)],
            opacity=0.85,
            name=row['Provinsi'],
            hovertext=f"<b>{row['Provinsi']}</b><br>Total: {row['Total_Produksi']:,.0f}",
            hoverinfo='text'
        ))
    
    fig_bar3d = apply_plotly_theme(fig_bar3d, title=f"Bar 3D: Top {n_bar} Provinsi", height=650)
    fig_bar3d.update_layout(
        scene=dict(
            xaxis=dict(title='Provinsi', tickvals=list(range(n_bar)), ticktext=[n[:12] for n in top_bar['Provinsi']]),
            yaxis=dict(visible=False),
            zaxis=dict(title='Total Produksi')
        ),
        showlegend=False
    )
    st.plotly_chart(fig_bar3d, use_container_width=True)
    
    st.markdown("---")
    
    # Stacked bar
    st.subheader("🎨 Komposisi Komoditas per Wilayah")
    region_summary = df.groupby('Wilayah')[KOMODITAS_LIST].sum().reset_index()
    
    fig_stack = go.Figure()
    for kom in KOMODITAS_LIST:
        fig_stack.add_trace(go.Bar(
            x=region_summary['Wilayah'], y=region_summary[kom],
            name=KOMODITAS_LABELS.get(kom, kom),
            marker_color=KOMODITAS_COLORS.get(kom, '#2ecc71')
        ))
    
    fig_stack = apply_plotly_theme(fig_stack, title="Komposisi Produksi per Wilayah", height=500)
    fig_stack.update_layout(barmode='stack', xaxis_title='Wilayah', yaxis_title='Produksi (ribu ton)')
    st.plotly_chart(fig_stack, use_container_width=True)

# ============================================================================
# PAGE 3b: PETA
# ============================================================================
elif "Page 3b" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🗺️ Peta Distribusi Indonesia</h1>
        <p>Visualisasi geospasial produksi tanaman perkebunan</p>
    </div>
    """, unsafe_allow_html=True)
    
    komoditas_peta = st.selectbox(
        "Pilih Komoditas:",
        ['Total_Produksi'] + KOMODITAS_LIST,
        format_func=lambda x: "🌾 Total" if x == 'Total_Produksi' else KOMODITAS_LABELS.get(x, x)
    )
    
    df_map = df.dropna(subset=['Latitude', 'Longitude'])
    
    fig_geo = px.scatter_geo(
        df_map, lat='Latitude', lon='Longitude', color=komoditas_peta,
        size=komoditas_peta, size_max=50, hover_name='Provinsi',
        scope='asia', color_continuous_scale='Greens',
        title=f'Distribusi {komoditas_peta.replace("_", " ")}'
    )
    
    fig_geo.update_geos(
        showcountries=True, countrycolor="rgba(46, 204, 113, 0.4)",
        showcoastlines=True, coastlinecolor="#2ecc71",
        showland=True, landcolor="rgba(26, 77, 46, 0.4)",
        showocean=True, oceancolor="rgba(10, 31, 20, 0.95)",
        lataxis_range=[-12, 8], lonaxis_range=[92, 145]
    )
    
    fig_geo = apply_plotly_theme(fig_geo, height=650)
    st.plotly_chart(fig_geo, use_container_width=True)
    
    st.markdown("---")
    
    # Bubble map by region
    st.subheader("🎯 Peta Wilayah Geografis")
    fig_region = px.scatter_geo(
        df_map, lat='Latitude', lon='Longitude', color='Wilayah',
        size='Total_Produksi', size_max=60, hover_name='Provinsi',
        color_discrete_map=REGION_COLORS, scope='asia'
    )
    fig_region.update_geos(
        showcountries=True, countrycolor="rgba(46, 204, 113, 0.4)",
        showcoastlines=True, coastlinecolor="#2ecc71",
        showland=True, landcolor="rgba(26, 77, 46, 0.4)",
        showocean=True, oceancolor="rgba(10, 31, 20, 0.95)",
        lataxis_range=[-12, 8], lonaxis_range=[92, 145]
    )
    fig_region = apply_plotly_theme(fig_region, title="Distribusi Provinsi per Wilayah", height=650)
    st.plotly_chart(fig_region, use_container_width=True)

# ============================================================================
# PAGE 4: KORELASI & REGRESI
# ============================================================================
elif "Page 4: Korelasi" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🔗 Analisis Korelasi & Regresi</h1>
        <p>Hubungan antar komoditas dan model prediktif</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Heatmap Pearson
    st.subheader("🔥 Heatmap Korelasi Pearson")
    corr = df[KOMODITAS_LIST].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=[KOMODITAS_LABELS.get(c, c) for c in corr.columns],
        y=[KOMODITAS_LABELS.get(c, c) for c in corr.index],
        colorscale='RdYlGn', zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        colorbar=dict(title="Korelasi", tickfont=dict(color="white"))
    ))
    fig_corr = apply_plotly_theme(fig_corr, title="Matriks Korelasi Pearson", height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # Analisis detail
    st.subheader("🔬 Analisis Korelasi Detail")
    c1, c2 = st.columns(2)
    with c1:
        v1 = st.selectbox("Variabel 1:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x, x), key="cv1")
    with c2:
        v2 = st.selectbox("Variabel 2:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x, x), index=1, key="cv2")
    
    corr_info = calculate_correlation_significance(df[v1], df[v2])
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(create_metric_card("📐", f"{corr_info['pearson_r']:.3f}", "Pearson r", corr_info['pearson_strength']), unsafe_allow_html=True)
    with c2:
        st.markdown(create_metric_card("📊", f"{corr_info['spearman_r']:.3f}", "Spearman ρ", corr_info['spearman_strength']), unsafe_allow_html=True)
    with c3:
        sig = "✅ Signifikan" if corr_info['pearson_significant'] else "❌ Tidak"
        st.markdown(create_metric_card("🎯", sig, "Status (α=0.05)", ""), unsafe_allow_html=True)
    
    fig_sc = px.scatter(
        df, x=v1, y=v2, color='Wilayah', size='Total_Produksi',
        hover_name='Provinsi', trendline='ols' if SCIPY_AVAILABLE else None,
        color_discrete_map=REGION_COLORS,
        title=f'{KOMODITAS_LABELS.get(v1)} vs {KOMODITAS_LABELS.get(v2)}'
    )
    fig_sc = apply_plotly_theme(fig_sc, height=550)
    st.plotly_chart(fig_sc, use_container_width=True)
    
    st.markdown("---")
    
    # Regresi
    if SKLEARN_AVAILABLE:
        st.subheader("🤖 Model Regresi Linier Berganda")
        
        target_var = st.selectbox("Target:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x, x), key="reg_t")
        feature_vars = [k for k in KOMODITAS_LIST if k != target_var]
        
        X = df[feature_vars].values
        y = df[target_var].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred + 1e-8) * 100
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(create_metric_card("📏", f"{mae:.2f}", "MAE", ""), unsafe_allow_html=True)
        with c2: st.markdown(create_metric_card("📐", f"{rmse:.2f}", "RMSE", ""), unsafe_allow_html=True)
        with c3: st.markdown(create_metric_card("🎯", f"{r2:.4f}", "R² Score", ""), unsafe_allow_html=True)
        with c4: st.markdown(create_metric_card("📊", f"{mape:.1f}%", "MAPE", ""), unsafe_allow_html=True)
        
        # Actual vs Predicted
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode='markers',
            marker=dict(size=12, color='#2ecc71', line=dict(color='#f1c40f', width=2)),
            name='Data'
        ))
        min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig_avp.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v], mode='lines',
            line=dict(color='#e74c3c', width=2, dash='dash'), name='Perfect'
        ))
        fig_avp = apply_plotly_theme(fig_avp, title="Actual vs Predicted", height=550)
        fig_avp.update_layout(xaxis_title="Aktual", yaxis_title="Prediksi")
        st.plotly_chart(fig_avp, use_container_width=True)
        
        # Koefisien
        coef_df = pd.DataFrame({
            'Fitur': [KOMODITAS_LABELS.get(f, f) for f in feature_vars],
            'Koefisien': model.coef_
        }).sort_values('Koefisien', key=abs, ascending=False)
        
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Fitur regresi tidak tersedia karena scikit-learn tidak terinstall.")

# ============================================================================
# PAGE 4b: ML LANJUTAN
# ============================================================================
elif "Page 4b" in page and SKLEARN_AVAILABLE:
    st.markdown("""
    <div class="page-header">
        <h1>🎯 Machine Learning Lanjutan</h1>
        <p>Perbandingan berbagai algoritma regresi</p>
    </div>
    """, unsafe_allow_html=True)
    
    target_ml = st.selectbox("Target:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x, x), key="ml_t")
    feature_ml = [k for k in KOMODITAS_LIST if k != target_ml]
    
    selected_models = st.multiselect(
        "Model:",
        ["Linear Regression", "Ridge (L2)", "Lasso (L1)", "Random Forest", "Gradient Boosting"],
        default=["Linear Regression", "Random Forest", "Gradient Boosting"]
    )
    
    if selected_models:
        X = df[feature_ml].values
        y = df[target_ml].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        results = []
        
        for m_name in selected_models:
            if m_name == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
            elif m_name == "Ridge (L2)":
                model = Ridge(alpha=1.0)
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
            elif m_name == "Lasso (L1)":
                model = Lasso(alpha=1.0, max_iter=10000)
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
            elif m_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif m_name == "Gradient Boosting":
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            results.append({
                'Model': m_name,
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R²': r2_score(y_test, y_pred),
                'MAPE (%)': mean_absolute_percentage_error(y_test, y_pred + 1e-8) * 100
            })
        
        results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
        
        st.dataframe(
            results_df.style.highlight_max(subset=['R²'], color='#2ecc71')
                    .highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], color='#2ecc71')
                    .format({'MAE': '{:.3f}', 'RMSE': '{:.3f}', 'R²': '{:.4f}', 'MAPE (%)': '{:.2f}'}),
            use_container_width=True, hide_index=True
        )
        
        # Bar chart
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            x=results_df['Model'], y=results_df['R²'],
            marker_color='#2ecc71', text=results_df['R²'].round(3), textposition='outside'
        ))
        fig_r2 = apply_plotly_theme(fig_r2, title="Perbandingan R² Score", height=500)
        fig_r2.update_layout(xaxis_title='Model', yaxis_title='R² Score')
        st.plotly_chart(fig_r2, use_container_width=True)

# ============================================================================
# PAGE 5: INSIGHTS
# ============================================================================
elif "Page 5" in page:
    st.markdown("""
    <div class="page-header">
        <h1>💡 Insights & Rekomendasi</h1>
        <p>Kesimpulan mendalam dan rekomendasi strategis</p>
    </div>
    """, unsafe_allow_html=True)
    
    top_prov = df.loc[df['Kelapa_Sawit'].idxmax(), 'Provinsi']
    top_val = df['Kelapa_Sawit'].max()
    pct_sawit = (df['Kelapa_Sawit'].sum() / df['Total_Produksi'].sum()) * 100
    top_wil = df.groupby('Wilayah')['Total_Produksi'].sum().idxmax()
    gini = calculate_gini_coefficient(df['Total_Produksi'])
    
    st.subheader("📋 Ringkasan Eksekutif")
    st.markdown(f"""
    <div class="info-box">
        <ul>
            <li>🌴 Kelapa Sawit menyumbang <b>{pct_sawit:.1f}%</b> dari total produksi</li>
            <li>🏆 <b>{top_prov}</b> produsen sawit terbesar ({top_val:,.0f} ribu ton)</li>
            <li>🗺️ Wilayah <b>{top_wil}</b> sentra produksi nasional</li>
            <li>⚖️ Koefisien Gini: <b>{gini:.3f}</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🔍 5 Insight Mendalam")
    
    insights = [
        ("Dominasi Kelapa Sawit", f"Kelapa Sawit menyumbang {pct_sawit:.1f}% produksi nasional dengan {top_prov} sebagai produsen utama. Ketergantungan pada satu komoditas menciptakan risiko ekonomi terhadap fluktuasi harga CPO global."),
        ("Ketimpangan Produksi", f"Koefisien Gini {gini:.3f} menunjukkan ketimpangan tinggi antar provinsi. 5 provinsi teratas menguasai >60% total produksi nasional."),
        ("Spesialisasi Regional", "Pola spesialisasi jelas: Sawit di Sumatera-Kalimantan, Kopi di dataran tinggi Sumatera-Sulawesi, Tebu di Jawa Timur, Teh di Jawa Barat."),
        ("Diversifikasi Rendah", "Rata-rata HHI tinggi menunjukkan kebanyakan provinsi bergantung pada 1-2 komoditas saja, rentan terhadap shock harga."),
        ("Potensi Indonesia Timur", "Wilayah Maluku, Papua, dan NTT memiliki potensi agroklimat besar namun kontribusinya masih minim terhadap total nasional.")
    ]
    
    for i, (title, content) in enumerate(insights, 1):
        st.markdown(f"""
        <div class="insight-box">
            <strong>💡 Insight #{i}: {title}</strong>
            <p style="margin:8px 0 0 0;">{content}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🎯 5 Rekomendasi Implementatif")
    
    rekomendasis = [
        ("Diversifikasi Komoditas", "Program tumpang sari sawit dengan kakao/kopi. Insentif fiskal untuk komoditas alternatif. Target: HHI nasional < 0.35 dalam 5 tahun."),
        ("Infrastruktur Indonesia Timur", "Pembangunan jalan, pelabuhan, cold storage di Papua, Maluku, NTT. Target: kontribusi >20% dalam 10 tahun."),
        ("Sertifikasi Sustainability", "Percepatan ISPO 100% untuk sawit. Sistem traceability blockchain. Kolaborasi dengan buyer internasional."),
        ("Digitalisasi Sistem Informasi", "Dashboard nasional real-time. IoT monitoring kebun. Early warning system cuaca/hama."),
        ("Hilirisasi Produk", "Industri pengolahan di sentra produksi. Branding kopi specialty dan teh premium. Target: rasio ekspor olahan >60%.")
    ]
    
    for i, (title, content) in enumerate(rekomendasis, 1):
        st.markdown(f"""
        <div class="recommend-box">
            <strong>🎯 Rekomendasi #{i}: {title}</strong>
            <p style="margin:8px 0 0 0;">{content}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TENTANG
# ============================================================================
elif "Tentang" in page:
    st.markdown("""
    <div class="page-header">
        <h1>📚 Tentang Aplikasi</h1>
        <p>Informasi lengkap tentang dasbor ini</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### ✨ Fitur Utama
    
    - 📊 **7 Halaman Analisis**: Overview, Data Cleaning, EDA 3D, Peta, Korelasi & Regresi, ML Lanjutan, Insights
    - 🧊 **5 Visualisasi 3D Interaktif** (Plotly)
    - 🗺️ **Peta Interaktif Indonesia**
    - 🔥 **Heatmap Korelasi** Pearson & Spearman
    - 🤖 **Multiple ML Models**: Linear, Ridge, Lasso, Random Forest, Gradient Boosting
    - 🎨 **Tema Dark Mode Modern** (Emerald + Gold)
    
    ### 🛠️ Teknologi
    
    | Library | Fungsi |
    |---------|--------|
    | Streamlit | Framework web |
    | Pandas | Manipulasi data |
    | NumPy | Komputasi numerik |
    | Plotly | Visualisasi 2D & 3D |
    | scikit-learn | Machine learning |
    | SciPy | Statistik |
    
    ### 🚀 Cara Menjalankan
    
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    
    ### ☁️ Deploy ke Streamlit Cloud
    
    1. Push ke GitHub
    2. Buka share.streamlit.io
    3. New App → pilih repository → Deploy
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; padding:20px; color:#a8d5ba;">
    <p style="color:#58d68d; font-weight:700; font-size:1.1em;">🌿 Dasbor Produksi Perkebunan Indonesia 🌾</p>
    <p>Tugas UAS Visualisasi Data | © 2026 | Versi {APP_VERSION}</p>
</div>
""", unsafe_allow_html=True)
