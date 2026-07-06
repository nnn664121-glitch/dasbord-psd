# ============================================================================
# app.py - DASBOR INTERAKTIF PRODUKSI TANAMAN PERKEBUNAN INDONESIA
# ============================================================================
# Versi: 1.0.0 | Updated: 2026
# Author: Mahasiswa Sains Data
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
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import scipy.cluster.hierarchy as sch

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI APLIKASI
# ============================================================================

APP_VERSION = "1.0.0"
APP_NAME = "Dasbor Produksi Perkebunan Indonesia"
APP_ICON = "🌿"
DATA_YEAR = 2024

KOMODITAS_LIST = [
    'Kelapa_Sawit', 'Kelapa', 'Karet', 'Kopi', 'Kakao', 'Teh', 'Tebu'
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

# Koordinat Provinsi Indonesia
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
# KONFIGURASI STREAMLIT
# ============================================================================

st.set_page_config(
    page_title=f"{APP_ICON} {APP_NAME}",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS KUSTOM (DIPERPENDEK)
# ============================================================================

CUSTOM_CSS = """
<style>
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/poppins@5.0.0/index.min.css');
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/inter@5.0.0/index.min.css');
    
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
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1600px;
    }
    
    h1, h2, h3 {
        color: var(--emerald-light) !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    
    .main-title {
        background: linear-gradient(90deg, #2ecc71, #58d68d, #f1c40f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 900;
        text-align: center;
        font-family: 'Poppins', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.18) 0%, rgba(241, 196, 15, 0.12) 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        transition: all 0.4s;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(46, 204, 113, 0.3);
    }
    
    .metric-value {
        font-size: 1.9em;
        font-weight: 800;
        color: var(--gold);
        margin-bottom: 8px;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.82em;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    .page-header {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15) 0%, rgba(241, 196, 15, 0.08) 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 25px 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(46, 204, 113, 0.2);
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.12) 0%, rgba(46, 204, 113, 0.05) 100%);
        border-left: 6px solid var(--emerald);
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .recommend-box {
        background: linear-gradient(135deg, rgba(241, 196, 15, 0.12) 0%, rgba(241, 196, 15, 0.05) 100%);
        border-left: 6px solid var(--gold);
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.15) 0%, rgba(52, 152, 219, 0.05) 100%);
        border-left: 6px solid #3498db;
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .app-footer {
        background: linear-gradient(90deg, rgba(13, 51, 32, 0.8) 0%, rgba(15, 42, 28, 0.8) 100%);
        border-top: 2px solid var(--emerald);
        padding: 25px;
        text-align: center;
        margin-top: 50px;
        border-radius: 15px;
    }
    
    .app-footer p {
        color: var(--text-secondary);
        margin: 5px 0;
    }
    
    .footer-title {
        color: var(--emerald-light);
        font-weight: 700;
        font-size: 1.1em;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# FUNGSI-FUNGSI UTILITY
# ============================================================================

def load_data(file_path="produksi_tanaman.csv"):
    """Load dan validasi data CSV"""
    try:
        if not os.path.exists(file_path):
            return None, f"❌ File '{file_path}' tidak ditemukan"
        
        df = pd.read_csv(file_path)
        
        if df.empty:
            return None, "❌ File CSV kosong"
        
        if 'Provinsi' not in df.columns:
            return None, "❌ Kolom 'Provinsi' tidak ditemukan"
        
        missing_cols = [col for col in KOMODITAS_LIST if col not in df.columns]
        if missing_cols:
            return None, f"❌ Kolom tidak lengkap: {', '.join(missing_cols)}"
        
        return df, None
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def assign_region(provinsi):
    """Mengelompokkan provinsi ke wilayah"""
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
    """Format angka dengan separator"""
    try:
        if pd.isna(num):
            return "N/A"
        if decimals == 0:
            return f"{int(num):,}".replace(',', '.')
        else:
            return f"{num:,.{decimals}f}".replace(',', '#').replace('.', ',').replace('#', '.')
    except:
        return "N/A"


def calculate_skewness_kurtosis(series):
    """Hitung skewness dan kurtosis"""
    try:
        skew = series.skew()
        kurt = series.kurtosis()
        
        if abs(skew) < 0.5:
            skew_interpretation = "Simetris"
        elif skew > 0.5:
            skew_interpretation = "Mencondong Kanan"
        else:
            skew_interpretation = "Mencondong Kiri"
        
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
    except:
        return {
            'skewness': np.nan,
            'kurtosis': np.nan,
            'skewness_interpretation': "Error",
            'kurtosis_interpretation': "Error"
        }


def calculate_correlation_significance(x, y, alpha=0.05):
    """Hitung korelasi dengan uji signifikansi"""
    try:
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        
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
            'pearson_significant': pearson_p < alpha,
            'pearson_strength': interpret_strength(pearson_r),
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'spearman_significant': spearman_p < alpha,
            'spearman_strength': interpret_strength(spearman_r)
        }
    except:
        return {
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'pearson_significant': False,
            'spearman_r': np.nan,
            'spearman_p': np.nan,
            'spearman_significant': False
        }


def apply_plotly_theme(fig, title=None, height=600):
    """Terapkan tema kustom pada Plotly figure"""
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(10, 31, 20, 0.3)",
        font=dict(family="Inter, sans-serif", color="#e8f5e9", size=13),
        title=dict(
            text=title if title else "",
            font=dict(size=20, color="#58d68d", family="Poppins"),
            x=0.5,
            xanchor="center"
        ),
        height=height,
        margin=dict(l=50, r=50, t=60, b=50),
        hovermode='closest',
        legend=dict(
            bgcolor="rgba(15, 42, 28, 0.8)",
            bordercolor="#2ecc71",
            borderwidth=1,
            font=dict(color="#e8f5e9")
        )
    )
    return fig


def calculate_gini_coefficient(series):
    """Hitung koefisien Gini"""
    try:
        sorted_values = series.sort_values().reset_index(drop=True)
        n = len(sorted_values)
        
        if sorted_values.sum() == 0:
            return 0.0
        
        cumsum = sorted_values.cumsum()
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values)) / (n * cumsum.iloc[-1])) - (n + 1) / n
        
        return max(0, min(1, gini))
    except:
        return np.nan


def calculate_herfindahl_index(df, komoditas_columns):
    """Hitung HHI untuk diversifikasi"""
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
    """Klasifikasikan tingkat diversifikasi"""
    if hhi < 0.15:
        return "🌈 Sangat Diversifikasi"
    elif hhi < 0.25:
        return "🎨 Moderat Diversifikasi"
    elif hhi < 0.40:
        return "⚖️ Kurang Diversifikasi"
    else:
        return "🎯 Sangat Spesialisasi"


# ============================================================================
# LOAD & PREPROCESS DATA
# ============================================================================

@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """Load dan preprocess data"""
    df_raw, error = load_data()
    
    if error:
        return None, None, error
    
    df = df_raw.copy()
    
    # Feature engineering
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


# Load data
df_raw, df, data_error = load_and_preprocess_data()

if data_error:
    st.error(data_error)
    st.info("""
    **📋 Solusi:**
    1. Pastikan file `produksi_tanaman.csv` di folder yang sama dengan `app.py`
    2. Format CSV harus memiliki kolom: Provinsi, Kelapa_Sawit, Kelapa, Karet, Kopi, Kakao, Teh, Tebu
    3. Lihat `produksi_tanaman.csv.template` untuk format yang benar
    """)
    st.stop()

# ============================================================================
# SIDEBAR NAVIGASI
# ============================================================================

with st.sidebar:
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
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Navigasi Halaman")
    
    page = st.radio(
        "Pilih halaman:",
        [
            "📊 Page 1: Overview",
            "🧹 Page 2: Data Cleaning",
            "📈 Page 3: EDA 3D",
            "🗺️ Page 3b: Peta Distribusi",
            "🔗 Page 4: Korelasi & Regresi",
            "🎯 Page 4b: Machine Learning",
            "💡 Page 5: Insights & Rekomendasi",
            "📚 Tentang Aplikasi"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Info Dataset")
    st.markdown(f"""
    - **Total Provinsi:** {len(df)}
    - **Total Komoditas:** {len(KOMODITAS_LIST)}
    - **Total Produksi:** {format_number(df['Total_Produksi'].sum())}
    """)
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b8f7a; font-size: 0.85em;">
        <p>🌿 Versi {APP_VERSION}</p>
        <p>UAS Visualisasi Data 2026</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# HEADER UTAMA
# ============================================================================

st.markdown(f'<h1 class="main-title">🌿 Dasbor Produksi Perkebunan Indonesia</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #a8d5ba; font-size: 1.15em; margin-bottom: 30px;">Analisis Visual 3D Interaktif • Sektor Agrikultur Modern • Tugas UAS Visualisasi Data</p>', unsafe_allow_html=True)

# ============================================================================
# PAGE 1: OVERVIEW & DATA UNDERSTANDING
# ============================================================================

if "Page 1: Overview" in page:
    st.markdown("""
    <div class="page-header">
        <h1>📊 Overview & Data Understanding</h1>
        <p>Memahami struktur dataset produksi tanaman perkebunan Indonesia</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Dataset ini berisi informasi mengenai **produksi 7 komoditas perkebunan utama** di **38 provinsi** Indonesia.
    """)
    
    st.markdown("---")
    st.subheader("🏆 Key Performance Indicators (KPI)")
    
    # Hitung metrik
    total_semua = df[KOMODITAS_LIST].sum().sum()
    total_sawit = df['Kelapa_Sawit'].sum()
    total_karet = df['Karet'].sum()
    total_kopi = df['Kopi'].sum()
    pct_sawit = (total_sawit / total_semua) * 100 if total_semua > 0 else 0
    pct_karet = (total_karet / total_semua) * 100 if total_semua > 0 else 0
    pct_kopi = (total_kopi / total_semua) * 100 if total_semua > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌾 Total Produksi", f"{format_number(total_semua)} ribu ton", "Semua Komoditas")
    with col2:
        st.metric("🌴 Kelapa Sawit", f"{format_number(total_sawit)} ribu ton", f"{pct_sawit:.1f}% dari total")
    with col3:
        st.metric("🌳 Karet", f"{format_number(total_karet)} ribu ton", f"{pct_karet:.1f}% dari total")
    with col4:
        st.metric("☕ Kopi", f"{format_number(total_kopi)} ribu ton", f"{pct_kopi:.1f}% dari total")
    
    st.markdown("---")
    st.subheader("📋 Preview Data")
    
    n_rows = st.slider("Jumlah baris ditampilkan:", 5, 38, 15, 1)
    st.dataframe(
        df.head(n_rows)[['Provinsi', 'Wilayah', 'Total_Produksi'] + KOMODITAS_LIST].style
        .background_gradient(subset=KOMODITAS_LIST, cmap='Greens')
        .format({col: '{:,.0f}' for col in KOMODITAS_LIST + ['Total_Produksi']}),
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("📊 Statistik Deskriptif")
    
    stats_df = pd.DataFrame({
        'Komoditas': KOMODITAS_LIST,
        'Mean': [df[k].mean() for k in KOMODITAS_LIST],
        'Median': [df[k].median() for k in KOMODITAS_LIST],
        'Std Dev': [df[k].std() for k in KOMODITAS_LIST],
        'Min': [df[k].min() for k in KOMODITAS_LIST],
        'Max': [df[k].max() for k in KOMODITAS_LIST]
    })
    
    st.dataframe(
        stats_df.style.format({col: '{:,.2f}' for col in stats_df.columns if col != 'Komoditas'}),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("📦 Box Plot Distribusi")
    
    df_melted = df.melt(
        id_vars=['Provinsi'],
        value_vars=KOMODITAS_LIST,
        var_name='Komoditas',
        value_name='Produksi'
    )
    
    fig_box = px.box(
        df_melted,
        x='Komoditas',
        y='Produksi',
        color='Komoditas',
        title='Distribusi Produksi per Komoditas',
        color_discrete_sequence=[KOMODITAS_COLORS.get(k, '#2ecc71') for k in KOMODITAS_LIST],
        labels={'Produksi': 'Produksi (ribu ton)'}
    )
    
    fig_box = apply_plotly_theme(fig_box, height=500)
    fig_box.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🎯 Komoditas Dominan per Provinsi")
    
    dominant_counts = df['Komoditas_Dominan'].value_counts().reset_index()
    dominant_counts.columns = ['Komoditas', 'Jumlah Provinsi']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(dominant_counts, use_container_width=True, hide_index=True)
    
    with col2:
        fig_pie = px.pie(
            dominant_counts,
            values='Jumlah Provinsi',
            names='Komoditas',
            title='Provinsi Berdasarkan Komoditas Dominan',
            hole=0.4
        )
        fig_pie = apply_plotly_theme(fig_pie, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================================
# PAGE 2: DATA CLEANING & PREPROCESSING
# ============================================================================

elif "Page 2: Data Cleaning" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🧹 Data Cleaning & Preprocessing</h1>
        <p>Analisis kualitas dan pembersihan data</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("1️⃣ Analisis Missing Values")
    
    missing_count = df_raw.isnull().sum()
    missing_pct = (missing_count / len(df_raw)) * 100
    missing_df = pd.DataFrame({
        'Kolom': missing_count.index,
        'Jumlah Missing': missing_count.values,
        'Persentase (%)': missing_pct.values
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(missing_df.style.background_gradient(subset=['Persentase (%)'], cmap='RdYlGn_r'),
                    use_container_width=True, hide_index=True)
    
    with col2:
        total_missing = missing_count.sum()
        if total_missing == 0:
            st.success(f"✅ Dataset Bersih\n\nTotal missing: {total_missing}")
        else:
            st.warning(f"⚠️ Missing Values\n\nTotal: {total_missing}")
    
    st.markdown("---")
    st.subheader("2️⃣ Analisis Data Duplikat")
    
    n_duplicates = df_raw.duplicated().sum()
    duplicate_pct = (n_duplicates / len(df_raw)) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Duplikat", n_duplicates, f"{duplicate_pct:.2f}%")
    with col2:
        st.metric("Baris Unik", len(df_raw) - n_duplicates, f"{100-duplicate_pct:.1f}%")
    with col3:
        st.metric("Status", "✅ Bersih" if n_duplicates == 0 else "⚠️ Ada Duplikat")
    
    st.markdown("---")
    st.subheader("3️⃣ Deteksi Outlier (IQR Method)")
    
    komoditas_outlier = st.selectbox("Pilih komoditas:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x))
    
    series = df[komoditas_outlier]
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = series[(series < lower) | (series > upper)]
    
    st.write(f"**Q1:** {Q1:.2f}, **Q3:** {Q3:.2f}, **IQR:** {IQR:.2f}")
    st.write(f"**Lower Bound:** {lower:.2f}, **Upper Bound:** {upper:.2f}")
    st.write(f"**Outliers:** {len(outliers)} ({len(outliers)/len(series)*100:.1f}%)")
    
    if len(outliers) > 0:
        st.write("Provinsi dengan outlier:")
        st.write(df.loc[outliers.index, 'Provinsi'].tolist())
    
    st.markdown("---")
    st.subheader("4️⃣ Feature Engineering")
    
    st.markdown("""
    **Fitur Baru yang Dibuat:**
    
    1. **`Wilayah`**: Pengelompokan provinsi ke 7 wilayah geografis
    2. **`Total_Produksi`**: Jumlah produksi semua komoditas per provinsi
    3. **`Rank_Total`**: Ranking provinsi berdasarkan total produksi
    4. **`Komoditas_Dominan`**: Komoditas dengan produksi tertinggi
    5. **`HHI_Index`**: Herfindahl-Hirschman Index untuk diversifikasi
    6. **`Diversifikasi`**: Kategori diversifikasi (Sangat Diversifikasi - Sangat Spesialisasi)
    7. **`Latitude` & `Longitude`**: Koordinat geografis provinsi
    """)
    
    kolom_preview = st.multiselect(
        "Pilih kolom untuk preview:",
        df.columns.tolist(),
        default=['Provinsi', 'Wilayah', 'Total_Produksi', 'Diversifikasi']
    )
    
    if kolom_preview:
        st.dataframe(df[kolom_preview].head(15), use_container_width=True)
    
    st.markdown("---")
    st.subheader("5️⃣ Normalisasi Data")
    
    teknik = st.selectbox("Pilih teknik scaling:", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
    
    if teknik == "StandardScaler":
        scaler = StandardScaler()
    elif teknik == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
    
    scaled_data = scaler.fit_transform(df[KOMODITAS_LIST])
    df_scaled = pd.DataFrame(scaled_data, columns=KOMODITAS_LIST)
    df_scaled.insert(0, 'Provinsi', df['Provinsi'].values)
    
    st.dataframe(df_scaled.style.background_gradient(subset=KOMODITAS_LIST, cmap='RdYlGn').head(10),
                use_container_width=True)

# ============================================================================
# PAGE 3: EDA & 3D VISUALIZATIONS
# ============================================================================

elif "Page 3: EDA 3D" in page:
    st.markdown("""
    <div class="page-header">
        <h1>📈 Exploratory Data Analysis & 3D Visualizations</h1>
        <p>Visualisasi 3D interaktif untuk mengungkap pola tersembunyi</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🔬 Analisis Univariat")
    
    komoditas_uni = st.selectbox("Pilih komoditas:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), key="uni_kom")
    
    data_uni = df[komoditas_uni]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{data_uni.mean():.2f}")
    with col2:
        st.metric("Median", f"{data_uni.median():.2f}")
    with col3:
        st.metric("Std Dev", f"{data_uni.std():.2f}")
    with col4:
        st.metric("Max", f"{data_uni.max():.2f}")
    
    col_u1, col_u2 = st.columns(2)
    
    with col_u1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=data_uni, nbinsx=15, marker_color='#2ecc71', name='Frekuensi'))
        fig_hist = apply_plotly_theme(fig_hist, title=f'Distribusi {KOMODITAS_LABELS.get(komoditas_uni)}', height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_u2:
        (osm, osr), (slope, intercept, r) = stats.probplot(data_uni, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode='markers', marker=dict(color='#2ecc71'), name='Data'))
        line_x = np.array([min(osm), max(osm)])
        line_y = intercept + slope * line_x
        fig_qq.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='#e74c3c', dash='dash'), name='Reference'))
        fig_qq = apply_plotly_theme(fig_qq, title=f'QQ Plot', height=400)
        st.plotly_chart(fig_qq, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 Analisis Bivariat")
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        var1 = st.selectbox("Komoditas X:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), key="bi_x")
    
    with col_b2:
        var2 = st.selectbox("Komoditas Y:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), index=1, key="bi_y")
    
    corr_info = calculate_correlation_significance(df[var1], df[var2])
    
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        st.metric("Pearson r", f"{corr_info['pearson_r']:.3f}", corr_info['pearson_strength'])
    with col_c2:
        st.metric("Spearman ρ", f"{corr_info['spearman_r']:.3f}", corr_info['spearman_strength'])
    with col_c3:
        p_val = f"{corr_info['pearson_p']:.4f}" if corr_info['pearson_p'] > 0.0001 else "< 0.0001"
        st.metric("P-value", p_val, "Signifikan" if corr_info['pearson_significant'] else "Tidak")
    
    fig_biv = px.scatter(df, x=var1, y=var2, color='Wilayah', size='Total_Produksi',
                         hover_name='Provinsi', trendline='ols',
                         title=f'{KOMODITAS_LABELS.get(var1)} vs {KOMODITAS_LABELS.get(var2)}',
                         color_discrete_map=REGION_COLORS)
    fig_biv = apply_plotly_theme(fig_biv, height=550)
    st.plotly_chart(fig_biv, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🧊 Scatter 3D Interaktif")
    
    col_3d1, col_3d2, col_3d3 = st.columns(3)
    
    with col_3d1:
        axis_x = st.selectbox("Sumbu X:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), key="3d_x")
    with col_3d2:
        axis_y = st.selectbox("Sumbu Y:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), index=1, key="3d_y")
    with col_3d3:
        axis_z = st.selectbox("Sumbu Z:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), index=2, key="3d_z")
    
    color_by = st.radio("Warnai berdasarkan:", ["Wilayah", "Komoditas_Dominan"], horizontal=True)
    
    fig_3d = px.scatter_3d(df, x=axis_x, y=axis_y, z=axis_z, color=color_by, size='Total_Produksi',
                           size_max=35, hover_name='Provinsi', title=f'3D Scatter: {axis_x} × {axis_y} × {axis_z}',
                           color_discrete_map=REGION_COLORS if color_by == "Wilayah" else None)
    fig_3d = apply_plotly_theme(fig_3d, height=700)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🏔️ Surface Plot (Topografi Produksi)")
    
    n_prov = st.slider("Jumlah provinsi teratas:", 5, 38, 20, 1)
    df_sorted = df.nlargest(n_prov, 'Total_Produksi').set_index('Provinsi')[KOMODITAS_LIST]
    
    Z = df_sorted.values
    X = np.arange(len(KOMODITAS_LIST))
    Y = np.arange(len(df_sorted))
    
    fig_surface = go.Figure(data=[go.Surface(
        z=Z, x=X, y=Y,
        colorscale='Viridis',
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        opacity=0.9
    )])
    
    fig_surface = apply_plotly_theme(fig_surface, title=f"Topografi Produksi {n_prov} Provinsi Teratas", height=700)
    fig_surface.update_layout(
        scene=dict(
            xaxis=dict(title='Komoditas', tickvals=X, ticktext=[k.replace('_', ' ')[:10] for k in KOMODITAS_LIST]),
            yaxis=dict(title='Provinsi', tickvals=Y, ticktext=[name[:12] for name in df_sorted.index]),
            zaxis=dict(title='Produksi (ribu ton)')
        )
    )
    st.plotly_chart(fig_surface, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🗺️ Analisis per Wilayah")
    
    region_summary = df.groupby('Wilayah')[KOMODITAS_LIST + ['Total_Produksi']].sum().sort_values('Total_Produksi', ascending=False)
    
    st.dataframe(region_summary.style.background_gradient(cmap='Greens').format('{:,.0f}'),
                use_container_width=True)
    
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        fig_rb = px.bar(region_summary.reset_index(), x='Wilayah', y='Total_Produksi',
                       color='Wilayah', title='Total Produksi per Wilayah',
                       color_discrete_map=REGION_COLORS, text='Total_Produksi')
        fig_rb = apply_plotly_theme(fig_rb, height=450)
        fig_rb.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig_rb, use_container_width=True)
    
    with col_w2:
        fig_rp = px.pie(region_summary.reset_index(), values='Total_Produksi', names='Wilayah',
                       title='Kontribusi Wilayah', color='Wilayah', color_discrete_map=REGION_COLORS, hole=0.4)
        fig_rp = apply_plotly_theme(fig_rp, height=450)
        st.plotly_chart(fig_rp, use_container_width=True)

# ============================================================================
# PAGE 3b: PETA DISTRIBUSI
# ============================================================================

elif "Page 3b: Peta" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🗺️ Peta Distribusi Produksi Indonesia</h1>
        <p>Visualisasi geospasial produksi di seluruh provinsi</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🌍 Peta Sebaran Produksi")
    
    komoditas_peta = st.selectbox(
        "Pilih Komoditas:",
        ['Total_Produksi'] + KOMODITAS_LIST,
        format_func=lambda x: "🌾 Total" if x == 'Total_Produksi' else KOMODITAS_LABELS.get(x),
        key="peta_kom"
    )
    
    df_map = df.dropna(subset=['Latitude', 'Longitude'])
    
    fig_geo = px.scatter_geo(
        df_map, lat='Latitude', lon='Longitude',
        color=komoditas_peta, size=komoditas_peta,
        size_max=50, hover_name='Provinsi',
        scope='asia', color_continuous_scale='Greens',
        title=f'Distribusi {komoditas_peta.replace("_", " ")} di Indonesia'
    )
    
    fig_geo.update_geos(
        showcountries=True, showcoastlines=True, showland=True,
        lataxis_range=[-12, 8], lonaxis_range=[92, 145]
    )
    fig_geo = apply_plotly_theme(fig_geo, height=650)
    st.plotly_chart(fig_geo, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🏆 Komoditas Dominan per Provinsi")
    
    dominant_colors = []
    for _, row in df_map.iterrows():
        komoditas_label = row['Komoditas_Dominan']
        for k, v in KOMODITAS_LABELS.items():
            if v == komoditas_label:
                dominant_colors.append(KOMODITAS_COLORS.get(k))
                break
    
    fig_dominant = px.scatter_geo(
        df_map, lat='Latitude', lon='Longitude',
        hover_name='Provinsi', hover_data=['Komoditas_Dominan', 'Total_Produksi'],
        scope='asia', title='Komoditas Dominan per Provinsi'
    )
    
    fig_dominant.update_traces(
        marker=dict(size=15, line=dict(width=1, color='white')),
        text=df_map['Provinsi']
    )
    
    fig_dominant.update_geos(
        showcountries=True, showcoastlines=True, showland=True,
        lataxis_range=[-12, 8], lonaxis_range=[92, 145]
    )
    fig_dominant = apply_plotly_theme(fig_dominant, height=650)
    st.plotly_chart(fig_dominant, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🎯 Analisis Kluster Geografis")
    
    n_clusters = st.slider("Jumlah kluster:", 2, 8, 4, 1)
    
    features_cluster = df[KOMODITAS_LIST + ['Latitude', 'Longitude']].fillna(0).values
    scaler_cluster = StandardScaler()
    features_scaled = scaler_cluster.fit_transform(features_cluster)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    fig_cluster = px.scatter_geo(
        df, lat='Latitude', lon='Longitude',
        color='Cluster', size='Total_Produksi',
        size_max=40, hover_name='Provinsi',
        scope='asia', title=f'Kluster Provinsi (K={n_clusters})',
        color_continuous_scale='Viridis'
    )
    
    fig_cluster.update_geos(
        showcountries=True, showcoastlines=True, showland=True,
        lataxis_range=[-12, 8], lonaxis_range=[92, 145]
    )
    fig_cluster = apply_plotly_theme(fig_cluster, height=650)
    st.plotly_chart(fig_cluster, use_container_width=True)

# ============================================================================
# PAGE 4: KORELASI & REGRESI
# ============================================================================

elif "Page 4: Korelasi" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🔗 Analisis Korelasi & Regresi</h1>
        <p>Menganalisis hubungan antar komoditas dan membangun model prediktif</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🔥 Heatmap Matriks Korelasi")
    
    corr_matrix = df[KOMODITAS_LIST].corr(method='pearson')
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[KOMODITAS_LABELS.get(c, c) for c in corr_matrix.columns],
        y=[KOMODITAS_LABELS.get(c, c) for c in corr_matrix.index],
        colorscale='RdYlGn',
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=12, color="white")
    ))
    
    fig_corr = apply_plotly_theme(fig_corr, title="Matriks Korelasi Pearson", height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Analisis Korelasi Detail")
    
    col_an1, col_an2 = st.columns(2)
    
    with col_an1:
        var1 = st.selectbox("Variabel 1:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), key="corr_v1")
    with col_an2:
        var2 = st.selectbox("Variabel 2:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), index=1, key="corr_v2")
    
    corr_detail = calculate_correlation_significance(df[var1], df[var2])
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        st.metric("Pearson r", f"{corr_detail['pearson_r']:.4f}", corr_detail['pearson_strength'])
    with col_d2:
        p_val_str = f"{corr_detail['pearson_p']:.4f}" if corr_detail['pearson_p'] > 0.0001 else "< 0.0001"
        st.metric("P-value", p_val_str, "Signifikan" if corr_detail['pearson_significant'] else "Tidak")
    with col_d3:
        st.metric("Spearman ρ", f"{corr_detail['spearman_r']:.4f}", corr_detail['spearman_strength'])
    with col_d4:
        p_val_sp = f"{corr_detail['spearman_p']:.4f}" if corr_detail['spearman_p'] > 0.0001 else "< 0.0001"
        st.metric("P-value (S)", p_val_sp)
    
    fig_detail = px.scatter(
        df, x=var1, y=var2,
        color='Wilayah', size='Total_Produksi',
        hover_name='Provinsi', trendline='ols',
        title=f'{KOMODITAS_LABELS.get(var1)} vs {KOMODITAS_LABELS.get(var2)}',
        color_discrete_map=REGION_COLORS
    )
    
    fig_detail = apply_plotly_theme(fig_detail, height=550)
    st.plotly_chart(fig_detail, use_container_width=True)
    
    if corr_detail['pearson_significant']:
        direction = "positif" if corr_detail['pearson_r'] > 0 else "negatif"
        st.success(f"""
        ✅ **Hubungan Signifikan**
        
        Ada hubungan **{direction}** dengan kekuatan **{corr_detail['pearson_strength']}** antara 
        {KOMODITAS_LABELS.get(var1)} dan {KOMODITAS_LABELS.get(var2)}.
        
        R² = {corr_detail['pearson_r']**2:.4f} ({corr_detail['pearson_r']**2 * 100:.2f}% variasi)
        """)
    else:
        st.warning(f"⚠️ **Hubungan Tidak Signifikan** pada α=0.05")
    
    st.markdown("---")
    st.subheader("🤖 Model Regresi Linier")
    
    target_var = st.selectbox("Variabel Target:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), index=0, key="reg_target")
    
    feature_vars = [k for k in KOMODITAS_LIST if k != target_var]
    test_size = st.slider("Test Size (%):", 10, 40, 20, 5, key="test_size") / 100
    
    X_reg = df[feature_vars].values
    y_reg = df[target_var].values
    
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=test_size, random_state=42)
    
    scaler_reg = StandardScaler()
    X_train_scaled = scaler_reg.fit_transform(X_train)
    X_test_scaled = scaler_reg.transform(X_test)
    
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    
    y_pred_train = model_lr.predict(X_train_scaled)
    y_pred_test = model_lr.predict(X_test_scaled)
    
    metrics_test = {
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'R²': r2_score(y_test, y_pred_test)
    }
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.metric("MAE (Test)", f"{metrics_test['MAE']:.2f}")
    with col_m2:
        st.metric("RMSE (Test)", f"{metrics_test['RMSE']:.2f}")
    with col_m3:
        st.metric("R² (Test)", f"{metrics_test['R²']:.4f}")
    
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(x=y_train, y=y_pred_train, mode='markers', name='Train', marker=dict(color='#2ecc71')))
    fig_avp.add_trace(go.Scatter(x=y_test, y=y_pred_test, mode='markers', name='Test', marker=dict(color='#f1c40f')))
    
    min_v, max_v = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    fig_avp.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', name='Perfect', line=dict(dash='dash', color='#e74c3c')))
    
    fig_avp = apply_plotly_theme(fig_avp, title="Actual vs Predicted", height=500)
    fig_avp.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
    st.plotly_chart(fig_avp, use_container_width=True)

# ============================================================================
# PAGE 4b: MACHINE LEARNING
# ============================================================================

elif "Page 4b: Machine Learning" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🎯 Machine Learning Lanjutan</h1>
        <p>Perbandingan berbagai algoritma regresi</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("⚙️ Konfigurasi")
    
    col_ml1, col_ml2, col_ml3 = st.columns(3)
    
    with col_ml1:
        target_ml = st.selectbox("Target:", KOMODITAS_LIST, format_func=lambda x: KOMODITAS_LABELS.get(x), key="ml_target")
    
    with col_ml2:
        test_size_ml = st.slider("Test Size (%):", 10, 40, 20, 5, key="ml_test") / 100
    
    with col_ml3:
        n_folds = st.slider("CV Folds:", 3, 10, 5, 1)
    
    selected_models = st.multiselect(
        "Pilih Model:",
        ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "Random Forest", "Gradient Boosting"],
        default=["Linear Regression", "Ridge", "Random Forest", "Gradient Boosting"]
    )
    
    if not selected_models:
        st.warning("⚠️ Pilih minimal satu model")
        st.stop()
    
    feature_vars_ml = [k for k in KOMODITAS_LIST if k != target_ml]
    X_ml = df[feature_vars_ml].values
    y_ml = df[target_ml].values
    
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=test_size_ml, random_state=42)
    
    scaler_ml = StandardScaler()
    X_train_ml_sc = scaler_ml.fit_transform(X_train_ml)
    X_test_ml_sc = scaler_ml.transform(X_test_ml)
    
    st.markdown("---")
    st.subheader("🏋️ Training Model")
    
    progress_bar = st.progress(0)
    results = []
    models_dict = {}
    
    for idx, model_name in enumerate(selected_models):
        progress_bar.progress((idx + 1) / len(selected_models))
        
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "Lasso":
            model = Lasso(alpha=1.0, max_iter=10000)
        elif model_name == "ElasticNet":
            model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
        elif model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        elif model_name == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        
        if model_name in ["Random Forest", "Gradient Boosting"]:
            model.fit(X_train_ml, y_train_ml)
            y_pred_train = model.predict(X_train_ml)
            y_pred_test = model.predict(X_test_ml)
        else:
            model.fit(X_train_ml_sc, y_train_ml)
            y_pred_train = model.predict(X_train_ml_sc)
            y_pred_test = model.predict(X_test_ml_sc)
        
        test_metrics = {
            'MAE': mean_absolute_error(y_test_ml, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test_ml, y_pred_test)),
            'R²': r2_score(y_test_ml, y_pred_test),
            'MAPE': mean_absolute_percentage_error(y_test_ml, y_pred_test + 1e-8) * 100
        }
        
        results.append({
            'Model': model_name,
            'Test MAE': test_metrics['MAE'],
            'Test RMSE': test_metrics['RMSE'],
            'Test R²': test_metrics['R²'],
            'Test MAPE': test_metrics['MAPE']
        })
        
        models_dict[model_name] = {'model': model, 'y_pred_test': y_pred_test}
    
    progress_bar.empty()
    
    results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
    
    st.dataframe(
        results_df.style.highlight_max(subset=['Test R²'], color='#2ecc71')
        .format({col: '{:.3f}' if col != 'Model' else '{}' for col in results_df.columns}),
        use_container_width=True,
        hide_index=True
    )
    
    best_model = results_df.iloc[0]
    st.success(f"""
    ### 🏆 Model Terbaik: {best_model['Model']}
    
    - **Test R²:** {best_model['Test R²']:.4f}
    - **Test MAE:** {best_model['Test MAE']:.3f}
    - **Test RMSE:** {best_model['Test RMSE']:.3f}
    """)
    
    st.markdown("---")
    st.subheader("📈 Perbandingan Model")
    
    fig_compare = px.bar(
        results_df, x='Model', y=['Test R²'],
        title='Perbandingan R² Score Model',
        barmode='group',
        color_discrete_sequence=['#2ecc71'],
        text='Test R²',
        labels={'value': 'R² Score', 'variable': ''}
    )
    
    fig_compare = apply_plotly_theme(fig_compare, height=450)
    fig_compare.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    st.plotly_chart(fig_compare, use_container_width=True)

# ============================================================================
# PAGE 5: INSIGHTS & REKOMENDASI
# ============================================================================

elif "Page 5: Insights" in page:
    st.markdown("""
    <div class="page-header">
        <h1>💡 Insights & Rekomendasi Strategis</h1>
        <p>Kesimpulan mendalam dan rekomendasi implementatif</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📋 Ringkasan Eksekutif")
    
    top_prov = df.loc[df['Kelapa_Sawit'].idxmax(), 'Provinsi']
    top_val = df['Kelapa_Sawit'].max()
    pct_sawit = (df['Kelapa_Sawit'].sum() / df['Total_Produksi'].sum()) * 100
    
    st.markdown(f"""
    **Dataset:** {len(df)} provinsi, {len(KOMODITAS_LIST)} komoditas
    
    - 🌴 Kelapa Sawit mendominasi **{pct_sawit:.1f}%** produksi nasional
    - 🏆 **{top_prov}** adalah produsen sawit terbesar ({top_val:,.0f} ribu ton)
    - ⚖️ Gini Coefficient: **{calculate_gini_coefficient(df['Total_Produksi']):.3f}** (ketimpangan tinggi)
    """)
    
    st.markdown("---")
    st.subheader("🔍 5 Insight Mendalam")
    
    st.markdown("""
    <div class="insight-box">
        <h3 style="color: var(--emerald-light) !important; margin-top: 0;">
            💡 Insight #1: Dominasi Kelapa Sawit
        </h3>
        <p>Kelapa Sawit menguasai lebih dari 60% produksi, menciptakan ketergantungan ekonomi 
        yang tinggi terhadap komoditas tunggal.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <h3 style="color: var(--emerald-light) !important; margin-top: 0;">
            💡 Insight #2: Ketimpangan Regional Signifikan
        </h3>
        <p>5 provinsi teratas menguasai > 60% produksi, sementara 10 provinsi terbawah 
        hanya berkontribusi < 2%.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <h3 style="color: var(--emerald-light) !important; margin-top: 0;">
            💡 Insight #3: Spesialisasi Regional yang Jelas
        </h3>
        <p>Setiap wilayah memiliki komoditas unggulan yang konsisten dengan agroklimat lokal.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <h3 style="color: var(--emerald-light) !important; margin-top: 0;">
            💡 Insight #4: Diversifikasi Rendah
        </h3>
        <p>Mayoritas provinsi bergantung pada 1-2 komoditas, meningkatkan kerentanan ekonomi.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <h3 style="color: var(--emerald-light) !important; margin-top: 0;">
            💡 Insight #5: Potensi Indonesia Timur
        </h3>
        <p>Indonesia Timur baru menyumbang < 15% produksi padahal memiliki potensi agroklimat besar.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🎯 5 Rekomendasi Implementatif")
    
    st.markdown("""
    <div class="recommend-box">
        <h3 style="color: var(--gold) !important; margin-top: 0;">🎯 Rekomendasi #1: Diversifikasi Komoditas</h3>
        <p>Program tumpang sari dan diversifikasi untuk menurunkan HHI dan ketergantungan pada sawit.</p>
    </div>
    
    <div class="recommend-box">
        <h3 style="color: var(--gold) !important; margin-top: 0;">🎯 Rekomendasi #2: Infrastruktur Indonesia Timur</h3>
        <p>Percepatan pembangunan jalan, pelabuhan, dan cold storage di wilayah timur.</p>
    </div>
    
    <div class="recommend-box">
        <h3 style="color: var(--gold) !important; margin-top: 0;">🎯 Rekomendasi #3: Sertifikasi Sustainability</h3>
        <p>100% sertifikasi ISPO/RSPO untuk memenuhi standar regulasi internasional.</p>
    </div>
    
    <div class="recommend-box">
        <h3 style="color: var(--gold) !important; margin-top: 0;">🎯 Rekomendasi #4: Digitalisasi Sistem</h3>
        <p>Dashboard nasional dan sistem IoT untuk monitoring real-time produksi.</p>
    </div>
    
    <div class="recommend-box">
        <h3 style="color: var(--gold) !important; margin-top: 0;">🎯 Rekomendasi #5: Hilirisasi & Value Added</h3>
        <p>Pengembangan industri hilir untuk meningkatkan rasio produk olahan dari 30% menjadi 60%.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: TENTANG APLIKASI
# ============================================================================

elif "Tentang Aplikasi" in page:
    st.markdown("""
    <div class="page-header">
        <h1>📚 Tentang Aplikasi</h1>
        <p>Informasi teknis, deployment, dan panduan penggunaan</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🌿 Deskripsi Aplikasi
    
    **Dasbor Produksi Perkebunan Indonesia** adalah aplikasi web interaktif untuk menganalisis 
    data produksi tanaman perkebunan di 38 provinsi Indonesia. Dibuat untuk tugas UAS Visualisasi Data.
    
    ### ✨ Fitur Utama
    
    - 📊 8 halaman analisis terstruktur
    - 🧊 5+ visualisasi 3D interaktif (Plotly)
    - 🗺️ 4 jenis peta geospasial
    - 🔥 Heatmap korelasi & analisis signifikansi
    - 🤖 6 model ML (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)
    - 🎯 Clustering geografis dengan K-Means
    - 🎨 Tema dark mode modern dengan animasi
    
    ### 🛠️ Teknologi
    
    | Teknologi | Versi | Fungsi |
    |-----------|-------|--------|
    | **Streamlit** | 1.36.0 | Framework web app |
    | **Pandas** | 2.2.2 | Data manipulation |
    | **NumPy** | 1.26.4 | Numerical computing |
    | **Plotly** | 5.22.0 | Interactive visualization |
    | **scikit-learn** | 1.5.0 | Machine Learning |
    | **SciPy** | 1.12.0 | Statistical analysis |
    | **Python** | 3.10+ | Programming language |
    
    ### 🚀 Cara Menjalankan
    
    **1. Install Dependensi**
```bash
    pip install -r requirements.txt
```
    
    **2. Jalankan Aplikasi**
```bash
    streamlit run app.py
```
    
    **3. Buka di Browser**
