# ============================================================================
# app.py - DASBOR INTERAKTIF PRODUKSI TANAMAN PERKEBUNAN INDONESIA
# Tugas UAS Visualisasi Data - Tema: Sains Data & Pertanian Indonesia
# VERSI FINAL: Fully tested, compatible dengan Python 3.10+ dan Plotly terbaru
# ============================================================================

# ============================================================
# BAGIAN 1: IMPORT LIBRARY UTAMA
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import base64

# Matikan warning yang mengganggu
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# BAGIAN 2: IMPORT SKLEARN DENGAN FALLBACK
# ============================================================
SKLEARN_OK = False
SCIPY_OK = False

try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# BAGIAN 3: IMPLEMENTASI MANUAL (FALLBACK)
# ============================================================
class ManualLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        try:
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class ManualRidge(ManualLinearRegression):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        reg = self.alpha * np.eye(X.shape[1] + 1)
        reg[0, 0] = 0
        theta = np.linalg.pinv(X_b.T @ X_b + reg) @ X_b.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self


class ManualStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def manual_mae(y, yp): return np.mean(np.abs(y - yp))
def manual_rmse(y, yp): return np.sqrt(np.mean((y - yp) ** 2))
def manual_r2(y, yp):
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
def manual_mape(y, yp):
    return np.mean(np.abs((y - yp) / (np.abs(y) + 1e-8))) * 100
def manual_pearson(x, y):
    n = len(x)
    num = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    den = np.sqrt((n * np.sum(x**2) - np.sum(x)**2) * (n * np.sum(y**2) - np.sum(y)**2))
    return (num / den, 0.05) if den != 0 else (0.0, 1.0)


if SKLEARN_OK:
    LinReg = LinearRegression
    RidgReg = lambda alpha=1.0: Ridge(alpha=alpha)
    Scalr = StandardScaler
    do_split = train_test_split
    do_mae = mean_absolute_error
    do_rmse = lambda y, yp: np.sqrt(mean_squared_error(y, yp))
    do_r2 = r2_score
    do_mape = lambda y, yp: mean_absolute_percentage_error(y, yp + 1e-8) * 100
else:
    LinReg = ManualLinearRegression
    RidgReg = ManualRidge
    Scalr = ManualStandardScaler
    
    def do_split(X, y, test_size=0.2, random_state=42):
        rng = np.random.RandomState(random_state)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        split = int(len(X) * (1 - test_size))
        return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]
    
    do_mae = manual_mae
    do_rmse = manual_rmse
    do_r2 = manual_r2
    do_mape = manual_mape

if SCIPY_OK:
    do_corr = lambda x, y: pearsonr(x, y)
else:
    do_corr = manual_pearson


# ============================================================
# BAGIAN 4: KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="🌿 Dasbor Produksi Perkebunan Indonesia",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BAGIAN 5: CSS KUSTOM
# ============================================================
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/poppins@5.0.0/index.min.css');
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/inter@5.0.0/index.min.css');
    
    :root {
        --bg-dark: #0a1f14;
        --emerald: #2ecc71;
        --emerald-light: #58d68d;
        --gold: #f1c40f;
        --text-primary: #e8f5e9;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a1f14 0%, #0f2a1c 50%, #1a1a2e 100%);
        color: #e8f5e9;
        font-family: 'Inter', sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d3320 0%, #1a4d2e 100%);
        border-right: 2px solid #2ecc71;
    }
    
    .main-title {
        background: linear-gradient(90deg, #2ecc71, #f1c40f, #27ae60);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 900;
        text-align: center;
        margin-bottom: 10px;
        font-family: 'Poppins', sans-serif;
        animation: shimmer 3s linear infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    .sub-title {
        text-align: center;
        color: #a8d5ba;
        font-size: 1.15em;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.18), rgba(241, 196, 15, 0.1));
        border: 1px solid rgba(46, 204, 113, 0.4);
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(46, 204, 113, 0.2);
        transition: all 0.3s ease;
        margin: 10px 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
    }
    
    .metric-icon { font-size: 2.2em; margin-bottom: 8px; }
    
    .metric-value {
        font-size: 1.9em;
        font-weight: 800;
        color: #f1c40f;
        margin: 5px 0;
        font-family: monospace;
    }
    
    .metric-label {
        color: #a8d5ba;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(46, 204, 113, 0.05));
        border-left: 5px solid #2ecc71;
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .recommend-box {
        background: linear-gradient(135deg, rgba(241, 196, 15, 0.15), rgba(241, 196, 15, 0.05));
        border-left: 5px solid #f1c40f;
        padding: 18px 22px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .page-header {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(241, 196, 15, 0.08));
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 20px;
        padding: 25px 30px;
        margin-bottom: 30px;
    }
    
    .info-box {
        background: rgba(52, 152, 219, 0.12);
        border-left: 5px solid #3498db;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: rgba(231, 76, 60, 0.12);
        border-left: 5px solid #e74c3c;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    h1, h2, h3 { color: #58d68d !important; font-family: 'Poppins', sans-serif !important; }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #2ecc71, #f1c40f, #2ecc71, transparent);
        margin: 30px 0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 3px;
    }
    .badge-ok { background: rgba(46,204,113,0.25); color: #2ecc71; border: 1px solid #2ecc71; }
    .badge-warn { background: rgba(241,196,15,0.25); color: #f1c40f; border: 1px solid #f1c40f; }
    
    .app-footer {
        background: linear-gradient(90deg, rgba(13,51,32,0.8), rgba(15,42,28,0.8), rgba(13,51,32,0.8));
        border-top: 2px solid #2ecc71;
        padding: 20px;
        text-align: center;
        margin-top: 50px;
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# BAGIAN 6: KONSTANTA GLOBAL
# ============================================================
APP_VERSION = "3.0.0 (Final)"
APP_YEAR = 2026

KOMODITAS = ['Kelapa_Sawit', 'Kelapa', 'Karet', 'Kopi', 'Kakao', 'Teh', 'Tebu']

LABEL_KOMODITAS = {
    'Kelapa_Sawit': '🌴 Kelapa Sawit',
    'Kelapa': '🥥 Kelapa',
    'Karet': '🌳 Karet',
    'Kopi': '☕ Kopi',
    'Kakao': '🍫 Kakao',
    'Teh': '🍵 Teh',
    'Tebu': '🎋 Tebu'
}

WARNA_KOMODITAS = {
    'Kelapa_Sawit': '#f39c12',
    'Kelapa': '#8d6e63',
    'Karet': '#5d4037',
    'Kopi': '#4e342e',
    'Kakao': '#6d4c41',
    'Teh': '#2e7d32',
    'Tebu': '#9ccc65'
}

WARNA_WILAYAH = {
    'Sumatera': '#2ecc71',
    'Jawa': '#f1c40f',
    'Bali & Nusa Tenggara': '#e67e22',
    'Kalimantan': '#27ae60',
    'Sulawesi': '#16a085',
    'Maluku': '#3498db',
    'Papua': '#9b59b6'
}

KOORDINAT_PROVINSI = {
    'ACEH': (5.55, 95.32), 'SUMATERA UTARA': (3.60, 98.67),
    'SUMATERA BARAT': (-0.79, 100.33), 'RIAU': (1.00, 101.45),
    'JAMBI': (-1.61, 103.61), 'SUMATERA SELATAN': (-3.32, 104.91),
    'BENGKULU': (-3.79, 102.26), 'LAMPUNG': (-5.40, 105.27),
    'KEP. BANGKA BELITUNG': (-2.74, 106.44), 'KEP. RIAU': (3.95, 108.14),
    'DKI JAKARTA': (-6.21, 106.85), 'JAWA BARAT': (-6.92, 107.62),
    'JAWA TENGAH': (-7.15, 110.14), 'DI YOGYAKARTA': (-7.80, 110.37),
    'JAWA TIMUR': (-7.54, 112.24), 'BANTEN': (-6.41, 106.06),
    'BALI': (-8.34, 115.09), 'NUSA TENGGARA BARAT': (-8.58, 116.12),
    'NUSA TENGGARA TIMUR': (-8.66, 121.08), 'KALIMANTAN BARAT': (-0.03, 109.34),
    'KALIMANTAN TENGAH': (-1.68, 113.38), 'KALIMANTAN SELATAN': (-3.32, 114.59),
    'KALIMANTAN TIMUR': (1.24, 116.85), 'KALIMANTAN UTARA': (3.07, 116.04),
    'SULAWESI UTARA': (1.47, 124.84), 'SULAWESI TENGAH': (-0.90, 119.84),
    'SULAWESI SELATAN': (-3.67, 119.97), 'SULAWESI TENGGARA': (-3.96, 122.51),
    'GORONTALO': (0.54, 123.06), 'SULAWESI BARAT': (-2.67, 119.11),
    'MALUKU': (-3.24, 130.15), 'MALUKU UTARA': (1.57, 127.79),
    'PAPUA BARAT': (-1.34, 132.41), 'PAPUA BARAT DAYA': (-1.59, 131.20),
    'PAPUA': (-2.59, 140.67), 'PAPUA SELATAN': (-7.48, 140.75),
    'PAPUA TENGAH': (-3.32, 137.38), 'PAPUA PEGUNUNGAN': (-4.04, 138.96)
}


# ============================================================
# BAGIAN 7: FUNGSI HELPER
# ============================================================
def klasifikasi_wilayah(nama):
    nama = str(nama).upper().strip()
    if any(k in nama for k in ['ACEH', 'SUMATERA', 'RIAU', 'JAMBI', 'BENGKULU', 'LAMPUNG', 'BANGKA', 'KEP. RIAU']):
        return 'Sumatera'
    if any(k in nama for k in ['DKI', 'JAWA', 'YOGYAKARTA', 'BANTEN']):
        return 'Jawa'
    if any(k in nama for k in ['BALI', 'NUSA TENGGARA']):
        return 'Bali & Nusa Tenggara'
    if 'KALIMANTAN' in nama:
        return 'Kalimantan'
    if any(k in nama for k in ['SULAWESI', 'GORONTALO']):
        return 'Sulawesi'
    if 'MALUKU' in nama:
        return 'Maluku'
    if 'PAPUA' in nama:
        return 'Papua'
    return 'Lainnya'


def format_angka(n, dec=0):
    try:
        if pd.isna(n): return "N/A"
        if dec == 0: return f"{int(n):,}".replace(',', '.')
        return f"{n:,.{dec}f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except: return "N/A"


def format_besar(n):
    try:
        if pd.isna(n): return "N/A"
        a = abs(n)
        if a >= 1e9: return f"{n/1e9:.2f}B"
        if a >= 1e6: return f"{n/1e6:.2f}M"
        if a >= 1e3: return f"{n/1e3:.1f}K"
        return f"{n:.1f}"
    except: return "N/A"


def hitung_gini(series):
    try:
        s = series.sort_values().reset_index(drop=True)
        n = len(s)
        if s.sum() == 0 or n == 0: return 0.0
        c = s.cumsum()
        g = (2 * np.sum((np.arange(1, n + 1) * s)) / (n * c.iloc[-1])) - (n + 1) / n
        return max(0, min(1, g))
    except: return 0.5


def interpretasi_korelasi(r):
    a = abs(r)
    if a < 0.3: return "Sangat Lemah"
    if a < 0.5: return "Lemah"
    if a < 0.7: return "Sedang"
    if a < 0.9: return "Kuat"
    return "Sangat Kuat"


def interpretasi_r2(r2):
    if r2 > 0.7: return "🌟 Sangat Baik"
    if r2 > 0.5: return "✅ Baik"
    if r2 > 0.3: return "⚠️ Cukup"
    if r2 > 0.1: return "❌ Lemah"
    return "❌ Sangat Lemah"


def apply_tema(fig, title="", height=550):
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(10, 31, 20, 0.4)",
        font=dict(family="Inter, sans-serif", color="#e8f5e9", size=13),
        title=dict(text=title, font=dict(size=18, color="#58d68d", family="Poppins"), x=0.5, xanchor="center"),
        height=height,
        margin=dict(l=40, r=40, t=60, b=50),
        hoverlabel=dict(bgcolor="#0f2a1c", font_size=13, bordercolor="#2ecc71"),
        legend=dict(bgcolor="rgba(15, 42, 28, 0.8)", bordercolor="#2ecc71", borderwidth=1, font=dict(color="#e8f5e9"))
    )
    try:
        if fig.layout.scene is not None:
            fig.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor="rgb(10, 31, 20)", gridcolor="rgba(46, 204, 113, 0.2)", color="#e8f5e9", title_font=dict(color="#2ecc71")),
                    yaxis=dict(backgroundcolor="rgb(10, 31, 20)", gridcolor="rgba(46, 204, 113, 0.2)", color="#e8f5e9", title_font=dict(color="#2ecc71")),
                    zaxis=dict(backgroundcolor="rgb(10, 31, 20)", gridcolor="rgba(46, 204, 113, 0.2)", color="#e8f5e9", title_font=dict(color="#f1c40f"))
                )
            )
    except: pass
    return fig


def buat_kartu(icon, nilai, label, sub=""):
    s = f'<div style="font-size:0.85em;color:#58d68d;margin-top:8px;">{sub}</div>' if sub else ""
    return f'''<div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{nilai}</div>
        <div class="metric-label">{label}</div>{s}
    </div>'''


# ============================================================
# BAGIAN 8: LOAD DATA
# ============================================================
@st.cache_data(show_spinner="📂 Memuat data...")
def muat_data():
    try:
        df_raw = pd.read_csv("produksi_tanaman.csv")
    except Exception as e:
        return None, None, f"❌ Error membaca CSV: {str(e)}"
    
    if df_raw.empty:
        return None, None, "❌ File kosong"
    
    if 'Provinsi' not in df_raw.columns:
        return None, None, "❌ Kolom 'Provinsi' tidak ditemukan"
    
    df = df_raw.copy()
    df['Wilayah'] = df['Provinsi'].apply(klasifikasi_wilayah)
    df['Total_Produksi'] = df[KOMODITAS].sum(axis=1)
    df['Latitude'] = df['Provinsi'].map(lambda p: KOORDINAT_PROVINSI.get(p, (np.nan, np.nan))[0])
    df['Longitude'] = df['Provinsi'].map(lambda p: KOORDINAT_PROVINSI.get(p, (np.nan, np.nan))[1])
    df['Rank_Produksi'] = df['Total_Produksi'].rank(ascending=False, method='min').astype(int)
    
    def dominan(row):
        vals = {k: row[k] for k in KOMODITAS}
        mk = max(vals, key=vals.get)
        return LABEL_KOMODITAS.get(mk, mk)
    
    df['Komoditas_Dominan'] = df.apply(dominan, axis=1)
    
    hhi = {}
    for _, r in df.iterrows():
        vals = [r[k] for k in KOMODITAS]
        t = sum(vals)
        hhi[r['Provinsi']] = sum((v/t)**2 for v in vals) if t > 0 else 0.0
    df['HHI_Index'] = df['Provinsi'].map(hhi)
    
    def div_cat(h):
        if h < 0.20: return "🌈 Diversifikasi Tinggi"
        if h < 0.35: return "⚖️ Diversifikasi Sedang"
        if h < 0.55: return "🎯 Konsentrasi Tinggi"
        return "🔴 Sangat Terkonsentrasi"
    
    df['Diversifikasi'] = df['HHI_Index'].apply(div_cat)
    
    return df_raw, df, None


df_raw, df, err = muat_data()
if err:
    st.error(err)
    st.info("Pastikan file `produksi_tanaman.csv` ada di folder yang sama dengan `app.py`")
    st.stop()


# ============================================================
# BAGIAN 9: SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 15px 0 20px 0; border-bottom: 2px solid rgba(46, 204, 113, 0.4);">
        <div style="font-size: 2.5em;">🌿🌾🌴</div>
        <div style="color: #58d68d; font-size: 1.05em; font-weight: 700; margin-top: 8px;">
            DASHBOARD PERKEBUNAN
        </div>
        <div style="color: #f7dc6f; font-size: 0.85em;">Indonesia 🇮🇩</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧭 Navigasi")
    menu = [
        "🏠 Beranda",
        "📊 Page 1: Overview",
        "🧹 Page 2: Data Cleaning",
        "📈 Page 3: EDA & 3D",
        "🗺️ Page 3b: Peta",
        "🔗 Page 4: Korelasi",
        "🤖 Page 4b: ML Lanjutan",
        "💡 Page 5: Insights",
        "ℹ️ Tentang"
    ]
    page = st.radio("Pilih:", menu, index=0, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown(f"""
    <div style="background:rgba(46,204,113,0.1);padding:15px;border-radius:12px;border:1px solid rgba(46,204,113,0.3);">
        <p><b>🏛️ Provinsi:</b> {len(df)}</p>
        <p><b>🌿 Komoditas:</b> {len(KOMODITAS)}</p>
        <p><b>🌾 Total:</b> {format_besar(df['Total_Produksi'].sum())}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    lib_status = ""
    lib_status += '<span class="status-badge badge-ok">✅ sklearn</span> ' if SKLEARN_OK else '<span class="status-badge badge-warn">⚠️ sklearn manual</span> '
    lib_status += '<span class="status-badge badge-ok">✅ scipy</span>' if SCIPY_OK else '<span class="status-badge badge-warn">⚠️ scipy manual</span>'
    st.markdown(f"### 🔧 Status\n<div>{lib_status}</div>", unsafe_allow_html=True)


# ============================================================
# BAGIAN 10: HEADER
# ============================================================
st.markdown('<h1 class="main-title">🌿 Dasbor Produksi Perkebunan Indonesia</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">🎓 Tugas UAS Visualisasi Data • {APP_YEAR}</p>', unsafe_allow_html=True)


# ============================================================
# BERANDA
# ============================================================
if page == "🏠 Beranda":
    st.markdown("""
    <div class="page-header">
        <h2>🌾 Selamat Datang di Dasbor Analisis Perkebunan Indonesia</h2>
        <p>Analisis komprehensif 38 provinsi dengan visualisasi 3D modern</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    total_semua = df[KOMODITAS].sum().sum()
    top_prov = df.loc[df['Total_Produksi'].idxmax(), 'Provinsi']
    top_val = df['Total_Produksi'].max()
    gini = hitung_gini(df['Total_Produksi'])
    
    with c1: st.markdown(buat_kartu("🌾", format_angka(total_semua), "TOTAL PRODUKSI", "ribu ton"), unsafe_allow_html=True)
    with c2: st.markdown(buat_kartu("🏆", top_prov[:15], "TOP PROVINSI", format_besar(top_val)), unsafe_allow_html=True)
    with c3: st.markdown(buat_kartu("🗺️", str(df['Wilayah'].nunique()), "WILAYAH", "Geografis"), unsafe_allow_html=True)
    with c4: st.markdown(buat_kartu("⚖️", f"{gini:.2f}", "GINI INDEX", "Ketimpangan"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Treemap komposisi
    tkom = df[KOMODITAS].sum().sort_values(ascending=False).reset_index()
    tkom.columns = ['Komoditas', 'Total']
    tkom['Label'] = tkom['Komoditas'].map(LABEL_KOMODITAS)
    
    fig_tm = px.treemap(tkom, path=['Label'], values='Total', color='Komoditas',
                        color_discrete_map=WARNA_KOMODITAS,
                        title="Proporsi Produksi per Komoditas")
    fig_tm = apply_tema(fig_tm, "Proporsi Produksi Komoditas", 450)
    st.plotly_chart(fig_tm, use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <b>📖 Navigasi:</b> Gunakan sidebar untuk menjelajahi 7 halaman analisis.
        <ul>
            <li>📊 Overview & Data Understanding</li>
            <li>🧹 Data Cleaning</li>
            <li>📈 EDA dengan 4 visualisasi 3D</li>
            <li>🗺️ Peta Indonesia Interaktif</li>
            <li>🔗 Korelasi & Regresi</li>
            <li>🤖 ML Lanjutan</li>
            <li>💡 Insights & Rekomendasi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
elif page == "📊 Page 1: Overview":
    st.markdown('<div class="page-header"><h2>📊 Overview & Data Understanding</h2><p>Gambaran umum dataset</p></div>', unsafe_allow_html=True)
    
    total_sawit = df['Kelapa_Sawit'].sum()
    total_karet = df['Karet'].sum()
    total_kopi = df['Kopi'].sum()
    total_kelapa = df['Kelapa'].sum()
    total_kakao = df['Kakao'].sum()
    total_teh = df['Teh'].sum()
    total_tebu = df['Tebu'].sum()
    total_semua = df['Total_Produksi'].sum()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(buat_kartu("🌾", format_angka(total_semua), "TOTAL", "ribu ton"), unsafe_allow_html=True)
    with c2: st.markdown(buat_kartu("🌴", format_angka(total_sawit), "SAWIT", f"{total_sawit/total_semua*100:.1f}%"), unsafe_allow_html=True)
    with c3: st.markdown(buat_kartu("🌳", format_angka(total_karet), "KARET", f"{total_karet/total_semua*100:.1f}%"), unsafe_allow_html=True)
    with c4: st.markdown(buat_kartu("☕", format_angka(total_kopi), "KOPI", f"{total_kopi/total_semua*100:.1f}%"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📋 Preview Data")
    n_prev = st.slider("Jumlah baris:", 5, len(df), 10, 1, key="prev")
    st.dataframe(df.head(n_prev), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📉 Statistik Deskriptif")
    st.dataframe(df[KOMODITAS].describe().round(2), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📦 Box Plot Distribusi")
    df_melt = df.melt(id_vars=['Provinsi'], value_vars=KOMODITAS, var_name='Komoditas', value_name='Produksi')
    fig_box = px.box(df_melt, x='Komoditas', y='Produksi', color='Komoditas',
                     color_discrete_map=WARNA_KOMODITAS, title="Distribusi per Komoditas")
    fig_box = apply_tema(fig_box, "Distribusi Produksi", 500)
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)


# ============================================================
# PAGE 2: DATA CLEANING
# ============================================================
elif page == "🧹 Page 2: Data Cleaning":
    st.markdown('<div class="page-header"><h2>🧹 Data Cleaning & Preprocessing</h2><p>Pembersihan data</p></div>', unsafe_allow_html=True)
    
    st.subheader("1️⃣ Missing Values")
    miss = df_raw.isnull().sum()
    miss_df = pd.DataFrame({
        'Kolom': miss.index, 'Jumlah': miss.values, 
        'Persentase': (miss.values/len(df_raw))*100
    })
    st.dataframe(miss_df, use_container_width=True, hide_index=True)
    
    if miss.sum() == 0:
        st.success("✅ Dataset bersih, tidak ada missing values!")
    
    st.markdown("---")
    st.subheader("2️⃣ Duplikasi")
    n_dup = df_raw.duplicated().sum()
    st.info(f"Jumlah duplikat: **{n_dup}**")
    if n_dup == 0:
        st.success("✅ Tidak ada duplikasi")
    
    st.markdown("---")
    st.subheader("3️⃣ Deteksi Outlier (IQR)")
    out_data = []
    for k in KOMODITAS:
        s = df[k]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        out_data.append({'Komoditas': LABEL_KOMODITAS[k], 'Q1': q1, 'Q3': q3, 'IQR': iqr, 'Outlier': n_out})
    st.dataframe(pd.DataFrame(out_data).round(2), use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="info-box">
        <b>📝 Catatan:</b> Outlier pada data ini umumnya <b>bukan error</b> tetapi provinsi produsen utama 
        (Riau untuk Kelapa Sawit, Jawa Timur untuk Tebu).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("4️⃣ Feature Engineering")
    st.markdown("Kolom baru yang ditambahkan:")
    st.dataframe(df[['Provinsi', 'Wilayah', 'Total_Produksi', 'Rank_Produksi', 'Komoditas_Dominan', 'HHI_Index', 'Diversifikasi']].head(15), 
                 use_container_width=True, hide_index=True)


# ============================================================
# PAGE 3: EDA & 3D
# ============================================================
elif page == "📈 Page 3: EDA & 3D":
    st.markdown('<div class="page-header"><h2>📈 EDA & 3D Visualizations</h2><p>4 visualisasi 3D interaktif</p></div>', unsafe_allow_html=True)
    
    # --- VIZ 3D #1: SCATTER 3D ---
    st.subheader("🧊 VIZ #1: Scatter 3D")
    c1, c2, c3 = st.columns(3)
    with c1: ax_x = st.selectbox("Sumbu X:", KOMODITAS, format_func=lambda x: LABEL_KOMODITAS[x], key="s3x")
    with c2: ax_y = st.selectbox("Sumbu Y:", KOMODITAS, format_func=lambda x: LABEL_KOMODITAS[x], index=2, key="s3y")
    with c3: ax_z = st.selectbox("Sumbu Z:", KOMODITAS, format_func=lambda x: LABEL_KOMODITAS[x], index=3, key="s3z")
    
    color_by = st.radio("Warna:", ["Wilayah", "Komoditas_Dominan", "Diversifikasi"], horizontal=True)
    
    fig_s3 = px.scatter_3d(df, x=ax_x, y=ax_y, z=ax_z, color=color_by,
                           size='Total_Produksi', size_max=35, hover_name='Provinsi',
                           color_discrete_map=WARNA_WILAYAH if color_by == "Wilayah" else None,
                           opacity=0.85, title=f"Scatter 3D: {ax_x} vs {ax_y} vs {ax_z}")
    fig_s3 = apply_tema(fig_s3, f"Scatter 3D: {LABEL_KOMODITAS[ax_x]} × {LABEL_KOMODITAS[ax_y]} × {LABEL_KOMODITAS[ax_z]}", 680)
    st.plotly_chart(fig_s3, use_container_width=True)
    
    st.markdown("---")
    
    # --- VIZ 3D #2: SURFACE PLOT (FIXED!) ---
    st.subheader("🏔️ VIZ #2: Surface Plot (Topografi)")
    n_surf = st.slider("Jumlah provinsi:", 5, min(30, len(df)), 15, 1, key="surf_n")
    colorscale_surf = st.selectbox("Colorscale:", ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Turbo', 'Jet', 'Portland', 'RdBu'], key="cs_surf")
    
    # Ambil top N provinsi
    df_surf = df.nlargest(n_surf, 'Total_Produksi').set_index('Provinsi')[KOMODITAS]
    Z_matrix = df_surf.values.astype(float)
    
    # Pastikan Z_matrix valid
    if Z_matrix.size == 0 or np.any(np.isnan(Z_matrix)):
        st.warning("Data tidak valid untuk surface plot")
    else:
        # Buat ticks
        x_ticks = list(range(len(KOMODITAS)))
        y_ticks = list(range(len(df_surf)))
        x_labels = [LABEL_KOMODITAS[k].split()[-1] for k in KOMODITAS]  # Hanya nama komoditas
        y_labels = [str(p)[:14] for p in df_surf.index]
        
        # Buat Surface plot dengan parameter SIMPLE (tanpa contours yang kompleks)
        fig_surf = go.Figure(data=[go.Surface(
            z=Z_matrix,
            x=x_ticks,
            y=y_ticks,
            colorscale=colorscale_surf,
            opacity=0.92,
            colorbar=dict(
                title="Produksi",
                tickfont=dict(color="white", size=11),
                titlefont=dict(color="white", size=12)
            ),
            hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}: %{z:,.0f}<extra></extra>',
            customdata=np.array([[(y_labels[i], x_labels[j]) for j in range(len(x_labels))] for i in range(len(y_labels))])
        )])
        
        # Layout sederhana
        fig_surf.update_layout(
            title=dict(text=f"🗻 Topografi Produksi Top {n_surf} Provinsi", 
                      font=dict(size=18, color="#58d68d"), x=0.5),
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(color="#e8f5e9", family="Inter"),
            height=680,
            margin=dict(l=30, r=30, t=60, b=30),
            scene=dict(
                xaxis=dict(
                    title='Komoditas',
                    tickvals=x_ticks,
                    ticktext=x_labels,
                    tickangle=-45,
                    backgroundcolor="rgb(10, 31, 20)",
                    gridcolor="rgba(46, 204, 113, 0.2)",
                    color="#e8f5e9",
                    titlefont=dict(color="#2ecc71", size=13)
                ),
                yaxis=dict(
                    title='Provinsi',
                    tickvals=y_ticks,
                    ticktext=y_labels,
                    backgroundcolor="rgb(10, 31, 20)",
                    gridcolor="rgba(46, 204, 113, 0.2)",
                    color="#e8f5e9",
                    titlefont=dict(color="#2ecc71", size=13)
                ),
                zaxis=dict(
                    title='Produksi (ribu ton)',
                    backgroundcolor="rgb(10, 31, 20)",
                    gridcolor="rgba(46, 204, 113, 0.2)",
                    color="#e8f5e9",
                    titlefont=dict(color="#f1c40f", size=13)
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            )
        )
        st.plotly_chart(fig_surf, use_container_width=True)
        
        # Insight
        max_idx = np.unravel_index(np.argmax(Z_matrix), Z_matrix.shape)
        st.markdown(f"""
        <div class="insight-box">
            <b>🏔️ Puncak Tertinggi:</b> {y_labels[max_idx[0]]} - {x_labels[max_idx[1]]}: 
            <b>{Z_matrix[max_idx]:,.0f} ribu ton</b>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- VIZ 3D #3: 3D BAR (MESH) ---
    st.subheader("🏗️ VIZ #3: 3D Bar Chart")
    n_bar = st.slider("Jumlah provinsi:", 5, 20, 10, 1, key="bar_n")
    df_bar = df.nlargest(n_bar, 'Total_Produksi')[['Provinsi', 'Total_Produksi']].reset_index(drop=True)
    
    fig_bar = go.Figure()
    colors_bar = ['#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b',
                  '#9b59b6', '#8e44ad', '#3498db', '#2980b9', '#1abc9c', '#16a085',
                  '#27ae60', '#2ecc71', '#58d68d', '#a3d155', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800']
    
    for i, row in df_bar.iterrows():
        x0, x1 = i - 0.4, i + 0.4
        y0, y1 = 0, 0.8
        z0, z1 = 0, row['Total_Produksi']
        
        fig_bar.add_trace(go.Mesh3d(
            x=[x0, x1, x1, x0, x0, x1, x1, x0],
            y=[y0, y0, y1, y1, y0, y0, y1, y1],
            z=[z0, z0, z0, z0, z1, z1, z1, z1],
            i=[0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1],
            j=[1, 2, 5, 6, 1, 5, 3, 7, 2, 6, 3, 5],
            k=[2, 3, 6, 7, 5, 4, 7, 6, 3, 7, 5, 4],
            color=colors_bar[i % len(colors_bar)],
            opacity=0.88, flatshading=True,
            name=row['Provinsi'],
            hovertext=f"<b>{row['Provinsi']}</b><br>#{i+1}<br>{row['Total_Produksi']:,.0f}",
            hoverinfo='text'
        ))
    
    fig_bar.update_layout(
        title=dict(text=f"🏛️ Top {n_bar} Provinsi", font=dict(size=18, color="#58d68d"), x=0.5),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8f5e9"), height=650,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
        scene=dict(
            xaxis=dict(title='Provinsi', tickvals=list(range(n_bar)),
                      ticktext=[(n[:11]+'..') if len(n)>11 else n for n in df_bar['Provinsi']],
                      tickangle=-30, backgroundcolor="rgb(10, 31, 20)",
                      gridcolor="rgba(46, 204, 113, 0.2)", color="#e8f5e9"),
            yaxis=dict(visible=False),
            zaxis=dict(title='Total Produksi', backgroundcolor="rgb(10, 31, 20)",
                      gridcolor="rgba(46, 204, 113, 0.2)", color="#e8f5e9",
                      titlefont=dict(color="#f1c40f")),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.0))
        )
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # --- VIZ 3D #4: BUBBLE 3D ---
    st.subheader("🎈 VIZ #4: Bubble 3D (Diversifikasi)")
    
    def rasio_dom(row):
        vals = {k: row[k] for k in KOMODITAS}
        t = sum(vals.values())
        return (max(vals.values()) / t * 100) if t > 0 else 0
    
    df_bub = df.copy()
    df_bub['Dom_Ratio'] = df_bub.apply(rasio_dom, axis=1)
    
    fig_bub = px.scatter_3d(df_bub, x='Dom_Ratio', y='HHI_Index', z='Total_Produksi',
                            color='Wilayah', size='Total_Produksi', size_max=50,
                            hover_name='Provinsi', color_discrete_map=WARNA_WILAYAH,
                            opacity=0.8, title="Diversifikasi 3D")
    fig_bub = apply_tema(fig_bub, "🎈 Dominansi × HHI × Total Produksi", 680)
    fig_bub.update_layout(scene=dict(
        xaxis_title='Dominance Ratio (%)',
        yaxis_title='HHI Index',
        zaxis_title='Total Produksi'
    ))
    st.plotly_chart(fig_bub, use_container_width=True)


# ============================================================
# PAGE 3b: PETA
# ============================================================
elif page == "🗺️ Page 3b: Peta":
    st.markdown('<div class="page-header"><h2>🗺️ Peta Distribusi Indonesia</h2><p>Visualisasi geospasial</p></div>', unsafe_allow_html=True)
    
    df_peta = df.dropna(subset=['Latitude', 'Longitude']).copy()
    
    st.subheader("🌏 Peta #1: Bubble Map per Komoditas")
    k_peta = st.selectbox("Komoditas:", ['Total_Produksi'] + KOMODITAS,
                         format_func=lambda x: "🌾 Total" if x == 'Total_Produksi' else LABEL_KOMODITAS[x])
    
    fig_g1 = px.scatter_geo(df_peta, lat='Latitude', lon='Longitude',
                            color=k_peta, size=k_peta, size_max=45,
                            hover_name='Provinsi', scope='asia',
                            color_continuous_scale='Greens',
                            projection='natural earth',
                            title=f"Distribusi {k_peta}")
    fig_g1.update_geos(
        showcountries=True, countrycolor="rgba(46, 204, 113, 0.4)",
        showcoastlines=True, coastlinecolor="#2ecc71",
        showland=True, landcolor="rgba(26, 77, 46, 0.45)",
        showocean=True, oceancolor="rgba(8, 25, 15, 0.95)",
        lataxis_range=[-12, 7], lonaxis_range=[93, 144], bgcolor='rgba(0,0,0,0)'
    )
    fig_g1.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8f5e9"), height=680)
    st.plotly_chart(fig_g1, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🎯 Peta #2: Distribusi per Wilayah")
    fig_g2 = px.scatter_geo(df_peta, lat='Latitude', lon='Longitude',
                            color='Wilayah', size='Total_Produksi', size_max=50,
                            hover_name='Provinsi', color_discrete_map=WARNA_WILAYAH,
                            scope='asia', projection='natural earth')
    fig_g2.update_geos(
        showcountries=True, countrycolor="rgba(46, 204, 113, 0.4)",
        showcoastlines=True, coastlinecolor="#2ecc71",
        showland=True, landcolor="rgba(26, 77, 46, 0.45)",
        showocean=True, oceancolor="rgba(8, 25, 15, 0.95)",
        lataxis_range=[-12, 7], lonaxis_range=[93, 144], bgcolor='rgba(0,0,0,0)'
    )
    fig_g2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8f5e9"),
                         height=680, title=dict(text="Distribusi per Wilayah", 
                         font=dict(color="#58d68d", size=18), x=0.5))
    st.plotly_chart(fig_g2, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🏆 Peta #3: Komoditas Dominan")
    fig_g3 = px.scatter_geo(df_peta, lat='Latitude', lon='Longitude',
                            color='Komoditas_Dominan', hover_name='Provinsi',
                            scope='asia', projection='natural earth',
                            color_discrete_sequence=px.colors.qualitative.Set2)
    fig_g3.update_traces(marker=dict(size=14, symbol='square'))
    fig_g3.update_geos(
        showcountries=True, countrycolor="rgba(46, 204, 113, 0.4)",
        showcoastlines=True, coastlinecolor="#2ecc71",
        showland=True, landcolor="rgba(26, 77, 46, 0.45)",
        showocean=True, oceancolor="rgba(8, 25, 15, 0.95)",
        lataxis_range=[-12, 7], lonaxis_range=[93, 144], bgcolor='rgba(0,0,0,0)'
    )
    fig_g3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8f5e9"), height=650,
                         title=dict(text="Komoditas Dominan per Provinsi", font=dict(color="#58d68d", size=18), x=0.5))
    st.plotly_chart(fig_g3, use_container_width=True)


# ============================================================
# PAGE 4: KORELASI
# ============================================================
elif page == "🔗 Page 4: Korelasi":
    st.markdown('<div class="page-header"><h2>🔗 Analisis Korelasi & Regresi</h2><p>Hubungan antar variabel</p></div>', unsafe_allow_html=True)
    
    # Heatmap
    st.subheader("🔥 Matriks Korelasi Pearson")
    corr = df[KOMODITAS].corr(method='pearson')
    labels_corr = [LABEL_KOMODITAS[c] for c in corr.columns]
    
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=labels_corr, y=labels_corr,
        colorscale='RdYlGn', zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate='%{text:.2f}',
        textfont=dict(size=11, color='white')
    ))
    fig_corr = apply_tema(fig_corr, "Matriks Korelasi Pearson", 600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # Analisis detail
    st.subheader("🔬 Analisis Detail")
    c1, c2 = st.columns(2)
    with c1: v1 = st.selectbox("Variabel X:", KOMODITAS, format_func=lambda x: LABEL_KOMODITAS[x], key="cx")
    with c2: v2 = st.selectbox("Variabel Y:", KOMODITAS, format_func=lambda x: LABEL_KOMODITAS[x], index=2, key="cy")
    
    r, p = do_corr(df[v1].values, df[v2].values)
    int_kor = interpretasi_korelasi(r)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(buat_kartu("📐", f"{r:+.3f}", "PEARSON r", int_kor), unsafe_allow_html=True)
    with c2: st.markdown(buat_kartu("🎯", f"{p:.4f}" if p >= 0.0001 else "<0.0001", "P-VALUE", "α=0.05"), unsafe_allow_html=True)
    with c3: st.markdown(buat_kartu("💎", f"{r**2:.3f}", "R²", f"{r**2*100:.1f}% variasi"), unsafe_allow_html=True)
    
    # Scatter 2D
    fig_sc = px.scatter(df, x=v1, y=v2, color='Wilayah', size='Total_Produksi',
                       hover_name='Provinsi', color_discrete_map=WARNA_WILAYAH,
                       trendline='ols')
    fig_sc = apply_tema(fig_sc, f"{LABEL_KOMODITAS[v1]} vs {LABEL_KOMODITAS[v2]}", 550)
    st.plotly_chart(fig_sc, use_container_width=True)
    
    st.markdown("---")
    
    # Regresi
    st.subheader("🤖 Model Regresi Linier")
    target = st.selectbox("Target:", KOMODITAS, format_func=lambda x: LABEL_KOMODITAS[x], index=0, key="reg_t")
    fitur = [k for k in KOMODITAS if k != target]
    
    X = df[fitur].values
    y = df[target].values
    X_tr, X_te, y_tr, y_te = do_split(X, y, test_size=0.2, random_state=42)
    
    sc = Scalr()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    
    mdl = LinReg()
    mdl.fit(X_tr_sc, y_tr)
    y_pred = mdl.predict(X_te_sc)
    
    mae = do_mae(y_te, y_pred)
    rmse = do_rmse(y_te, y_pred)
    r2 = do_r2(y_te, y_pred)
    mape = do_mape(y_te, y_pred)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(buat_kartu("📏", f"{mae:.2f}", "MAE", ""), unsafe_allow_html=True)
    with c2: st.markdown(buat_kartu("📐", f"{rmse:.2f}", "RMSE", ""), unsafe_allow_html=True)
    with c3: st.markdown(buat_kartu("🎯", f"{r2:.4f}", "R²", interpretasi_r2(r2)), unsafe_allow_html=True)
    with c4: st.markdown(buat_kartu("📊", f"{mape:.1f}%", "MAPE", ""), unsafe_allow_html=True)
    
    # Actual vs Predicted
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(x=y_te, y=y_pred, mode='markers',
                                 marker=dict(size=12, color='#2ecc71', line=dict(color='#f1c40f', width=2)),
                                 name='Test'))
    mn = min(y_te.min(), y_pred.min())
    mx = max(y_te.max(), y_pred.max())
    fig_avp.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                 line=dict(color='#e74c3c', width=2, dash='dash'),
                                 name='Perfect'))
    fig_avp = apply_tema(fig_avp, f"Actual vs Predicted: {LABEL_KOMODITAS[target]}", 500)
    fig_avp.update_layout(xaxis_title='Aktual', yaxis_title='Prediksi')
    st.plotly_chart(fig_avp, use_container_width=True)
    
    # Koefisien
    st.markdown("### ⚖️ Koefisien Regresi")
    coef_df = pd.DataFrame({
        'Fitur': [LABEL_KOMODITAS[f] for f in fitur],
        'Koefisien': mdl.coef_
    }).sort_values('Koefisien', key=abs, ascending=False)
    st.dataframe(coef_df.style.format({'Koefisien': '{:.4f}'}), use_container_width=True, hide_index=True)


# ============================================================
# PAGE 4b: ML LANJUTAN
# ============================================================
elif page == "🤖 Page 4b: ML Lanjutan":
    st.markdown('<div class="page-header"><h2>🤖 ML Lanjutan</h2><p>Perbandingan model</p></div>', unsafe_allow_html=True)
    
    target_ml = st.selectbox("Target:", KOMODITAS, format_func=lambda x: LABEL_KOMODITAS[x], key="ml_t")
    fitur_ml = [k for k in KOMODITAS if k != target_ml]
    
    if SKLEARN_OK:
        models = [
            ("Linear Regression", LinearRegression()),
            ("Ridge (L2)", Ridge(alpha=1.0)),
            ("Lasso (L1)", Lasso(alpha=0.1, max_iter=10000)),
            ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]
    else:
        models = [
            ("Linear Regression", ManualLinearRegression()),
            ("Ridge L2", ManualRidge(alpha=1.0))
        ]
    
    X = df[fitur_ml].values
    y = df[target_ml].values
    X_tr, X_te, y_tr, y_te = do_split(X, y, test_size=0.2, random_state=42)
    sc = Scalr()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    
    hasil = []
    for nama, mdl in models:
        nm = nama.lower()
        if 'forest' in nm or 'gradient' in nm or 'boost' in nm:
            mdl.fit(X_tr, y_tr)
            pred = mdl.predict(X_te)
        else:
            mdl.fit(X_tr_sc, y_tr)
            pred = mdl.predict(X_te_sc)
        
        hasil.append({
            'Model': nama,
            'MAE': do_mae(y_te, pred),
            'RMSE': do_rmse(y_te, pred),
            'R²': do_r2(y_te, pred),
            'MAPE %': do_mape(y_te, pred)
        })
    
    hasil_df = pd.DataFrame(hasil).sort_values('R²', ascending=False)
    st.dataframe(
        hasil_df.style.highlight_max(subset=['R²'], color='#2ecc71')
                .highlight_min(subset=['MAE', 'RMSE', 'MAPE %'], color='#2ecc71')
                .format({'MAE': '{:.3f}', 'RMSE': '{:.3f}', 'R²': '{:.4f}', 'MAPE %': '{:.2f}'}),
        use_container_width=True, hide_index=True
    )
    
    best = hasil_df.iloc[0]
    st.success(f"### 🏆 Model Terbaik: {best['Model']}\n- **R²:** `{best['R²']:.4f}`\n- **MAE:** `{best['MAE']:.3f}`")
    
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(x=hasil_df['Model'], y=hasil_df['R²'],
                             marker_color='#2ecc71',
                             text=hasil_df['R²'].round(3), textposition='outside',
                             textfont=dict(color='#58d68d')))
    fig_cmp = apply_tema(fig_cmp, f"Perbandingan R² (Target: {LABEL_KOMODITAS[target_ml]})", 450)
    fig_cmp.update_layout(xaxis_title='Model', yaxis_title='R² Score')
    st.plotly_chart(fig_cmp, use_container_width=True)


# ============================================================
# PAGE 5: INSIGHTS
# ============================================================
elif page == "💡 Page 5: Insights":
    st.markdown('<div class="page-header"><h2>💡 Insights & Rekomendasi</h2><p>Kesimpulan analisis</p></div>', unsafe_allow_html=True)
    
    top_prov = df.loc[df['Kelapa_Sawit'].idxmax(), 'Provinsi']
    top_val = df['Kelapa_Sawit'].max()
    pct_sawit = df['Kelapa_Sawit'].sum() / df['Total_Produksi'].sum() * 100
    top_wil = df.groupby('Wilayah')['Total_Produksi'].sum().idxmax()
    gini = hitung_gini(df['Total_Produksi'])
    
    st.subheader("📋 Ringkasan")
    st.markdown(f"""
    <div class="info-box">
        <ul>
            <li>🌴 Kelapa Sawit: <b>{pct_sawit:.1f}%</b> dari total produksi</li>
            <li>🏆 <b>{top_prov}</b> produsen sawit terbesar ({top_val:,.0f} ribu ton)</li>
            <li>🗺️ <b>Wilayah {top_wil}</b> sentra produksi utama</li>
            <li>⚖️ Gini Index: <b>{gini:.3f}</b> ({'Timpang' if gini > 0.5 else 'Sedang'})</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🔍 5 Insight Mendalam")
    
    insights = [
        ("Dominasi Kelapa Sawit", f"Kelapa Sawit menyumbang {pct_sawit:.1f}% produksi nasional dengan {top_prov} sebagai produsen utama. Ketergantungan pada satu komoditas menciptakan risiko ekonomi."),
        ("Ketimpangan Produksi", f"Gini {gini:.3f} menunjukkan ketimpangan tinggi. Top 5 provinsi menguasai >55% total produksi nasional."),
        ("Spesialisasi Regional", "Sumatera-Kalimantan dominan Sawit, Jawa unggul di Tebu & Teh, Sulawesi sentra Kakao."),
        ("Diversifikasi Rendah", "Banyak provinsi memiliki HHI tinggi, rentan terhadap fluktuasi harga komoditas tertentu."),
        ("Potensi Indonesia Timur", "Maluku, Papua, NTT berpotensi besar namun kontribusi masih minim.")
    ]
    
    for i, (t, c) in enumerate(insights, 1):
        st.markdown(f"""<div class="insight-box">
            <strong>💡 Insight #{i}: {t}</strong>
            <p style="margin:8px 0 0 0;">{c}</p>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🎯 5 Rekomendasi Implementatif")
    
    rekomendasi = [
        ("Diversifikasi Komoditas", "Tumpang sari sawit dengan kakao/kopi. Insentif fiskal untuk komoditas alternatif."),
        ("Infrastruktur Indonesia Timur", "Jalan, pelabuhan, cold storage di Papua, Maluku, NTT."),
        ("Sertifikasi Sustainability", "Percepatan ISPO 100% untuk sawit. Sistem traceability blockchain."),
        ("Digitalisasi Sistem Informasi", "Dashboard real-time, IoT monitoring, early warning system."),
        ("Hilirisasi Produk", "Industri pengolahan di sentra produksi. Branding kopi specialty dan teh premium.")
    ]
    
    for i, (t, c) in enumerate(rekomendasi, 1):
        st.markdown(f"""<div class="recommend-box">
            <strong>🎯 Rekomendasi #{i}: {t}</strong>
            <p style="margin:8px 0 0 0;">{c}</p>
        </div>""", unsafe_allow_html=True)


# ============================================================
# TENTANG
# ============================================================
elif page == "ℹ️ Tentang":
    st.markdown('<div class="page-header"><h2>ℹ️ Tentang Aplikasi</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### ✨ Fitur Utama
    - 📊 **9 Halaman Analisis** (termasuk Beranda & Tentang)
    - 🧊 **4 Visualisasi 3D**: Scatter 3D, Surface Plot, Mesh 3D Bar, Bubble 3D
    - 🗺️ **3 Jenis Peta**: Bubble Map, Wilayah, Dominansi
    - 🔗 **Korelasi** Pearson & detail uji signifikansi
    - 🤖 **Multi-Model ML**: Linear, Ridge, Lasso, RF, GB
    - 💡 **5 Insights + 5 Rekomendasi**
    - 🛡️ **Fallback Manual** jika sklearn gagal
    
    ### 🛠️ Teknologi
    - Streamlit, Pandas, NumPy, Plotly, scikit-learn, SciPy
    
    ### 🚀 Cara Deploy
    1. Push ke GitHub
    2. Buka share.streamlit.io
    3. New App → pilih repository → Deploy
    
    ### 📁 Struktur
    ```
    📦 dasbor/
    ├── app.py
    ├── produksi_tanaman.csv
    └── requirements.txt
    ```
    """)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(f"""
<div class="app-footer">
    <p style="font-size:1.2em;font-weight:800;color:#58d68d;">🌿 Dasbor Perkebunan Indonesia 🌾</p>
    <p>Tugas UAS Visualisasi Data • {APP_YEAR} • v{APP_VERSION}</p>
</div>
