"""
UTS Business Intelligence — Analisis Transaksi Toko
Flow: Login → Cover (Identitas) → Menu Tahapan 1–6
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, networkx as nx
import os  # Tambahan untuk mengecek keberadaan file foto

from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(page_title="UTS BI — Analisis Transaksi Toko",
                   page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# =========================================================
# 🔧 KONFIGURASI — SILAKAN EDIT DI SINI
# =========================================================
# Kredensial login (Hanya 1 akun)
LOGIN_CREDENTIALS = {
    "admin": "admin123"
}

# Nama file foto profil (pastikan file ada di folder yang sama dengan UTS.py)
FOTO_PROFIL = "foto.png"

# ✅ Nama sistem singkat (tampil di setiap header menu)
SYSTEM_NAME = "SIBI-Toko"
SYSTEM_DESC = "Sistem Business Intelligence Analisis Transaksi Toko"

# Informasi identitas mahasiswa (TAMPIL DI COVER)
STUDENT_INFO = {
    "Nama Lengkap":     "LUVI DA CONCEICAO SILVA",
    "NPM":              "51230061",
    "Program Studi":    "Teknologi Informasi",
    "Universitas":      "Universitas Timor",
    "Mata Kuliah":      "Business Intelligence",
    "Dosen Pengampu":   "Krisantus Jumarto Tey Seran, S.T., M.T.",
    "Judul UTS":        "Integrasi Data Mining & Business Intelligence pada Analisis Transaksi Toko",
}

# =========================================================
# FUNGSI FORMAT ANGKA (TITIK UNTUK RIBUAN)
# =========================================================
def format_angka(val, desimal=0):
    """Mengubah 1000 -> 1.000"""
    return f"{val:,.{desimal}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def format_rupiah(val):
    """Mengubah 1500000 -> Rp 1.500.000"""
    return f"Rp {val:,.0f}".replace(",", ".")

# =========================================================
# CSS GLOBAL
# =========================================================
st.markdown("""
<style>
.login-card {
    background: linear-gradient(135deg, #1f77b4 0%, #6a4c93 100%);
    padding: 30px; border-radius: 15px; color: white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}
.cover-hero {
    background: linear-gradient(135deg, #0d47a1 0%, #6a1b9a 100%);
    padding: 40px; border-radius: 15px; color: white;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}
.cover-hero h1 { margin: 0; font-size: 32px; }
.cover-hero p { margin-top: 8px; font-size: 16px; opacity: 0.9; }
.id-card {
    background: #ffffff; border-left: 6px solid #1f77b4;
    padding: 18px; border-radius: 10px; margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.id-card b { color: #1f77b4; display:inline-block; width: 160px; }
.feature-box {
    background: #f8f9fa; border-radius: 12px; padding: 20px;
    border-top: 4px solid #1f77b4; height: 100%;
}
.feature-box h4 { color: #1f77b4; margin-top: 0; }

/* ✅ Banner identitas ringkas di atas setiap menu */
.identity-banner {
    background: linear-gradient(135deg, #0d47a1 0%, #6a1b9a 100%);
    padding: 14px 20px; border-radius: 12px; color: white;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    margin-bottom: 8px;
}
.identity-banner .row1 {
    font-size: 15px; font-weight: 600;
}
.identity-banner .row2 {
    font-size: 13px; opacity: 0.88; margin-top: 4px;
}
.identity-photo {
    border-radius: 50%;
    overflow: hidden;
    border: 3px solid #ffffff;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE INIT
# =========================================================
for k, v in {"logged_in": False, "show_cover": False, "username": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# ✅ FUNGSI HEADER IDENTITAS RINGKAS (dipakai di semua menu 1–7)
# =========================================================
def render_identity_header(df):
    """Menampilkan foto + identitas singkat + nama sistem di atas setiap menu."""
    col_photo, col_info = st.columns([1, 8])
    with col_photo:
        if os.path.exists(FOTO_PROFIL):
            st.markdown('<div class="identity-photo">', unsafe_allow_html=True)
            st.image(FOTO_PROFIL, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("👤")
    with col_info:
        # Mengambil info tanggal dan jumlah baris dari dataset
        min_date = df['Transaction_Date'].min().date()
        max_date = df['Transaction_Date'].max().date()
        total_rows = format_angka(len(df))
        
        st.markdown(f"""
        <div class="identity-banner">
            <div class="row1">
                👨‍🎓 {STUDENT_INFO['Nama Lengkap']}
                <span style="opacity:0.9;">({STUDENT_INFO['NPM']})</span>
                &nbsp;·&nbsp; {STUDENT_INFO['Program Studi']}
                &nbsp;·&nbsp; {STUDENT_INFO['Universitas']}
            </div>
            <div class="row2" style="display: flex; justify-content: space-between;">
                <span>📊 <b>{SYSTEM_NAME}</b> — {SYSTEM_DESC}</span>
                <span style="font-size: 12px;">📅 {min_date} s/d {max_date} &nbsp;|&nbsp; {total_rows} baris</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("")

# =========================================================
# HALAMAN 1 — LOGIN
# =========================================================
def page_login():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="login-card">
            <h2 style="margin:0;">🔐 Login Aplikasi</h2>
            <p style="margin-top:8px;">UTS Business Intelligence — Analisis Transaksi Toko</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("👤 Username", placeholder="admin")
            password = st.text_input("🔑 Password", type="password", placeholder="•••••••")
            submit = st.form_submit_button("🚀 Masuk", use_container_width=True)

            if submit:
                if username in LOGIN_CREDENTIALS and LOGIN_CREDENTIALS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login berhasil! Memuat...")
                    st.rerun()
                else:
                    st.error("❌ Username atau password salah.")

        st.caption("💡 **Akun**: `admin / admin123`")

# =========================================================
# HALAMAN 2 — COVER (IDENTITAS + SISTEM)
# =========================================================
def page_cover():
    st.markdown(f"""
    <div class="cover-hero">
        <h1>📊 UTS Business Intelligence</h1>
        <p><b>{STUDENT_INFO['Judul UTS']}</b></p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("👨‍🎓 Identitas Mahasiswa")
        if os.path.exists(FOTO_PROFIL):
            st.image(FOTO_PROFIL, width=180, caption="Foto Profil")
        else:
            st.warning(f"⚠️ File foto `{FOTO_PROFIL}` tidak ditemukan. Pastikan file berada di folder yang sama dengan `UTS.py`.")

        st.markdown("<br>", unsafe_allow_html=True)

        for k, v in STUDENT_INFO.items():
            st.markdown(
                f'<div class="id-card"><b>{k}</b> : {v}</div>',
                unsafe_allow_html=True)

    with col2:
        st.subheader("ℹ️ Tentang Sistem")
        st.markdown(f"""
        Aplikasi ini adalah **platform analitik end-to-end** yang mengintegrasikan
        teknik *Data Mining* dan *Business Intelligence* untuk mendukung
        pengambilan keputusan berbasis data pada studi kasus **transaksi toko retail**.

        Sistem memandu pengguna menjalankan alur kerja analisis lengkap —
        mulai dari pemahaman bisnis, eksplorasi data, pembersihan data,
        penemuan pola asosiasi produk, segmentasi pelanggan, hingga
        penyajian dashboard dan rekomendasi strategis.

        **Nama Sistem:** `{SYSTEM_NAME}` — {SYSTEM_DESC}
        """)

        st.markdown("#### 🎯 Metode yang Digunakan")
        st.markdown("""
        - **Apriori / Market Basket Analysis** → menemukan produk yang sering dibeli bersamaan.
        - **K-Means Clustering** → mengelompokkan pelanggan berdasarkan RFM.
        - **RFM Analysis** → Recency, Frequency, Monetary sebagai fitur perilaku pelanggan.
        - **Business Intelligence Dashboard** → KPI, tren, dan visualisasi interaktif.
        """)

    st.markdown("---")
    st.subheader("🧩 Modul / Tahapan Sistem")
    cols = st.columns(3)
    features = [
        ("1️⃣ Business Understanding", "Merumuskan pertanyaan bisnis & ringkasan dataset."),
        ("2️⃣ EDA", "Audit kualitas data, visualisasi awal, insight awal."),
        ("3️⃣ Preprocessing", "Pembersihan data, basket_size, RFM, standardisasi."),
        ("4️⃣ Apriori", "Frequent itemsets & association rules + rekomendasi bundling."),
        ("5️⃣ K-Means", "Segmentasi pelanggan RFM + strategi pemasaran per segmen."),
        ("6️⃣ Dashboard & Rekomendasi", "KPI interaktif, tren, dan rekomendasi keputusan."),
    ]
    for i, (title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-box">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    if c1.button("▶️ Mulai Analisis", use_container_width=True, type="primary"):
        st.session_state.show_cover = True
        st.rerun()
    if c2.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.show_cover = False
        st.session_state.username = ""
        st.rerun()

# =========================================================
# GERBANG AKSES
# =========================================================
if not st.session_state.logged_in:
    page_login()
    st.stop()

if not st.session_state.show_cover:
    page_cover()
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data(show_spinner="Memuat Data_Raw...")
def load_raw(b):
    return pd.read_excel(io.BytesIO(b), sheet_name="Data_Raw")

# =========================================================
# SIDEBAR — UPLOAD & NAVIGASI
# =========================================================
st.sidebar.title("🎯 UTS Business Intelligence")
st.sidebar.caption(f"Sistem: **{SYSTEM_NAME}**")
st.sidebar.caption(f"Login sebagai: **{st.session_state.username}**")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.show_cover = False
    st.session_state.username = ""
    st.rerun()
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload `P8. UTS_Business Intelligence.xlsx`",
    type=["xlsx", "xls"])

st.sidebar.markdown("---")
menu = st.sidebar.radio("🧭 Navigasi Tahapan:", [
    "1. Business Understanding",
    "2. EDA (Exploratory Data Analysis)",
    "3. Preprocessing Data",
    "4. Apriori (Market Basket)",
    "5. K-Means (Segmentasi Pelanggan)",
    "6. Dashboard BI",
    "7. Rekomendasi Keputusan"
])

# =========================================================
# LOAD DATASET
# =========================================================
if uploaded_file is not None:
    df_raw = load_raw(uploaded_file.read())
    st.sidebar.success(f"✅ {format_angka(len(df_raw))} baris dimuat")
else:
    try:
        with open("P8. UTS_Business Intelligence.xlsx", "rb") as f:
            df_raw = load_raw(f.read())
        st.sidebar.info("📂 Menggunakan file default di folder kerja")
    except FileNotFoundError:
        st.error("⚠️ Silakan upload file Excel di sidebar untuk memulai.")
        st.stop()

# =========================================================
# PREPROCESSING
# =========================================================
@st.cache_data(show_spinner="Membersihkan data...")
def preprocess(df_in):
    df = df_in.copy()
    log = {"Baris awal": len(df)}

    n0 = len(df)
    df = df.drop_duplicates()
    log["Duplikat dihapus"] = n0 - len(df)

    for c in ["Region", "Customer_Type", "Product_Name", "Category", "Payment_Method"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.title()
            df[c] = df[c].replace({"Nan": np.nan, "None": np.nan, "": np.nan})

    if "Category" in df.columns:
        cat_map = {"Sembako":"Sembako","Minuman":"Minuman","Makanan":"Makanan",
                   "Perawatan":"Perawatan","Bayi":"Bayi","Elektronik":"Elektronik",
                   "Rumah Tangga":"Rumah Tangga","Atk":"ATK"}
        df["Category"] = df["Category"].map(lambda x: cat_map.get(x, x) if pd.notna(x) else x)

    if "Payment_Method" in df.columns:
        df["Payment_Method"] = df["Payment_Method"].map(
            lambda x: "QRIS" if pd.notna(x) and str(x).upper() == "QRIS" else x)

    for c in ["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Quantity" in df.columns:
        bad_qty = (df["Quantity"] <= 0) | (df["Quantity"] > 100) | df["Quantity"].isna()
        log["Qty tidak valid dibuang"] = int(bad_qty.sum())
        df = df[~bad_qty]

    if "Unit_Price" in df.columns:
        bad_price = (df["Unit_Price"] <= 0) | (df["Unit_Price"] > 200000) | df["Unit_Price"].isna()
        log["Harga tidak valid dibuang"] = int(bad_price.sum())
        df = df[~bad_price]

    if all(c in df.columns for c in ["Quantity","Unit_Price","Discount_Rate","Line_Total"]):
        df["Discount_Rate"] = df["Discount_Rate"].fillna(0)
        expected = df["Quantity"] * df["Unit_Price"] * (1 - df["Discount_Rate"])
        mismatch = (df["Line_Total"].isna()
                    | (df["Line_Total"] > expected*1.5 + 500)
                    | (df["Line_Total"] < expected*0.5 - 500))
        log["Line_Total diperbaiki"] = int(mismatch.sum())
        df.loc[mismatch, "Line_Total"] = expected[mismatch]

    critical = [c for c in ["Transaction_ID","Product_Name","Quantity","Unit_Price","Line_Total"]
                if c in df.columns]
    n0 = len(df); df = df.dropna(subset=critical)
    log["Field kritikal kosong dibuang"] = n0 - len(df)

    if "Customer_ID" in df.columns:
        df["Customer_ID"] = df["Customer_ID"].fillna("GUEST")
        log["Customer_ID → GUEST"] = int((df["Customer_ID"]=="GUEST").sum())
    if "Customer_Type" in df.columns:
        df["Customer_Type"] = df["Customer_Type"].fillna("Non-Member")
    if "Region" in df.columns:
        df["Region"] = df["Region"].fillna("Unknown")
    if "Payment_Method" in df.columns:
        df["Payment_Method"] = df["Payment_Method"].fillna("Unknown")
    if "Category" in df.columns:
        df["Category"] = df["Category"].fillna("Unknown")
    if "Product_ID" in df.columns:
        mode_pid = (df.groupby("Product_Name")["Product_ID"]
                      .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "P000"))
        df["Product_ID"] = df.apply(
            lambda r: mode_pid.get(r["Product_Name"],"P000")
                      if pd.isna(r["Product_ID"]) else r["Product_ID"], axis=1)

    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    n0 = len(df); df = df.dropna(subset=["Transaction_Date"])
    log["Tanggal invalid dibuang"] = n0 - len(df)

    basket_size = (df.groupby("Transaction_ID")["Product_Name"]
                     .nunique().reset_index(name="Basket_Size"))
    df = df.merge(basket_size, on="Transaction_ID", how="left")

    ref_date = df["Transaction_Date"].max() + pd.Timedelta(days=1)
    rfm = (df[df["Customer_ID"]!="GUEST"]
           .groupby("Customer_ID")
           .agg(Recency=("Transaction_Date", lambda s: (ref_date - s.max()).days),
                Frequency=("Transaction_ID","nunique"),
                Monetary=("Line_Total","sum"))
           .reset_index())
    log["Tanggal acuan RFM"] = str(ref_date.date())

    log["Baris akhir"] = len(df)
    return df, rfm, log, ref_date

df_clean, rfm, prep_log, ref_date = preprocess(df_raw)

# =========================================================
# TAHAP 1 — BUSINESS UNDERSTANDING
# =========================================================
if menu.startswith("1."):
    render_identity_header(df_clean)
    st.header("1️⃣ Business Understanding")

    st.subheader("🏪 Konteks Bisnis")
    st.write("""
    Toko retail memiliki data transaksi tahun 2025. Manajemen ingin memahami
    pola pembelian pelanggan untuk **meningkatkan penjualan** melalui
    *cross-selling*, penempatan produk, dan retensi pelanggan.
    """)

    st.subheader("❓ Pertanyaan Bisnis (min. 3 — Tugas 1)")
    st.markdown("""
    1. **Apriori**: Produk apa yang sering dibeli bersamaan? (bundle/rak)
    2. **Segmen Pelanggan**: Segmen mana yang paling bernilai (Monetary tinggi
       & Frequency tinggi) untuk program loyalitas?
    3. **Pola Waktu**: Kapan penjualan memuncak (musim/bulan) untuk perencanaan stok?
    4. **Kategori**: Kategori mana yang paling berkontribusi terhadap total revenue?
    5. **Risiko Churn**: Berapa pelanggan dengan Recency tinggi (tidak aktif)?
    """)

    st.subheader("📋 Ringkasan Dataset (Tugas 2)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baris", format_angka(len(df_clean)))
    col2.metric("Kolom", df_clean.shape[1])
    col3.metric("Transaksi unik", format_angka(df_clean['Transaction_ID'].nunique()))
    col4.metric("Pelanggan unik", format_angka(df_clean['Customer_ID'].nunique()))

    col1b, col2b, col3b, col4b = st.columns(4)
    col1b.metric("Produk unik", format_angka(df_clean['Product_Name'].nunique()))
    col2b.metric("Total Penjualan", format_rupiah(df_clean['Line_Total'].sum()))
    col3b.metric("Rata-rata Basket", format_angka(df_clean['Basket_Size'].mean(), 2) + " produk")
    col4b.metric("Rentang Tanggal",
                 f"{df_clean['Transaction_Date'].min().date()} → {df_clean['Transaction_Date'].max().date()}")

    st.subheader("📐 Tipe Data & Struktur")
    info = pd.DataFrame({
        "Kolom": df_clean.columns,
        "Tipe": df_clean.dtypes.astype(str).values,
        "Non-Null": df_clean.notna().sum().values,
        "%Null": (df_clean.isna().mean()*100).round(2).values
    })
    st.dataframe(info, use_container_width=True)

    st.subheader("🔎 Sample Data_Raw")
    st.dataframe(df_raw.head(20), use_container_width=True)

# =========================================================
# TAHAP 2 — EDA
# =========================================================
elif menu.startswith("2."):
    render_identity_header(df_clean)
    st.header("2️⃣ EDA — Exploratory Data Analysis (Tugas 3)")

    st.subheader("🧹 Audit Kualitas Data")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duplikasi eksak", format_angka(df_raw.duplicated().sum()))
    c2.metric("Missing (sel total)", format_angka(df_raw.isna().sum().sum()))
    c3.metric("Qty tidak wajar", format_angka(((pd.to_numeric(df_raw['Quantity'],errors='coerce')<=0) |
                                     (pd.to_numeric(df_raw['Quantity'],errors='coerce')>100)).sum()))
    c4.metric("Harga tidak wajar", format_angka(((pd.to_numeric(df_raw['Unit_Price'],errors='coerce')<=0) |
                                       (pd.to_numeric(df_raw['Unit_Price'],errors='coerce')>200000)).sum()))

    with st.expander("Lihat detail missing & inkonsistensi"):
        st.write("**Missing per kolom (Data_Raw):**")
        st.dataframe(df_raw.isna().sum().to_frame("Missing").T, use_container_width=True)
        st.write("**Inkonsistensi teks (contoh):**")
        st.write("Kategori:", sorted(df_raw['Category'].astype(str).str.strip().unique().tolist()))
        st.write("Payment :", sorted(df_raw['Payment_Method'].astype(str).str.strip().unique().tolist()))

    st.subheader("📊 Visualisasi 1 — Revenue per Kategori")
    cat_rev = (df_clean.groupby("Category")["Line_Total"].sum()
                       .sort_values(ascending=False).reset_index())
    fig = px.bar(cat_rev, x="Category", y="Line_Total",
                 color="Line_Total", color_continuous_scale="Blues",
                 title="Total Revenue per Kategori")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Visualisasi 2 — Tren Penjualan Bulanan")
    month = (df_clean.assign(Bulan=df_clean["Transaction_Date"].dt.to_period("M").astype(str))
                      .groupby("Bulan")["Line_Total"].sum().reset_index())
    fig2 = px.line(month, x="Bulan", y="Line_Total", markers=True,
                   title="Total Penjualan per Bulan (2025)")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("💡 Insight Awal (min. 3)")
    top_cat = cat_rev.iloc[0]
    peak_month = month.sort_values("Line_Total", ascending=False).iloc[0]
    avg_basket = df_clean["Basket_Size"].mean()
    st.markdown(f"""
    1. Kategori **{top_cat['Category']}** mendominasi revenue ({format_rupiah(top_cat['Line_Total'])}).
    2. Puncak penjualan terjadi pada **{peak_month['Bulan']}** ({format_rupiah(peak_month['Line_Total'])}) — siapkan stok & promosi.
    3. Rata-rata **basket_size = {format_angka(avg_basket, 2)}** produk per transaksi → peluang *cross-selling*.
    """)

# =========================================================
# TAHAP 3 — PREPROCESSING
# =========================================================
elif menu.startswith("3."):
    render_identity_header(df_clean)
    st.header("3️⃣ Preprocessing Data (Tugas 4–8)")

    st.subheader("📋 Log Keputusan Preprocessing (Tugas 5)")
    log_df = pd.DataFrame(list(prep_log.items()), columns=["Langkah", "Nilai"])
    log_df["Nilai"] = log_df["Nilai"].apply(lambda x: format_angka(x) if isinstance(x, (int, float)) else x)
    st.dataframe(log_df, use_container_width=True)

    st.subheader("📊 Perbandingan Sebelum vs Sesudah")
    c1, c2 = st.columns(2)
    c1.metric("Sebelum", f"{format_angka(len(df_raw))} baris")
    c2.metric("Sesudah", f"{format_angka(len(df_clean))} baris",
              delta=format_angka(len(df_clean)-len(df_raw)))

    st.subheader("🧾 Dataset Bersih (Data_Clean)")
    st.dataframe(df_clean.head(30), use_container_width=True)

    st.subheader("🛒 Basket Size (Tugas 6)")
    st.write(f"Basket_size terbentuk untuk **{format_angka(df_clean['Transaction_ID'].nunique())} transaksi**.")
    bs = df_clean.drop_duplicates("Transaction_ID")["Basket_Size"]
    st.bar_chart(bs.value_counts().sort_index().rename("Jumlah Transaksi"))

    st.subheader("👥 Tabel RFM (Tugas 7)")
    st.write(f"Tanggal acuan RFM: **{prep_log['Tanggal acuan RFM']}**")
    st.dataframe(rfm.head(20), use_container_width=True)

    st.subheader("⚖️ Rencana Standardisasi untuk K-Means (Tugas 8)")
    st.markdown("""
    - Fitur: `Recency`, `Frequency`, `Monetary`.
    - Menggunakan **StandardScaler** karena Monetary (puluhan juta) jauh lebih besar
      dibanding Recency/Frequency → wajib di-scale agar tidak mendominasi jarak Euclidean.
    - Untuk data besar (>1 juta baris): gunakan **sampling** pelanggan atau
      **agregasi** RFM (sudah agregat per pelanggan). Untuk streaming, gunakan **chunking**.
    """)

# =========================================================
# TAHAP 4 — APRIORI
# =========================================================
elif menu.startswith("4."):
    render_identity_header(df_clean)
    st.header("4️⃣ Apriori — Market Basket Analysis (Tugas 9–12)")

    basket = (df_clean.groupby(["Transaction_ID", "Product_Name"])["Quantity"]
                       .sum().unstack().fillna(0))
    basket_bool = (basket > 0).astype(int)
    st.caption(f"Matriks transaksi × produk: **{format_angka(basket_bool.shape[0])} transaksi × "
               f"{format_angka(basket_bool.shape[1])} produk**")

    col1, col2 = st.columns(2)
    min_sup = col1.slider("Minimum Support", 0.005, 0.20, 0.02, 0.005)
    min_conf = col2.slider("Minimum Confidence", 0.10, 0.90, 0.30, 0.05)

    st.caption("**Alasan**: support 2% = kombinasi muncul ≥ ~2% transaksi; "
               "confidence 30% = dari A, 30% juga beli B → aturan cukup kuat "
               "untuk aksi promosi tanpa terlalu banyak noise.")

    try:
        freq_raw = apriori(basket_bool, min_support=min_sup, use_colnames=True, max_len=3)
    except Exception as e:
        st.error(f"Gagal Apriori: {e}")
        st.stop()

    if freq_raw.empty:
        st.warning("Tidak ada itemset memenuhi support. Turunkan min_support.")
    else:
        st.subheader("📦 Frequent Itemsets (Tugas 10)")
        freq_display = freq_raw.copy()
        freq_display["itemsets"] = freq_display["itemsets"].apply(lambda x: ", ".join(sorted(x)))
        st.dataframe(freq_display.sort_values("support", ascending=False).head(20), use_container_width=True)

        rules = association_rules(freq_raw, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["lift"] > 1].sort_values(["lift", "confidence"], ascending=False)

        st.subheader("📜 Top-10 Association Rules (Tugas 11)")
        disp = rules.head(10).copy()
        disp["antecedents"] = disp["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        disp["consequents"] = disp["consequents"].apply(lambda x: ", ".join(sorted(x)))
        st.dataframe(
            disp[["antecedents","consequents","support","confidence","lift"]],
            use_container_width=True)

        st.subheader("🎨 Visualisasi Aturan")
        if not rules.empty:
            top15 = rules.head(15).copy()
            top15["antecedents_str"] = top15["antecedents"].apply(lambda x: ", ".join(sorted(x)))
            top15["consequents_str"] = top15["consequents"].apply(lambda x: ", ".join(sorted(x)))
            top15["rule"] = top15["antecedents_str"] + " → " + top15["consequents_str"]

            fig = px.scatter(top15, x="support", y="confidence", size="lift",
                             color="lift", hover_name="rule",
                             title="Peta Aturan (Support vs Confidence, ukuran = Lift)",
                             color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

            G = nx.DiGraph()
            for _, r in top15.iterrows():
                G.add_edge(r["antecedents_str"], r["consequents_str"], weight=float(r["lift"]))
            pos = nx.spring_layout(G, k=0.9, seed=42)
            edge_x, edge_y = [], []
            for u, v in G.edges():
                edge_x += [pos[u][0], pos[v][0], None]
                edge_y += [pos[u][1], pos[v][1], None]
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            fig_n = go.Figure()
            fig_n.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                       line=dict(width=1, color="#999"), hoverinfo="none"))
            fig_n.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                       marker=dict(size=16, color="#1f77b4"),
                                       text=list(G.nodes()), textposition="top center"))
            fig_n.update_layout(title="Network Aturan Asosiasi (Top-15)",
                                showlegend=False, height=550)
            st.plotly_chart(fig_n, use_container_width=True)

            st.subheader("🛍️ Rekomendasi Cross-Selling / Bundling")
            for _, r in rules.head(3).iterrows():
                a = ", ".join(sorted(r["antecedents"]))
                b = ", ".join(sorted(r["consequents"]))
                st.markdown(
                    f"- **Bundling**: Jika pelanggan beli *{a}*, tawarkan *{b}* "
                    f"(lift = **{format_angka(r['lift'], 2)}**, conf = **{format_angka(r['confidence']*100, 1)}%**). "
                    f"Letakkan produk berdekatan di rak.")

# =========================================================
# TAHAP 5 — K-MEANS
# =========================================================
elif menu.startswith("5."):
    render_identity_header(df_clean)
    st.header("5️⃣ K-Means — Segmentasi Pelanggan (Tugas 13–16)")

    if rfm.empty:
        st.warning("RFM kosong (tidak ada pelanggan non-guest).")
        st.stop()

    X = rfm[["Recency", "Frequency", "Monetary"]].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    st.subheader("🔍 Uji Kandidat k (Tugas 14)")
    ks, inertias, sils = list(range(2, 9)), [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_))

    c1, c2 = st.columns(2)
    fig1 = px.line(x=ks, y=inertias, markers=True,
                   labels={"x":"k","y":"Inertia"}, title="Elbow Method")
    c1.plotly_chart(fig1, use_container_width=True)
    fig2 = px.line(x=ks, y=sils, markers=True,
                   labels={"x":"k","y":"Silhouette"}, title="Silhouette Score")
    c2.plotly_chart(fig2, use_container_width=True)

    best_k = ks[int(np.argmax(sils))]
    k_chosen = st.slider("Pilih k (default = silhouette terbaik)",
                         2, 8, best_k)
    st.info(f"Silhouette tertinggi: **k={best_k}**. "
            f"Interpretasi bisnis: k yang moderat (3–4) lebih mudah dijadikan segmen.")

    km = KMeans(n_clusters=k_chosen, random_state=42, n_init=10).fit(Xs)
    rfm["Cluster"] = km.labels_

    st.subheader("👥 Profil Klaster (Tugas 15)")
    profile = (rfm.groupby("Cluster")
                  .agg(Jumlah=("Customer_ID","count"),
                       Recency_mean=("Recency","mean"),
                       Frequency_mean=("Frequency","mean"),
                       Monetary_mean=("Monetary","mean"),
                       Monetary_median=("Monetary","median"))
                  .round(2).reset_index())

    def label(row):
        if row["Monetary_mean"] >= profile["Monetary_mean"].quantile(.66) \
           and row["Frequency_mean"] >= profile["Frequency_mean"].median():
            return "Loyal (High Value)"
        if row["Recency_mean"] >= profile["Recency_mean"].quantile(.66):
            return "Berisiko Churn"
        if row["Frequency_mean"] <= profile["Frequency_mean"].quantile(.33):
            return "Pelanggan Baru / Potensial"
        return "Reguler"

    profile["Label"] = profile.apply(label, axis=1)

    profile_display = profile.copy()
    profile_display["Jumlah"] = profile_display["Jumlah"].apply(format_angka)
    profile_display["Recency_mean"] = profile_display["Recency_mean"].apply(lambda x: format_angka(x, 1))
    profile_display["Frequency_mean"] = profile_display["Frequency_mean"].apply(lambda x: format_angka(x, 1))
    profile_display["Monetary_mean"] = profile_display["Monetary_mean"].apply(format_rupiah)
    profile_display["Monetary_median"] = profile_display["Monetary_median"].apply(format_rupiah)

    st.dataframe(profile_display, use_container_width=True)

    st.subheader("🎯 Scatter Plot Segmen")
    fig3 = px.scatter(rfm, x="Recency", y="Monetary",
                      color=rfm["Cluster"].astype(str),
                      size="Frequency", hover_data=["Customer_ID"],
                      title="Segmentasi Pelanggan (Recency vs Monetary)")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📣 Strategi Pemasaran per Segmen")
    strat = {
        "Loyal (High Value)": "Program loyalitas VIP, akses early sale, cashback eksklusif.",
        "Reguler": "Promo bundling & up-selling kategori premium.",
        "Pelanggan Baru / Potensial": "Welcome voucher, edukasi produk, program referral.",
        "Berisiko Churn": "Reaktivasi: voucher diskon besar, survei kepuasan, follow-up WhatsApp."
    }
    for _, r in profile.iterrows():
        st.markdown(f"- **{r['Label']}** ({format_angka(r['Jumlah'])} pelanggan) → {strat.get(r['Label'],'-')}")

# =========================================================
# TAHAP 6 — DASHBOARD BI
# =========================================================
elif menu.startswith("6."):
    render_identity_header(df_clean)
    st.header("6️⃣ Dashboard Business Intelligence (Tugas 17–19)")

    st.sidebar.markdown("### 🔎 Filter Dashboard")
    dmin, dmax = df_clean["Transaction_Date"].min(), df_clean["Transaction_Date"].max()
    rng = st.sidebar.date_input("Rentang Tanggal", (dmin.date(), dmax.date()),
                                min_value=dmin.date(), max_value=dmax.date())
    kat = st.sidebar.multiselect("Kategori",
                                 sorted(df_clean["Category"].dropna().unique()),
                                 default=None)

    dff = df_clean.copy()
    if len(rng) == 2:
        dff = dff[(dff["Transaction_Date"].dt.date >= rng[0]) &
                  (dff["Transaction_Date"].dt.date <= rng[1])]
    if kat:
        dff = dff[dff["Category"].isin(kat)]

    st.subheader("📌 KPI Utama")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Penjualan", format_rupiah(dff['Line_Total'].sum()))
    c2.metric("Transaksi Unik", format_angka(dff['Transaction_ID'].nunique()))
    c3.metric("Pelanggan Unik", format_angka(dff['Customer_ID'].nunique()))
    c4.metric("Rata-rata Basket Size",
              format_angka(dff.drop_duplicates('Transaction_ID')['Basket_Size'].mean(), 2))

    st.subheader("📈 Tren Penjualan Bulanan")
    m = (dff.assign(Bulan=dff["Transaction_Date"].dt.to_period("M").astype(str))
            .groupby("Bulan")["Line_Total"].sum().reset_index())
    st.plotly_chart(px.bar(m, x="Bulan", y="Line_Total",
                           title="Revenue per Bulan"), use_container_width=True)

    st.subheader("🏆 Top 10 Produk & Kategori")
    c1, c2 = st.columns(2)
    top_prod = (dff.groupby("Product_Name")["Line_Total"].sum()
                   .sort_values(ascending=False).head(10).reset_index())
    c1.plotly_chart(px.bar(top_prod, x="Line_Total", y="Product_Name",
                           orientation="h", title="Top-10 Produk by Revenue"),
                    use_container_width=True)
    top_cat = (dff.groupby("Category")["Line_Total"].sum()
                  .sort_values(ascending=False).reset_index())
    c2.plotly_chart(px.pie(top_cat, names="Category", values="Line_Total",
                           title="Share Revenue per Kategori"),
                    use_container_width=True)

    st.subheader("🛒 Snapshot Hasil Apriori (Top-5)")
    basket = (dff.groupby(["Transaction_ID","Product_Name"])["Quantity"]
                 .sum().unstack().fillna(0))
    basket_bool = (basket > 0).astype(int)
    try:
        freq = apriori(basket_bool, min_support=0.02, use_colnames=True, max_len=3)
        if not freq.empty:
            rules = association_rules(freq, metric="confidence", min_threshold=0.30)
            rules = rules[rules["lift"] > 1].sort_values("lift", ascending=False).head(5)
            if not rules.empty:
                rules_display = rules.copy()
                rules_display["rule"] = rules_display["antecedents"].apply(lambda x: ", ".join(sorted(x))) + \
                                " → " + rules_display["consequents"].apply(lambda x: ", ".join(sorted(x)))
                st.dataframe(rules_display[["rule","support","confidence","lift"]],
                             use_container_width=True)
            else:
                st.info("Tidak ada rule lift>1 pada filter saat ini.")
    except Exception as e:
        st.warning(f"Apriori dilewati: {e}")

    st.subheader("👥 Profil Segmen K-Means")
    if not rfm.empty:
        X = rfm[["Recency","Frequency","Monetary"]].values
        Xs = StandardScaler().fit_transform(X)
        kk = min(4, max(2, len(rfm)//5))
        km = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(Xs)
        tmp = rfm.copy(); tmp["Cluster"] = km.labels_
        prof = (tmp.groupby("Cluster")
                   .agg(Jumlah=("Customer_ID","count"),
                        R=("Recency","mean"),
                        F=("Frequency","mean"),
                        M=("Monetary","mean")).round(1).reset_index())
        st.dataframe(prof, use_container_width=True)

    st.caption("💡 **Cara membaca**: Manajemen dapat memfilter tanggal/kategori, "
               "memantau KPI, dan mengidentifikasi segmen untuk kampanye. "
               "Bonus: integrasi Apriori–K-Means memungkinkan bundling per segmen.")

# =========================================================
# TAHAP 7 — REKOMENDASI
# =========================================================
elif menu.startswith("7."):
    render_identity_header(df_clean)
    st.header("7️⃣ Rekomendasi Keputusan (Tugas 20–22)")

    st.markdown("""
    > ⚠️ **Pemisahan fakta vs asumsi vs dampak aktual**
    > - **Fakta** = angka dari data historis tahun 2025 (mis. lift, rata-rata RFM).
    > - **Estimasi** = proyeksi berbasis asumsi (mis. +5% revenue).
    > - **Dampak aktual** = *belum diuji* → harus divalidasi lewat A/B test / pilot.
    """)

    st.subheader("🚀 Rekomendasi Prioritas 1 — Bundling Produk (Cross-Selling)")
    st.markdown(f"""
    - **Bukti data**: Aturan asosiasi dengan **lift > 1** (lihat tab Apriori).
    - **Tindakan**: Buat paket bundling produk dengan lift tertinggi; letakkan
      produk berdekatan di rak, misal Sabun Mandi + Sampo.
    - **KPI keberhasilan**: Basket_size rata-rata naik dari {format_angka(df_clean['Basket_Size'].mean(), 2)} → target +10%;
      attachment rate produk B.
    - **Estimasi biaya/manfaat**: Biaya promosi/desain ~{format_rupiah(2000000)}; estimasi uplift revenue
      3–7% dari produk terkait (*asumsi*, perlu divalidasi).
    - **Evaluasi**: Bandingkan basket_size & revenue 30 hari sebelum vs sesudah.
    """)

    st.subheader("💎 Rekomendasi Prioritas 2 — Retensi Pelanggan (Segmen Loyals)")
    st.markdown("""
    - **Bukti data**: Segmen *Loyal (High Value)* teridentifikasi di K-Means
      (Monetary & Frequency tinggi).
    - **Tindakan**: Program member VIP, akses early sale, cashback eksklusif.
    - **KPI keberhasilan**: Retention rate pelanggan top-20% ≥ 85%;
      Frequency +5% YoY.
    - **Estimasi biaya/manfaat**: Reward 2–3% dari revenue segmen; manfaat = LTV naik
      (asumsi, belum diuji).
    - **Evaluasi**: Cohort analysis pelanggan top vs kontrol.
    """)

    st.subheader("🎯 Rekomendasi Prioritas 3 — Reaktivasi Segmen 'Berisiko Churn'")
    st.markdown("""
    - **Bukti data**: Recency tinggi (≥ kuartil 3) tapi Frequency historis > 1.
    - **Tindakan**: Kirim voucher diskon 20–30% + personalisasi produk favorit.
    - **KPI keberhasilan**: Reaktivasi ≥ 15% segmen dalam 30 hari.
    - **Biaya/manfaat**: Biaya voucher vs nilai transaksi ulang (estimasi, wajib pilot).
    - **Evaluasi**: A/B test — 50% dapat voucher vs 50% tidak.
    """)

    st.subheader("⚠️ Disclaimer Dampak")
    st.info("Semua proyeksi kenaikan penjualan bersifat **estimasi berbasis asumsi** "
            "dan belum divalidasi melalui uji lapangan. Wajib pilot/A-B test sebelum "
            "diklaim sebagai dampak aktual.")