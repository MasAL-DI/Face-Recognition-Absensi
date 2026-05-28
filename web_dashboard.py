"""
Dashboard web Streamlit untuk proyek absensi wajah.

Fitur:
- login admin/guru
- ringkasan jumlah siswa, sesi, dan kehadiran
- tabel rekap absensi
- filter tanggal/kelas/mapel
- ekspor CSV
- audit login sederhana

Jalankan:
streamlit run web_dashboard.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from auth_db import list_users, verify_login
from config_lanjutan import DASHBOARD_TITLE, DATABASE_FILE
from security_utils import decrypt_text

st.set_page_config(page_title=DASHBOARD_TITLE, page_icon="📌", layout="wide")


@st.cache_data(ttl=10)
def table_exists(db_path: str, table_name: str) -> bool:
    path = Path(db_path)
    if not path.exists():
        return False
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
    return row is not None


@st.cache_data(ttl=10)
def read_sql(db_path: str, query: str, params: tuple = ()) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def decrypt_dataframe_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda x: decrypt_text(str(x)) if pd.notna(x) else x)
    return out


def login_panel() -> None:
    st.title(DASHBOARD_TITLE)
    st.caption("Login diperlukan agar dashboard tidak menjadi papan pengumuman gratis untuk siapa saja yang kebetulan lewat.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Masuk")

    if submitted:
        user = verify_login(username, password)
        if user:
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("Login gagal. Periksa username atau password.")


def metric_card(label: str, value: int | str) -> None:
    st.metric(label, value)


def get_summary(db_path: str) -> dict:
    result = {"siswa": 0, "sesi": 0, "absensi": 0, "kelas": 0}
    if not Path(db_path).exists():
        return result

    if table_exists(db_path, "siswa"):
        df = read_sql(db_path, "SELECT COUNT(*) AS n, COUNT(DISTINCT kelas) AS kelas FROM siswa")
        if not df.empty:
            result["siswa"] = int(df.loc[0, "n"])
            result["kelas"] = int(df.loc[0, "kelas"])

    if table_exists(db_path, "sesi_pelajaran"):
        df = read_sql(db_path, "SELECT COUNT(*) AS n FROM sesi_pelajaran")
        if not df.empty:
            result["sesi"] = int(df.loc[0, "n"])

    if table_exists(db_path, "absensi"):
        df = read_sql(db_path, "SELECT COUNT(*) AS n FROM absensi")
        if not df.empty:
            result["absensi"] = int(df.loc[0, "n"])

    return result


def load_rekap(db_path: str) -> pd.DataFrame:
    needed = all(table_exists(db_path, t) for t in ["siswa", "sesi_pelajaran", "mapel", "absensi"])
    if not needed:
        return pd.DataFrame()

    query = """
    SELECT
        sp.tanggal,
        sp.kelas,
        m.nama_mapel,
        sp.nama_guru,
        sp.jam_mulai,
        sp.jam_selesai,
        s.no_absen,
        s.nis,
        s.nama,
        a.jam_absen,
        a.skor_lbph AS skor_or_similarity,
        a.liveness_method,
        a.liveness_passed,
        a.liveness_detail
    FROM absensi a
    JOIN siswa s ON s.id_siswa = a.id_siswa
    JOIN sesi_pelajaran sp ON sp.id_sesi = a.id_sesi
    JOIN mapel m ON m.id_mapel = sp.id_mapel
    ORDER BY sp.tanggal DESC, sp.jam_mulai DESC, s.no_absen ASC
    """
    df = read_sql(db_path, query)
    return decrypt_dataframe_columns(df, ["nis", "nama", "nama_guru"])


def load_siswa(db_path: str) -> pd.DataFrame:
    if not table_exists(db_path, "siswa"):
        return pd.DataFrame()
    df = read_sql(
        db_path,
        "SELECT no_absen, nis, nama, kelas, folder_wajah FROM siswa ORDER BY kelas, no_absen",
    )
    return decrypt_dataframe_columns(df, ["nis", "nama"])


def dashboard() -> None:
    user = st.session_state.get("user")
    st.sidebar.title("Menu")
    st.sidebar.write(f"Masuk sebagai: **{user['nama_lengkap']}**")
    st.sidebar.write(f"Role: **{user['role']}**")
    if st.sidebar.button("Keluar"):
        st.session_state.clear()
        st.rerun()

    db_path = str(DATABASE_FILE)
    st.title(DASHBOARD_TITLE)
    st.caption("Dashboard rekap absensi, sesi kelas, dan monitoring liveness detection.")

    if not Path(db_path).exists():
        st.error(f"Database absensi belum ditemukan: {db_path}")
        st.stop()

    summary = get_summary(db_path)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total siswa", summary["siswa"])
    with col2:
        metric_card("Total kelas", summary["kelas"])
    with col3:
        metric_card("Total sesi", summary["sesi"])
    with col4:
        metric_card("Total data hadir", summary["absensi"])

    tab_rekap, tab_siswa, tab_user, tab_audit = st.tabs(["Rekap Absensi", "Data Siswa", "User", "Audit Login"])

    with tab_rekap:
        df = load_rekap(db_path)
        if df.empty:
            st.warning("Data rekap belum tersedia atau struktur tabel belum sesuai.")
        else:
            c1, c2, c3 = st.columns(3)
            tanggal_options = ["Semua"] + sorted(df["tanggal"].dropna().unique().tolist(), reverse=True)
            kelas_options = ["Semua"] + sorted(df["kelas"].dropna().unique().tolist())
            mapel_options = ["Semua"] + sorted(df["nama_mapel"].dropna().unique().tolist())

            tanggal = c1.selectbox("Tanggal", tanggal_options)
            kelas = c2.selectbox("Kelas", kelas_options)
            mapel = c3.selectbox("Mata pelajaran", mapel_options)

            filtered = df.copy()
            if tanggal != "Semua":
                filtered = filtered[filtered["tanggal"] == tanggal]
            if kelas != "Semua":
                filtered = filtered[filtered["kelas"] == kelas]
            if mapel != "Semua":
                filtered = filtered[filtered["nama_mapel"] == mapel]

            st.dataframe(filtered, use_container_width=True, hide_index=True)
            csv = filtered.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Unduh CSV", data=csv, file_name="rekap_absensi.csv", mime="text/csv")

            if not filtered.empty:
                chart = filtered.groupby(["tanggal", "kelas"], as_index=False).size()
                chart = chart.rename(columns={"size": "jumlah_hadir"})
                st.bar_chart(chart, x="tanggal", y="jumlah_hadir", color="kelas")

    with tab_siswa:
        df_siswa = load_siswa(db_path)
        if df_siswa.empty:
            st.warning("Data siswa belum tersedia.")
        else:
            st.dataframe(df_siswa, use_container_width=True, hide_index=True)

    with tab_user:
        if user["role"] != "admin":
            st.info("Hanya admin yang dapat melihat daftar user.")
        else:
            users = pd.DataFrame(list_users())
            if users.empty:
                st.warning("Belum ada user.")
            else:
                st.dataframe(users, use_container_width=True, hide_index=True)
            st.caption("Pembuatan user baru dilakukan lewat terminal: python buat_admin.py")

    with tab_audit:
        audit_db = Path("database/auth_users.db")
        if table_exists(str(audit_db), "audit_login"):
            audit = read_sql(
                str(audit_db),
                "SELECT username, role, success, message, created_at FROM audit_login ORDER BY created_at DESC LIMIT 100",
            )
            st.dataframe(audit, use_container_width=True, hide_index=True)
        else:
            st.info("Audit login belum tersedia.")


def main() -> None:
    if "user" not in st.session_state:
        login_panel()
    else:
        dashboard()


if __name__ == "__main__":
    main()
