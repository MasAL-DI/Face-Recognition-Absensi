"""
export_rekap_excel.py
Ekspor rekap absensi satu sesi ke file Excel.
"""

from __future__ import annotations

from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from config import EXPORT_DIR
from database import get_rekap_sesi, get_sesi_by_id, init_db, list_sesi

BASE_DIR = Path(__file__).resolve().parent
EXPORT_PATH = BASE_DIR / EXPORT_DIR


def choose_session() -> int | None:
    sessions = list_sesi(limit=30)
    if not sessions:
        print("Belum ada sesi.")
        return None

    print("\nSesi terbaru:")
    for s in sessions:
        print(f"[{s['id_sesi']}] {s['tanggal']} | {s['kelas']} | {s['nama_mapel']} | {s['jam_mulai']}-{s['jam_selesai']} | hadir {s['jumlah_hadir']}")

    raw = input("Masukkan ID sesi yang akan diekspor: ").strip()
    if not raw.isdigit():
        print("ID harus angka.")
        return None
    return int(raw)


def export_session(id_sesi: int) -> Path | None:
    sesi = get_sesi_by_id(id_sesi)
    if not sesi:
        print("Sesi tidak ditemukan.")
        return None

    rows = get_rekap_sesi(id_sesi)
    EXPORT_PATH.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Rekap Absensi"

    title = "REKAP ABSENSI SISWA"
    ws.merge_cells("A1:J1")
    ws["A1"] = title
    ws["A1"].font = Font(size=14, bold=True)
    ws["A1"].alignment = Alignment(horizontal="center")

    meta = [
        ("Tanggal", sesi["tanggal"]),
        ("Kelas", sesi["kelas"]),
        ("Mata Pelajaran", sesi["nama_mapel"]),
        ("Guru", sesi["nama_guru"] or "-"),
        ("Jam", f"{sesi['jam_mulai']} - {sesi['jam_selesai']}"),
    ]

    start_row = 3
    for i, (k, v) in enumerate(meta, start=start_row):
        ws[f"A{i}"] = k
        ws[f"B{i}"] = v
        ws[f"A{i}"].font = Font(bold=True)

    header_row = 9
    headers = ["No Absen", "NIS", "Nama", "Kelas", "Status", "Jam Absen", "Skor LBPH", "Liveness", "Metode", "Detail"]
    for col, h in enumerate(headers, start=1):
        cell = ws.cell(header_row, col, h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center")

    for r_idx, row in enumerate(rows, start=header_row + 1):
        status = "HADIR" if row["jam_absen"] else "BELUM/ABSEN"
        values = [
            int(row["no_absen"]),
            row["nis"],
            row["nama"],
            row["kelas"],
            status,
            row["jam_absen"] or "-",
            row["skor_lbph"] if row["skor_lbph"] is not None else "-",
            "YA" if row["liveness_passed"] else "-",
            row["liveness_method"] or "-",
            row["liveness_detail"] or "-",
        ]
        for c_idx, value in enumerate(values, start=1):
            ws.cell(r_idx, c_idx, value)

    thin = Side(style="thin", color="D9E2F3")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for row_cells in ws.iter_rows(min_row=header_row, max_row=header_row + len(rows), min_col=1, max_col=len(headers)):
        for cell in row_cells:
            cell.border = border
            cell.alignment = Alignment(vertical="center", wrap_text=True)

    for col in range(1, len(headers) + 1):
        letter = get_column_letter(col)
        max_len = max(len(str(ws.cell(row, col).value or "")) for row in range(1, header_row + len(rows) + 1))
        ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 36)

    safe_mapel = str(sesi["nama_mapel"]).replace(" ", "_").replace("/", "_")
    out = EXPORT_PATH / f"rekap_{sesi['tanggal']}_{sesi['kelas'].replace(' ', '_')}_{safe_mapel}_{id_sesi}.xlsx"
    wb.save(out)
    return out


def main() -> None:
    init_db()
    id_sesi = choose_session()
    if id_sesi is None:
        return
    out = export_session(id_sesi)
    if out:
        print(f"File ekspor dibuat: {out}")


if __name__ == "__main__":
    main()
