"""
import_siswa_excel.py
Mengimpor data siswa dari Excel (.xlsx) ke SQLite.

Format header yang didukung:
- no_absen / no absen / nomor absen
- nis
- nama
- kelas

Jika kolom kelas kosong, sistem memakai KELAS_AKTIF dari config.py.
"""

from __future__ import annotations

from pathlib import Path
from openpyxl import load_workbook

from config import KELAS_AKTIF
from database import init_db, upsert_siswa

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EXCEL = BASE_DIR / "data_siswa.xlsx"


def clean_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return " ".join(str(value).strip().split())


def normalize_header(value) -> str:
    text = clean_cell(value).lower()
    text = text.replace(".", "")
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    return text


def find_header(headers: dict[str, int], variants: list[str]) -> int | None:
    for variant in variants:
        if variant in headers:
            return headers[variant]
    return None


def main() -> None:
    init_db()

    print("=" * 80)
    print("IMPORT DATA SISWA DARI EXCEL")
    print("=" * 80)
    print(f"Kelas default: {KELAS_AKTIF}")

    path_input = input(f"Path Excel [Enter = {DEFAULT_EXCEL.name}]: ").strip()
    excel_path = Path(path_input) if path_input else DEFAULT_EXCEL
    if not excel_path.is_absolute():
        excel_path = BASE_DIR / excel_path

    if not excel_path.exists():
        print(f"File tidak ditemukan: {excel_path}")
        return

    wb = load_workbook(excel_path, data_only=True)
    ws = wb.active

    headers = {}
    for idx, cell in enumerate(ws[1], start=1):
        h = normalize_header(cell.value)
        if h:
            headers[h] = idx

    no_absen_col = find_header(headers, ["no_absen", "nomor_absen", "no", "absen"])
    nis_col = find_header(headers, ["nis", "nomor_induk", "nomor_induk_siswa"])
    nama_col = find_header(headers, ["nama", "nama_siswa"])
    kelas_col = find_header(headers, ["kelas"])

    if no_absen_col is None or nis_col is None or nama_col is None:
        print("Header wajib tidak lengkap.")
        print("Minimal header Excel: no_absen, nis, nama")
        print("Kolom kelas boleh ada. Jika kosong/tidak ada, memakai KELAS_AKTIF.")
        return

    inserted = 0
    updated = 0
    skipped = 0
    errors: list[str] = []

    for row_num in range(2, ws.max_row + 1):
        no_absen_raw = clean_cell(ws.cell(row_num, no_absen_col).value)
        nis = clean_cell(ws.cell(row_num, nis_col).value)
        nama = clean_cell(ws.cell(row_num, nama_col).value)
        kelas = clean_cell(ws.cell(row_num, kelas_col).value) if kelas_col else KELAS_AKTIF
        kelas = kelas or KELAS_AKTIF

        if not no_absen_raw and not nis and not nama:
            continue

        try:
            no_absen = int(no_absen_raw)
            status, _, folder = upsert_siswa(no_absen, nis, nama, kelas)

            # Pastikan folder wajah dibuat.
            (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

            if status == "inserted":
                inserted += 1
            else:
                updated += 1

            print(f"{status.upper():8} | {no_absen:02d} | {nis} | {nama} | {kelas}")

        except Exception as exc:
            skipped += 1
            errors.append(f"Baris {row_num}: {exc}")

    print("\n" + "=" * 80)
    print("IMPORT SELESAI")
    print(f"Siswa baru       : {inserted}")
    print(f"Siswa diperbarui : {updated}")
    print(f"Baris gagal      : {skipped}")

    if errors:
        print("\nDetail gagal:")
        for item in errors:
            print("- " + item)

    print("\nLangkah berikutnya untuk versi ArcFace:")
    print("1. python daftar_wajah.py")
    print("2. python build_face_embeddings.py")
    print("3. python absensi_lanjutan_arcface_cnn.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
