"""
rekap.py
Laporan absensi:
- rekap per sesi
- ringkasan per tanggal
- riwayat siswa
"""

from __future__ import annotations

from datetime import datetime

from database import (
    get_absensi_by_siswa,
    get_rekap_sesi,
    get_ringkasan_tanggal,
    get_sesi_by_id,
    init_db,
    list_sesi,
    search_siswa,
)


def line(n: int = 120) -> str:
    return "-" * n


def choose_session() -> int | None:
    sessions = list_sesi(limit=30)
    if not sessions:
        print("Belum ada sesi absensi.")
        return None

    print("\nSesi terbaru:")
    print(line())
    print(f"{'ID':<5} {'Tanggal':<12} {'Kelas':<14} {'Mapel':<24} {'Jam':<13} {'Hadir'}")
    print(line())
    for s in sessions:
        jam = f"{s['jam_mulai']}-{s['jam_selesai']}"
        print(f"{s['id_sesi']:<5} {s['tanggal']:<12} {s['kelas']:<14} {s['nama_mapel']:<24} {jam:<13} {s['jumlah_hadir']}")
    print(line())

    raw = input("ID sesi: ").strip()
    if not raw.isdigit():
        print("ID sesi harus angka.")
        return None
    return int(raw)


def report_session(id_sesi: int) -> None:
    sesi = get_sesi_by_id(id_sesi)
    if sesi is None:
        print("Sesi tidak ditemukan.")
        return

    rows = get_rekap_sesi(id_sesi)

    print("\n" + "=" * 120)
    print("REKAP ABSENSI PER SESI")
    print("=" * 120)
    print(f"Tanggal       : {sesi['tanggal']}")
    print(f"Kelas         : {sesi['kelas']}")
    print(f"Mata pelajaran: {sesi['nama_mapel']}")
    print(f"Guru          : {sesi['nama_guru'] or '-'}")
    print(f"Jam           : {sesi['jam_mulai']} - {sesi['jam_selesai']}")
    print(line())
    print(f"{'No':<5} {'NIS':<16} {'Nama':<30} {'Status':<13} {'Jam':<10} {'Skor':<8} {'Live':<6} {'Detail'}")
    print(line())

    hadir = 0
    for r in rows:
        status = "HADIR" if r["jam_absen"] else "BELUM"
        jam = r["jam_absen"] or "-"
        skor = f"{r['skor_lbph']:.1f}" if r["skor_lbph"] is not None else "-"
        live = "YA" if r["liveness_passed"] else "-"
        if r["jam_absen"]:
            hadir += 1
        print(
            f"{int(r['no_absen']):<5} "
            f"{r['nis']:<16} "
            f"{r['nama']:<30} "
            f"{status:<13} "
            f"{jam:<10} "
            f"{skor:<8} "
            f"{live:<6} "
            f"{r['liveness_detail'] or '-'}"
        )

    print(line())
    total = len(rows)
    pct = hadir / total * 100 if total else 0
    print(f"Hadir: {hadir}/{total} siswa ({pct:.2f}%)")


def report_date() -> None:
    default_date = datetime.now().strftime("%Y-%m-%d")
    tanggal = input(f"Tanggal [default {default_date}]: ").strip() or default_date
    rows = get_ringkasan_tanggal(tanggal)

    print("\n" + "=" * 100)
    print(f"RINGKASAN SESI TANGGAL {tanggal}")
    print("=" * 100)
    if not rows:
        print("Tidak ada sesi.")
        return

    print(f"{'ID':<5} {'Kelas':<14} {'Mapel':<24} {'Jam':<13} {'Guru':<18} {'Hadir'}")
    print(line(100))
    for r in rows:
        jam = f"{r['jam_mulai']}-{r['jam_selesai']}"
        hadir = f"{r['jumlah_hadir']}/{r['total_siswa']}"
        print(f"{r['id_sesi']:<5} {r['kelas']:<14} {r['nama_mapel']:<24} {jam:<13} {(r['nama_guru'] or '-'):<18} {hadir}")


def choose_student():
    keyword = input("Masukkan no absen/NIS/nama/kelas: ").strip()
    if not keyword:
        print("Kata kunci kosong.")
        return None

    rows = search_siswa(keyword)
    if not rows:
        print("Siswa tidak ditemukan.")
        return None

    if len(rows) == 1:
        return rows[0]

    print("\nHasil pencarian:")
    print(line(100))
    print(f"{'Pilih':<6} {'No':<5} {'NIS':<16} {'Nama':<30} {'Kelas':<14}")
    print(line(100))
    for i, r in enumerate(rows, start=1):
        print(f"{i:<6} {int(r['no_absen']):<5} {r['nis']:<16} {r['nama']:<30} {r['kelas']:<14}")

    raw = input("Pilih nomor: ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(rows)):
        print("Pilihan tidak valid.")
        return None
    return rows[int(raw) - 1]


def report_student_history() -> None:
    siswa = choose_student()
    if not siswa:
        return

    rows = get_absensi_by_siswa(int(siswa["id_siswa"]))
    print("\n" + "=" * 110)
    print("RIWAYAT ABSENSI SISWA")
    print("=" * 110)
    print(f"No. Absen: {siswa['no_absen']}")
    print(f"NIS      : {siswa['nis']}")
    print(f"Nama     : {siswa['nama']}")
    print(f"Kelas    : {siswa['kelas']}")
    print(line(110))
    print(f"{'Tanggal':<12} {'Mapel':<24} {'Jam Pelajaran':<16} {'Jam Absen':<10} {'Skor':<8} {'Live'}")
    print(line(110))

    if not rows:
        print("Belum ada riwayat.")
        return

    for r in rows:
        jam_pel = f"{r['jam_mulai']}-{r['jam_selesai']}"
        skor = f"{r['skor_lbph']:.1f}" if r["skor_lbph"] is not None else "-"
        live = "YA" if r["liveness_passed"] else "TIDAK"
        print(f"{r['tanggal']:<12} {r['nama_mapel']:<24} {jam_pel:<16} {r['jam_absen']:<10} {skor:<8} {live}")


def main() -> None:
    init_db()
    print("=" * 100)
    print("LAPORAN ABSENSI")
    print("=" * 100)
    print("[1] Rekap satu sesi")
    print("[2] Ringkasan semua sesi pada tanggal")
    print("[3] Riwayat satu siswa")
    choice = input("Pilihan: ").strip()

    if choice == "1":
        id_sesi = choose_session()
        if id_sesi is not None:
            report_session(id_sesi)
    elif choice == "2":
        report_date()
    elif choice == "3":
        report_student_history()
    else:
        print("Pilihan tidak valid.")


if __name__ == "__main__":
    main()
