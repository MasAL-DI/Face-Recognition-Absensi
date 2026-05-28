"""
log_uji_manual.py
Mencatat hasil uji skripsi secara manual ke tabel log_uji.

Gunakan untuk mencatat:
- wajah normal
- wajah tertutup
- orang tidak terdaftar
- foto dari HP
- cahaya terang/redup
"""

from database import init_db, log_uji


def main() -> None:
    init_db()
    print("=" * 80)
    print("LOG UJI MANUAL")
    print("=" * 80)

    jenis_uji = input("Jenis uji [normal/tertutup/orang_lain/foto_hp/cahaya/jarak]: ").strip()
    nama_subjek = input("Nama subjek: ").strip()
    kelas = input("Kelas: ").strip()
    kondisi_cahaya = input("Kondisi cahaya [terang/redup/backlight]: ").strip()
    jarak_raw = input("Jarak kamera cm [kosongkan jika tidak diukur]: ").strip()
    expected_label = input("Label seharusnya: ").strip()
    predicted_label = input("Label yang muncul: ").strip()
    skor_raw = input("Skor LBPH [kosongkan jika tidak ada]: ").strip()
    live_raw = input("Liveness passed? [y/n/kosong]: ").strip().lower()
    hasil = input("Hasil [TP/TN/FP/FN/BERHASIL/GAGAL]: ").strip()
    catatan = input("Catatan: ").strip()

    jarak_cm = int(jarak_raw) if jarak_raw.isdigit() else None
    skor = float(skor_raw) if skor_raw else None
    liveness_passed = None
    if live_raw == "y":
        liveness_passed = True
    elif live_raw == "n":
        liveness_passed = False

    log_uji(
        jenis_uji=jenis_uji,
        nama_subjek=nama_subjek,
        kelas=kelas,
        kondisi_cahaya=kondisi_cahaya,
        jarak_cm=jarak_cm,
        expected_label=expected_label,
        predicted_label=predicted_label,
        skor_lbph=skor,
        liveness_passed=liveness_passed,
        hasil=hasil,
        catatan=catatan,
    )

    print("Log uji tersimpan.")


if __name__ == "__main__":
    main()
