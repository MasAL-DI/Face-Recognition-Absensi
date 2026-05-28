"""
reset_data.py
Reset data uji.
Aman untuk menghapus absensi tanpa menghapus data siswa.
"""

from database import delete_absensi_all, delete_sesi_and_absensi_all


def main() -> None:
    print("=" * 70)
    print("RESET DATA")
    print("=" * 70)
    print("[1] Hapus semua data absensi saja")
    print("[2] Hapus semua sesi pelajaran + absensi")
    print("[3] Batal")
    choice = input("Pilihan: ").strip()

    if choice == "1":
        confirm = input("Yakin hapus semua data absensi? ketik HAPUS: ").strip()
        if confirm == "HAPUS":
            delete_absensi_all()
            print("Semua data absensi dihapus.")
        else:
            print("Dibatalkan.")
    elif choice == "2":
        confirm = input("Yakin hapus semua sesi + absensi? ketik HAPUS: ").strip()
        if confirm == "HAPUS":
            delete_sesi_and_absensi_all()
            print("Semua sesi dan absensi dihapus.")
        else:
            print("Dibatalkan.")
    else:
        print("Batal.")


if __name__ == "__main__":
    main()
