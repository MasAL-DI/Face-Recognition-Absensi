import cv2
from pathlib import Path

SAVE_DIR = Path("dataset_spoof/train/spoof")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Arahkan webcam ke foto/layar HP yang menampilkan wajah")
print("SPACE = simpan | ESC = selesai")

while True:
    ret, frame = cap.read()
    cv2.putText(frame, f"Tersimpan: {count} | SPACE simpan | ESC selesai",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Kumpulkan Data Spoof", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == 32:
        filename = SAVE_DIR / f"spoof_{count:03d}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"Tersimpan: {filename.name}")
        count += 1

cap.release()
cv2.destroyAllWindows()
print(f"Total foto spoof: {count}")