py -3.11 -m venv .venv311
.\.venv311\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

python -m PyInstaller --windowed --name CheatingDetector --collect-data cv2 main.py

Copy-Item yolov8n.pt dist\CheatingDetector\ -Force

Write-Host "Application creee : dist\CheatingDetector\CheatingDetector.exe"