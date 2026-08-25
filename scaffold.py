import os

backend_dir = r"C:\Users\debad\CodeCortexPlatform\backend"
dirs = ["api", "core", "services", "db", "models"]

for d in dirs:
    os.makedirs(os.path.join(backend_dir, d), exist_ok=True)
    with open(os.path.join(backend_dir, d, "__init__.py"), "w") as f:
        pass

requirements = \"\"\"
fastapi==0.104.1
uvicorn[standard]==0.24.0.post1
celery==5.3.6
redis==5.0.1
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
python-multipart==0.0.6
websockets==12.0
PyJWT==2.8.0
faiss-cpu==1.7.4
opencv-python==4.8.1.78
Pillow==10.1.0
insightface==0.7.3
pytesseract==0.3.10
\"\"\"
with open(os.path.join(backend_dir, "requirements.txt"), "w") as f:
    f.write(requirements.strip())

