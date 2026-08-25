import os
from celery import Celery
import time

# Use Redis as the broker and backend
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "codecortex_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@celery_app.task(bind=True, name="process_verification")
def process_verification(self, tracking_id, data):
    """
    Simulates the Heavy ML Pipeline processing sequentially/parallel.
    In production, this would call paddleocr, insightface, opencv etc.
    """
    import random
    
    # Update state: ELA
    self.update_state(state='PROGRESS', meta={'module': 'ELA Forensics', 'progress': 10})
    time.sleep(1)
    ela_pass = random.choice([True, False])
    
    # Update state: MRZ
    self.update_state(state='PROGRESS', meta={'module': 'MRZ Extraction', 'progress': 30})
    time.sleep(1)
    mrz_valid = True
    
    # Update state: Facial Biometrics
    self.update_state(state='PROGRESS', meta={'module': 'Facial Biometrics', 'progress': 50})
    time.sleep(1.5)
    face_similarity = random.uniform(40.0, 99.9)
    
    # Update state: Fingerprint/Iris
    self.update_state(state='PROGRESS', meta={'module': 'Fingerprint & Iris Match', 'progress': 80})
    time.sleep(1)
    
    # Update state: Interpol
    self.update_state(state='PROGRESS', meta={'module': 'INTERPOL Database Search', 'progress': 95})
    time.sleep(1)
    
    decision = "HIGH_RISK" if not ela_pass or face_similarity < 60 else "VERIFIED"
    risk_score = int((100 - face_similarity) + (50 if not ela_pass else 0))
    if risk_score > 100: risk_score = 99
    
    return {
        "status": "complete",
        "progress": 100,
        "result": {
            "ela_pass": ela_pass,
            "mrz_valid": mrz_valid,
            "face_similarity": round(face_similarity, 2),
            "fingerprint_match": True,
            "iris_match": True,
            "risk_score": risk_score,
            "decision": decision
        }
    }
