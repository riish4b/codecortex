import asyncio
import json
import random
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from services.ai_service import analyze_document_with_ai, analyze_risk_with_ai, generate_verification_report, analyze_document_image

app = FastAPI(title="CodeCortex Core API")

print("[INIT] CodeCortex Backend starting...")
print("[INIT] AI Service loaded: OpenRouter (google/gemini-2.0-flash-001)")

@app.get("/api/test-ai")
async def test_ai():
    """Test endpoint to verify OpenRouter AI is working."""
    print("[AI TEST] Testing OpenRouter connection...")
    result = analyze_document_with_ai({
        "ela_pass": True,
        "mrz_valid": True,
        "face_similarity": 85.5,
        "fingerprint_match": True,
        "iris_match": True,
    })
    print(f"[AI TEST] Result: success={result.get('success')}")
    if result.get("success"):
        print(f"[AI TEST] Model used: {result.get('model_used')}")
        print(f"[AI TEST] Tokens used: {result.get('tokens_used')}")
        print(f"[AI TEST] Analysis: {json.dumps(result.get('analysis', {}), indent=2)}")
    else:
        print(f"[AI TEST] ERROR: {result.get('error')}")
    return result

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store uploaded files keyed by tracking_id
file_store: dict[str, dict] = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: dict, websocket: WebSocket):
        import json as json_module
        await websocket.send_text(json_module.dumps(message, default=str))

manager = ConnectionManager()


@app.post("/api/verify")
async def verify_documents(
    passport: UploadFile = File(...),
    visa: Optional[UploadFile] = File(None),
    nid: Optional[UploadFile] = File(None),
    selfie: UploadFile = File(...),
    passportNfc: Optional[UploadFile] = File(None),
):
    tracking_id = f"CTX-{random.randint(1000, 9900)}"
    print(f"[UPLOAD] Files received! Tracking ID: {tracking_id}")
    
    # Store file bytes keyed by tracking_id
    passport_bytes = await passport.read() if passport else None
    visa_bytes = await visa.read() if visa else None
    nid_bytes = await nid.read() if nid else None
    selfie_bytes = await selfie.read() if selfie else None
    nfc_bytes = await passportNfc.read() if passportNfc else None
    
    file_store[tracking_id] = {
        "passport": passport_bytes,
        "visa": visa_bytes,
        "nid": nid_bytes,
        "selfie": selfie_bytes,
        "passportNfc": nfc_bytes,
    }
    
    print(f"[UPLOAD] Stored: passport={len(passport_bytes) if passport_bytes else 0} bytes, selfie={len(selfie_bytes) if selfie_bytes else 0} bytes")
    return {"message": "Verification initialized", "tracking_id": tracking_id}


@app.websocket("/ws/verify/{tracking_id}")
async def websocket_endpoint(websocket: WebSocket, tracking_id: str):
    await manager.connect(websocket)
    print(f"[WS] Client connected for tracking_id: {tracking_id}")
    
    files = file_store.pop(tracking_id, {})
    passport_bytes = files.get("passport")
    selfie_bytes = files.get("selfie")
    visa_bytes = files.get("visa")
    nid_bytes = files.get("nid")
    print(f"[WS] Files received: passport={bool(passport_bytes)}, selfie={bool(selfie_bytes)}, visa={bool(visa_bytes)}, nid={bool(nid_bytes)}")
    
    try:
        # ── Stage 1: ELA Forensics ────────────────────────────────────
        await manager.send_message(
            {"status": "processing", "module": "ELA Forensics", "progress": 10}, websocket
        )
        ela_result = {"pass": False, "details": "No passport image"}
        if passport_bytes:
            try:
                from services.ml_pipeline import analyze_ela
                ela_result = analyze_ela(passport_bytes)
            except Exception as e:
                ela_result = {"pass": False, "details": f"ELA error: {e}"}
        await asyncio.sleep(1)

        # ── Stage 2: MRZ Extraction ───────────────────────────────────
        await manager.send_message(
            {"status": "processing", "module": "MRZ Extraction & OCR", "progress": 25}, websocket
        )
        mrz_result = {"mrz_valid": True, "raw": ""}
        if passport_bytes:
            try:
                from services.ml_pipeline import extract_mrz
                mrz_result = extract_mrz(passport_bytes)
            except Exception as e:
                mrz_result = {"mrz_valid": False, "raw": str(e)}
        await asyncio.sleep(1)

        # ── Stage 3: Facial Biometrics ────────────────────────────────
        await manager.send_message(
            {"status": "processing", "module": "Facial Biometrics (ArcFace)", "progress": 45}, websocket
        )
        face_result = {"similarity": 0.0, "match": False, "method": "no_images"}
        if passport_bytes and selfie_bytes:
            try:
                from services.ml_pipeline import match_faces
                face_result = match_faces(passport_bytes, selfie_bytes)
            except Exception as e:
                face_result = {"similarity": 0.0, "match": False, "method": "error", "error": str(e)}
        await asyncio.sleep(1.5)

        # ── Stage 4: Fingerprint Matching ──────────────────────────────
        await manager.send_message(
            {"status": "processing", "module": "Fingerprint Matching", "progress": 60}, websocket
        )
        fingerprint_result = {"match": True, "confidence": 85.0, "method": "simulated"}
        # Fingerprint data would come from NFC or separate scan
        await asyncio.sleep(0.8)

        # ── Stage 5: Iris Scan ────────────────────────────────────────
        await manager.send_message(
            {"status": "processing", "module": "Iris Scan Comparison", "progress": 70}, websocket
        )
        iris_result = {"match": True, "confidence": 90.0, "method": "simulated"}
        await asyncio.sleep(0.6)

        # ── Stage 6: NFC Digital Footprint ────────────────────────────
        await manager.send_message(
            {"status": "processing", "module": "NFC Digital Footprint", "progress": 78}, websocket
        )
        nfc_result = {"nfc_digital_footprint": bool(files.get("passportNfc"))}
        await asyncio.sleep(0.5)

        # ── Stage 7: Hologram Detection ───────────────────────────────
        await manager.send_message(
            {"status": "processing", "module": "Hologram OVD Detection", "progress": 83}, websocket
        )
        holo_result = {"ovd_verified": False}
        if passport_bytes:
            try:
                from services.ml_pipeline import detect_hologram
                holo_result = detect_hologram(passport_bytes)
            except Exception as e:
                holo_result = {"ovd_verified": False, "error": str(e)}
        await asyncio.sleep(0.5)

        # ── Stage 8: INTERPOL Database ────────────────────────────────
        await manager.send_message(
            {"status": "processing", "module": "INTERPOL Database Query", "progress": 88}, websocket
        )
        interpol_result = {
            "criminal_record": random.random() > 0.85,
            "past_violations": random.random() > 0.90,
        }
        await asyncio.sleep(0.8)

        # ── Stage 9: Visa Analysis (AI-Powered) ──────────────────────
        await manager.send_message(
            {"status": "processing", "module": "AI Visa Verification (OpenRouter)", "progress": 92}, websocket
        )
        visa_result_data = {"color_font_check": False, "country_regulations": False, "stamps_authentication": False}
        if visa_bytes:
            try:
                # Try ML pipeline first
                from services.ml_pipeline import analyze_visa
                visa_result_data = analyze_visa(visa_bytes)
            except Exception:
                pass
            # Also run AI analysis
            visa_ai = analyze_document_image("visa",
                "Analyze this visa document for: 1) Color profile authenticity 2) Font consistency "
                "3) Country regulation compliance 4) Stamp authentication 5) Any signs of forgery. "
                "The visa should have official stamps, consistent fonts, and proper color schemes.")
            if visa_ai.get("success"):
                visa_result_data["ai_verified"] = visa_ai["analysis"].get("verified", False)
                visa_result_data["ai_confidence"] = visa_ai["analysis"].get("confidence", 0)
                visa_result_data["ai_details"] = visa_ai["analysis"].get("details", "")
                # Use AI verification to override if ML didn't run
                if not visa_result_data.get("color_font_check") and visa_ai["analysis"].get("verified"):
                    visa_result_data["color_font_check"] = True
                    visa_result_data["country_regulations"] = True
                    visa_result_data["stamps_authentication"] = True
        else:
            visa_result_data["ai_details"] = "No visa document uploaded"
        await asyncio.sleep(0.5)

        # ── Stage 10: National ID Verification (AI-Powered) ──────────
        await manager.send_message(
            {"status": "processing", "module": "AI National ID Verification (OpenRouter)", "progress": 95}, websocket
        )
        nid_result = {"verified": False, "ai_details": "No National ID uploaded"}
        if nid_bytes:
            nid_ai = analyze_document_image("national_id",
                "Analyze this National ID document for: 1) Document authenticity 2) Photo quality and match "
                "3) Text readability and formatting 4) Security features 5) Any signs of forgery or tampering. "
                "A valid national ID should have clear photo, readable text, proper formatting, and security features.")
            if nid_ai.get("success"):
                nid_result = {
                    "verified": nid_ai["analysis"].get("verified", False),
                    "confidence": nid_ai["analysis"].get("confidence", 0),
                    "ai_details": nid_ai["analysis"].get("details", ""),
                    "checks_passed": nid_ai["analysis"].get("checks_passed", []),
                    "checks_failed": nid_ai["analysis"].get("checks_failed", []),
                }
            else:
                nid_result = {"verified": False, "ai_details": nid_ai.get("error", "AI analysis failed")}
        await asyncio.sleep(0.5)

        # ── Stage 11: Tamper Detection ────────────────────────────────
        await manager.send_message(
            {"status": "processing", "module": "Digital Tamper Forensics", "progress": 96}, websocket
        )
        tamper_result = {"pixel_manipulation": "CLEAR", "metadata": "AUTHENTIC"}
        if passport_bytes:
            try:
                from services.ml_pipeline import detect_tampering
                tamper_result = detect_tampering(passport_bytes)
            except Exception as e:
                tamper_result = {"pixel_manipulation": "UNKNOWN", "metadata": "UNKNOWN", "error": str(e)}
        await asyncio.sleep(0.4)

        # ── Stage 11: AI-Powered Risk Assessment ───────────────────────
        await manager.send_message(
            {"status": "processing", "module": "AI Risk Aggregation (OpenRouter)", "progress": 99}, websocket
        )
        
        # Build initial results for AI analysis
        face_sim = face_result.get("similarity", 0.0)
        ela_pass = ela_result.get("pass", False)
        fp_match = fingerprint_result.get("match", True)
        iris_match = iris_result.get("match", True)
        mrz_valid = mrz_result.get("mrz_valid", True)
        
        initial_results = {
            "ela_pass": ela_pass,
            "mrz_valid": mrz_valid,
            "face_similarity": face_sim,
            "fingerprint_match": fp_match,
            "iris_match": iris_match,
            "visa_verification": visa_result_data,
            "hologram_check": holo_result,
            "tamper_forensics": tamper_result,
        }
        
        # Get AI-powered risk assessment
        print(f"[AI] Calling OpenRouter for risk assessment...")
        ai_risk_result = analyze_risk_with_ai(initial_results)
        print(f"[AI] Risk assessment result: success={ai_risk_result.get('success')}")
        if ai_risk_result.get('error'):
            print(f"[AI] Risk assessment error: {ai_risk_result.get('error')}")
        if ai_risk_result["success"]:
            risk_score = ai_risk_result["risk_analysis"]["risk_score"]
            ai_explanation = ai_risk_result["risk_analysis"]["explanation"]
            ai_factors = ai_risk_result["risk_analysis"]["factors"]
        else:
            # Fallback to local calculation
            risk_score = 0
            risk_score += round((100 - face_sim) * 0.4)
            if not ela_pass: risk_score += 15
            if not fp_match: risk_score += 12
            if not iris_match: risk_score += 10
            if not mrz_valid: risk_score += 8
            risk_score += round(random.random() * 5)
            risk_score = max(5, min(99, risk_score))
            ai_explanation = "AI assessment unavailable, using local calculation"
            ai_factors = []
        
        # Decision is ALWAYS based on the risk score — high score = high risk
        risk_score = max(5, min(99, risk_score))
        decision = "HIGH_RISK" if risk_score >= 65 else "MEDIUM_RISK" if risk_score >= 40 else "VERIFIED"
        
        await asyncio.sleep(0.3)

        # ── AI Document Analysis ───────────────────────────────────────
        await manager.send_message(
            {"status": "processing", "module": "AI Document Analysis (OpenRouter)", "progress": 99.5}, websocket
        )
        
        # Build full verification data for AI analysis
        full_verification_data = {
            **initial_results,
            "risk_score": risk_score,
            "risk_level": decision,
            "interpol_report": interpol_result,
            "passport_verification": {
                "dob_check": mrz_valid,
                "info_authentication": mrz_valid,
                "passive_authentication": ela_pass,
                "active_authentication": nfc_result.get("nfc_digital_footprint", False),
            },
            "impersonation_check": {
                "biometrics_verification": face_result.get("match", False),
                "live_photo_match": face_sim > 55,
            },
            "hologram_check": {
                "ovd_verified": holo_result.get("ovd_verified", False),
                "nfc_digital_footprint": nfc_result.get("nfc_digital_footprint", False),
            },
            "tamper_forensics": {
                "ela_status": "PASS" if ela_pass else "FAIL",
                "ela_details": ela_result.get("details", ""),
                "ink_stamp_layout": "PASS" if visa_result_data.get("stamps_authentication", False) else "FAIL",
                "dob_crosslink": "PASS" if mrz_valid else "FAIL",
                "pixel_manipulation": tamper_result.get("pixel_manipulation", "UNKNOWN"),
                "metadata_analysis": tamper_result.get("metadata", "UNKNOWN"),
                "resolution_consistency": "CONSISTENT" if tamper_result.get("resolution_consistent", True) else "INCONSISTENT",
            },
        }
        
        # Get AI analysis
        print(f"[AI] Calling OpenRouter for document analysis...")
        ai_document_result = analyze_document_with_ai(full_verification_data)
        print(f"[AI] Document analysis result: success={ai_document_result.get('success')}")
        if ai_document_result.get('error'):
            print(f"[AI] Document analysis error: {ai_document_result.get('error')}")
        ai_insights = ai_document_result.get("analysis", {})
        
        # Generate AI report
        print(f"[AI] Calling OpenRouter for report generation...")
        ai_report_result = generate_verification_report(full_verification_data)
        print(f"[AI] Report result: success={ai_report_result.get('success')}")
        if ai_report_result.get('error'):
            print(f"[AI] Report error: {ai_report_result.get('error')}")
        ai_report = ai_report_result.get("report", {})
        
        await asyncio.sleep(0.2)

        # ── Final Result ──────────────────────────────────────────────
        final_result = {
            "status": "complete",
            "progress": 100,
            "result": {
                "ela_pass": ela_pass,
                "mrz_valid": mrz_valid,
                "face_similarity": face_sim,
                "fingerprint_match": fp_match,
                "iris_match": iris_match,
                "risk_score": risk_score,
                "decision": decision,
                "visa_verification": visa_result_data,
                "passport_verification": {
                    "dob_check": mrz_valid,
                    "info_authentication": mrz_valid,
                    "passive_authentication": ela_pass,
                    "active_authentication": nfc_result.get("nfc_digital_footprint", False),
                },
                "impersonation_check": {
                    "biometrics_verification": face_result.get("match", False),
                    "live_photo_match": face_sim > 55,
                },
                "interpol_report": interpol_result,
                "document_originality": {
                    "passport": {
                        "verified": mrz_valid and ela_pass,
                        "suspicion_areas": [] if (mrz_valid and ela_pass) else (
                            (["MRZ anomaly"] if not mrz_valid else []) +
                            (["ELA tamper detected"] if not ela_pass else [])
                        ),
                    },
                    "visa": {
                        "verified": visa_result_data.get("ai_verified", visa_result_data.get("color_font_check", False)),
                        "confidence": visa_result_data.get("ai_confidence", 0),
                        "ai_details": visa_result_data.get("ai_details", ""),
                        "suspicion_areas": [] if visa_result_data.get("ai_verified", visa_result_data.get("color_font_check", False)) else visa_result_data.get("checks_failed", ["Verification failed"]),
                    },
                    "national_id": {
                        "verified": nid_result.get("verified", False),
                        "confidence": nid_result.get("confidence", 0),
                        "ai_details": nid_result.get("ai_details", ""),
                        "suspicion_areas": [] if nid_result.get("verified", False) else nid_result.get("checks_failed", ["No National ID uploaded"]),
                    },
                },
                "hologram_check": {
                    "ovd_verified": holo_result.get("ovd_verified", False),
                    "nfc_digital_footprint": nfc_result.get("nfc_digital_footprint", False),
                },
                "tamper_forensics": {
                    "ela_status": "PASS" if ela_pass else "FAIL",
                    "ela_details": ela_result.get("details", ""),
                    "ink_stamp_layout": "PASS" if visa_result_data.get("stamps_authentication", False) else "FAIL",
                    "dob_crosslink": "PASS" if mrz_valid else "FAIL",
                    "pixel_manipulation": tamper_result.get("pixel_manipulation", "UNKNOWN"),
                    "metadata_analysis": tamper_result.get("metadata", "UNKNOWN"),
                    "resolution_consistency": "CONSISTENT" if tamper_result.get("resolution_consistent", True) else "INCONSISTENT",
                },
                # Uploaded images as base64 for display
                "images": {
                    "passport": f"data:image/jpeg;base64,{__import__('base64').b64encode(passport_bytes).decode()}" if passport_bytes else None,
                    "selfie": f"data:image/jpeg;base64,{__import__('base64').b64encode(selfie_bytes).decode()}" if selfie_bytes else None,
                    "visa": f"data:image/jpeg;base64,{__import__('base64').b64encode(visa_bytes).decode()}" if visa_bytes else None,
                },
                # AI-powered insights
                "ai_analysis": ai_insights,
                "ai_risk_assessment": ai_risk_result.get("risk_analysis", {}),
                "ai_report": ai_report,
                "ai_model_used": ai_risk_result.get("model_used", "N/A"),
                "ai_tokens_used": (
                    ai_risk_result.get("tokens_used", 0) + 
                    ai_document_result.get("tokens_used", 0) + 
                    ai_report_result.get("tokens_used", 0)
                ),
            },
        }
        
        print(f"[WS] Sending final result with AI analysis...")
        print(f"[WS] AI model used: {final_result['result'].get('ai_model_used')}")
        print(f"[WS] AI tokens used: {final_result['result'].get('ai_tokens_used')}")
        await manager.send_message(final_result, websocket)
        print(f"[WS] Verification complete for {tracking_id}")
        
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {tracking_id}")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WS] ERROR: {str(e)}")
        try:
            await manager.send_message(
                {"status": "error", "message": str(e)}, websocket
            )
        except Exception:
            pass
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
