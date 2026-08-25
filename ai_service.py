"""
CodeCortex AI Service - OpenRouter Integration
===============================================
Uses OpenRouter API to provide real AI-powered analysis for document verification.
"""

import os
import json
import numpy as np
from openai import OpenAI


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Initialize OpenAI client with OpenRouter
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

# Model selection - using a capable but cost-effective model
MODEL = "google/gemini-2.5-flash"


def analyze_document_with_ai(document_data: dict) -> dict:
    """
    Use AI to analyze document verification results and provide detailed insights.
    
    Args:
        document_data: Dictionary containing verification results from various modules
        
    Returns:
        AI-generated analysis with risk assessment and recommendations
    """
    try:
        # Build context for the AI
        context = f"""
You are an expert document verification analyst. Analyze the following verification results and provide:
1. Overall risk assessment
2. Key findings and concerns
3. Confidence level in the verification
4. Recommended actions

Verification Results:
{json.dumps(document_data, indent=2, cls=NumpyEncoder)}

Provide your analysis in JSON format with the following structure:
{{
    "risk_assessment": "LOW/MEDIUM/HIGH/CRITICAL",
    "confidence_score": 0-100,
    "key_findings": ["finding1", "finding2", ...],
    "concerns": ["concern1", "concern2", ...],
    "recommendations": ["recommendation1", "recommendation2", ...],
    "ai_insight": "detailed analysis text"
}}
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a document verification expert AI. Always respond in valid JSON format."},
                {"role": "user", "content": context}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        # Parse the AI response
        ai_response = response.choices[0].message.content
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            start = ai_response.find('{')
            end = ai_response.rfind('}') + 1
            if start != -1 and end != -1:
                analysis = json.loads(ai_response[start:end])
            else:
                analysis = {
                    "risk_assessment": "MEDIUM",
                    "confidence_score": 70,
                    "key_findings": ["AI analysis completed"],
                    "concerns": [],
                    "recommendations": ["Manual review recommended"],
                    "ai_insight": ai_response
                }
        except json.JSONDecodeError:
            analysis = {
                "risk_assessment": "MEDIUM",
                "confidence_score": 70,
                "key_findings": ["AI response parsing failed"],
                "concerns": [],
                "recommendations": ["Manual review recommended"],
                "ai_insight": ai_response
            }
        
        return {
            "success": True,
            "analysis": analysis,
            "model_used": MODEL,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "analysis": {
                "risk_assessment": "UNKNOWN",
                "confidence_score": 0,
                "key_findings": [f"AI service error: {str(e)}"],
                "concerns": ["AI analysis unavailable"],
                "recommendations": ["Proceed with manual verification"],
                "ai_insight": f"AI service encountered an error: {str(e)}"
            }
        }


def analyze_risk_with_ai(verification_results: dict) -> dict:
    """
    Use AI to calculate and explain risk scores based on verification results.
    
    Args:
        verification_results: Complete verification results from all modules
        
    Returns:
        AI-enhanced risk assessment with detailed explanation
    """
    try:
        context = f"""
Based on the following document verification results, calculate a comprehensive risk score and explain your reasoning.

Results:
- ELA (Error Level Analysis): {verification_results.get('ela_pass', 'N/A')}
- MRZ Validation: {verification_results.get('mrz_valid', 'N/A')}
- Face Similarity: {verification_results.get('face_similarity', 'N/A')}%
- Fingerprint Match: {verification_results.get('fingerprint_match', 'N/A')}
- Iris Match: {verification_results.get('iris_match', 'N/A')}

Provide a JSON response:
{{
    "risk_score": 0-100,
    "risk_level": "LOW/MEDIUM/HIGH/CRITICAL",
    "explanation": "detailed explanation of the risk assessment",
    "factors": [
        {{"factor": "factor_name", "impact": "positive/negative", "weight": 0-1, "description": "explanation"}}
    ],
    "confidence": 0-100
}}
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a risk assessment expert. Always respond in valid JSON format."},
                {"role": "user", "content": context}
            ],
            temperature=0.2,
            max_tokens=800,
        )
        
        ai_response = response.choices[0].message.content
        
        try:
            start = ai_response.find('{')
            end = ai_response.rfind('}') + 1
            if start != -1 and end != -1:
                risk_analysis = json.loads(ai_response[start:end])
            else:
                risk_analysis = {
                    "risk_score": 50,
                    "risk_level": "MEDIUM",
                    "explanation": "AI response parsing failed",
                    "factors": [],
                    "confidence": 50
                }
        except json.JSONDecodeError:
            risk_analysis = {
                "risk_score": 50,
                "risk_level": "MEDIUM",
                "explanation": "AI response parsing failed",
                "factors": [],
                "confidence": 50
            }
        
        return {
            "success": True,
            "risk_analysis": risk_analysis,
            "model_used": MODEL,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "risk_analysis": {
                "risk_score": 50,
                "risk_level": "MEDIUM",
                "explanation": f"AI service error: {str(e)}",
                "factors": [],
                "confidence": 0
            }
        }


def analyze_document_image(document_type: str, image_description: str) -> dict:
    """
    Use AI to analyze a document image and determine if it's authentic.
    
    Args:
        document_type: Type of document (visa, national_id, passport)
        image_description: Description of what the AI should look for
        
    Returns:
        AI-generated analysis with authenticity assessment
    """
    try:
        context = f"""
You are a document verification expert analyzing a {document_type} document.

Analyze the following verification checks and determine if the document is authentic:
{image_description}

Provide a JSON response:
{{
    "verified": true/false,
    "confidence": 0-100,
    "checks_passed": ["check1", "check2"],
    "checks_failed": ["check1"],
    "details": "detailed analysis"
}}
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a document verification expert. Always respond in valid JSON format."},
                {"role": "user", "content": context}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        
        ai_response = response.choices[0].message.content
        
        try:
            start = ai_response.find('{')
            end = ai_response.rfind('}') + 1
            if start != -1 and end != -1:
                analysis = json.loads(ai_response[start:end])
            else:
                analysis = {"verified": False, "confidence": 0, "details": "AI response parsing failed"}
        except json.JSONDecodeError:
            analysis = {"verified": False, "confidence": 0, "details": "AI response parsing failed"}
        
        return {
            "success": True,
            "analysis": analysis,
            "model_used": MODEL,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "analysis": {"verified": False, "confidence": 0, "details": f"AI error: {str(e)}"}
        }


def generate_verification_report(verification_results: dict) -> dict:
    """
    Generate a comprehensive verification report using AI.
    
    Args:
        verification_results: Complete verification results
        
    Returns:
        AI-generated report with summary and recommendations
    """
    try:
        context = f"""
Generate a professional document verification report based on the followingresults:
{json.dumps(verification_results, indent=2, cls=NumpyEncoder)}

The report should include:
1. Executive summary
2. Document authenticity assessment
3. Biometric verification status
4. Security features verification
5. Risk factors identified
6. Final recommendation

Format as JSON:
{{
    "report_title": "Document Verification Report",
    "executive_summary": "brief summary",
    "document_authenticity": {{
        "status": "VERIFIED/SUSPICIOUS/REJECTED",
        "details": "explanation"
    }},
    "biometric_verification": {{
        "face_match": "PASS/FAIL",
        "fingerprint": "PASS/FAIL/NOT_AVAILABLE",
        "iris": "PASS/FAIL/NOT_AVAILABLE"
    }},
    "security_features": {{
        "ela_analysis": "PASS/FAIL",
        "hologram": "DETECTED/NOT_DETECTED",
        "mrz": "VALID/INVALID"
    }},
    "risk_factors": ["factor1", "factor2"],
    "recommendation": "APPROVE/REJECT/REVIEW_REQUIRED",
    "additional_notes": "any other observations"
}}
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a document verification report generator. Always respond in valid JSON format."},
                {"role": "user", "content": context}
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        
        ai_response = response.choices[0].message.content
        
        try:
            start = ai_response.find('{')
            end = ai_response.rfind('}') + 1
            if start != -1 and end != -1:
                report = json.loads(ai_response[start:end])
            else:
                report = {
                    "report_title": "Document Verification Report",
                    "executive_summary": "AI report generation incomplete",
                    "recommendation": "REVIEW_REQUIRED"
                }
        except json.JSONDecodeError:
            report = {
                "report_title": "Document Verification Report",
                "executive_summary": "AI report generation incomplete",
                "recommendation": "REVIEW_REQUIRED"
            }
        
        return {
            "success": True,
            "report": report,
            "model_used": MODEL,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "report": {
                "report_title": "Document Verification Report",
                "executive_summary": f"Report generation failed: {str(e)}",
                "recommendation": "REVIEW_REQUIRED"
            }
        }
