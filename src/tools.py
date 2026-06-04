import os
from typing import List, Dict, Any
import pdfplumber

def read_pdf_content(file_path: str) -> Dict[str, Any]:
    """
    Reads and extracts unstructured text and structured table data from patient PDFs.
    Implements strict error handling to ensure a missing or corrupted file 
    returns a controlled failure message to the agent loop instead of crashing.
    """
    if not os.path.exists(file_path):
        return {
            "status": "error",
            "message": f"CRITICAL TOOL FAILURE: Document not found at path '{file_path}'."
        }
    
    try:
        extracted_text = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    # Append structural source-tags matching real medical records
                    extracted_text.append(f"--- PAGE {page_num} ---\n{text}")
                
                # Extract structural tables to capture layout-heavy lab grids
                tables = page.extract_tables()
                for table in tables:
                    extracted_text.append(f"[Table Data Format]: {str(table)}")
                    
        return {
            "status": "success",
            "content": "\n\n".join(extracted_text)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"CRITICAL TOOL FAILURE: Failed parsing PDF content. Internal error: {str(e)}"
        }

def drug_interaction_lookup(medications: List[str]) -> Dict[str, Any]:
    """
    Mock external clinical drug interaction registry database lookup.
    Actively flags dangerous therapeutic pairings to force agent awareness.
    """
    if not medications:
        return {"status": "success", "warnings": [], "message": "No medications provided for interaction screening."}
    
    meds_lower = [m.lower() for m in medications]
    warnings = []
    
    # Standard dangerous clinical cross-checking pairs
    if any("warfarin" in m for m in meds_lower) and any("aspirin" in m for m in meds_lower):
        warnings.append("CRITICAL SAFETY WARNING: Concurrent use of Warfarin and Aspirin exponentially increases major bleeding risks.")
    if any("meropenem" in m for m in meds_lower) and any("valproic" in m for m in meds_lower):
        warnings.append("CLINICAL ALERT: Meropenem drastically reduces serum Valproic Acid concentrations, risking breakthrough seizures.")
        
    return {
        "status": "success",
        "warnings": warnings,
        "checked_count": len(medications)
    }

def escalate_to_clinician(reason: str, missing_field: str) -> Dict[str, Any]:
    """
    Explicitly halts autonomous generation paths for unverified sections.
    Passes clinical context to human-in-the-loop validation queue.
    """
    return {
        "status": "ESCALATED",
        "escalation_summary": f"MANUAL INTERVENTION REQ: [{missing_field}]. Reason: {reason}",
        "action_required": "Clinician must manually review raw patient records to extract this entity safely."
    }