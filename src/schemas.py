from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentDecision(BaseModel):
    """Enforces the ReAct pattern structure on every single iteration."""
    thought: str = Field(
        description="Detailed internal clinical reasoning and evaluation of what to do next."
    )
    tool_name: str = Field(
        description="The precise name of the action to execute: 'read_pdf', 'lookup_drugs', 'escalate', or 'finalize'."
    )
    tool_input: Dict[str, Any] = Field(
        default_factory=dict,
        description="The key-value parameters to pass directly to the chosen tool."
    )

class DischargeSummaryDraft(BaseModel):
    """The final structured document model required for clinical safety sign-off."""
    patient_demographics: Dict[str, Any] = Field(description="Name, age, gender, admission dates, discharge dates.")
    principal_diagnosis: str = Field(description="Primary clinical diagnosis clearly backed by reports.")
    secondary_diagnoses: List[str] = Field(description="Comorbidities, active, and secondary problems tracked.")
    hospital_course: str = Field(description="Chronological clinical summary of stabilization, ward course, and transition.")
    procedures_performed: List[str] = Field(description="List of all lines, scopes, scans, or surgeries performed.")
    discharge_medications: List[Dict[str, Any]] = Field(description="Name, dosage, frequency, duration, and reconciliation flags.")
    allergies: str = Field(description="Documented adversities. Must state 'Not Known' if missing.")
    pending_results: List[str] = Field(description="Explicitly tracks tests sent but unreturned at time of discharge.")
    discharge_condition: str = Field(description="Hemodynamic stability metrics.")
    clinical_flags_for_review: List[str] = Field(description="Explicit conflicts or unverified adjustments flagged for doctors.")