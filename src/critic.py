import json
import os
from typing import Dict, Any, List

class SimulatedDoctorReviewer:
    """
    A simulated clinical reviewer that applies a consistent, hidden preference policy
    to the agent's generated drafts, outputting a 'ground-truth corrected' pair.
    """
    def __init__(self, memory_file_path: str = "data/feedback_memory.jsonl"):
        self.memory_file_path = memory_file_path

    def apply_hidden_clinical_policy(self, draft_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies a predictable set of stylistic and structural revisions:
        1. Explodes standard clinical shorthand (PO -> orally, BID -> twice daily).
        2. Force-bulletizes long paragraph structures in the 'hospital_course' block.
        3. Standardizes structural typography for diagnoses.
        """
        # Deep copy to leave original draft intact for distance calculations
        corrected_payload = json.loads(json.dumps(draft_payload))
        
        # Policy Rule 1: Bulletize long paragraphs in Hospital Course
        hc = corrected_payload.get("hospital_course", "")
        if hc and "\n*" not in hc and "•" not in hc and len(hc) > 100:
            sentences = hc.split(". ")
            bulleted_hc = "\n".join([f"• {s.strip()}" for s in sentences if s.strip()])
            corrected_payload["hospital_course"] = bulleted_hc

        # Policy Rule 2: Expand Medication shorthand strings safely
        meds = corrected_payload.get("discharge_medications", [])
        if isinstance(meds, list):
            for med in meds:
                if isinstance(med, dict):
                    freq = med.get("frequency", "")
                    # Expand common systemic acronyms
                    freq = freq.replace("1-0-1", "twice daily").replace("1-1-1", "three times daily")
                    freq = freq.replace("1-0-0", "once daily in the morning").replace("0-0-1", "once daily at bedtime")
                    med["frequency"] = freq
                    
                    dosage = med.get("dosage", "")
                    med["dosage"] = dosage.upper()

        # Policy Rule 3: Enforce upper-casing for principal diagnoses
        pd = corrected_payload.get("principal_diagnosis", "")
        if pd and not pd.isupper():
            corrected_payload["principal_diagnosis"] = pd.upper()

        return corrected_payload

    def commit_edit_to_long_term_memory(self, original_draft: dict, corrected_version: dict, rewards: float):
        """
        Appends the (draft, corrected, reward) triplet to the local memory lines.
        This provides the dataset required to demonstrate your optimization curve.
        """
        log_entry = {
            "reward_signal": rewards,
            "original_draft_snippet": {
                "principal_diagnosis": original_draft.get("principal_diagnosis"),
                "hospital_course": original_draft.get("hospital_course")[:150] if original_draft.get("hospital_course") else ""
            },
            "corrected_version_snippet": {
                "principal_diagnosis": corrected_version.get("principal_diagnosis"),
                "hospital_course": corrected_version.get("hospital_course")[:150] if corrected_version.get("hospital_course") else ""
            }
        }
        
        with open(self.memory_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def compile_few_shot_context_buffer(self) -> str:
        """
        Reads accumulated memory history to build a dynamic engineering context block.
        This forces future agent loops to adapt to preference paths without fine-tuning.
        """
        if not os.path.exists(self.memory_file_path):
            return "No prior correction history logged."

        memory_context = []
        try:
            with open(self.memory_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-3:] # Grab the last 3 corrections to fit token budgets
                for idx, line in enumerate(lines):
                    data = json.loads(line.strip())
                    memory_context.append(
                        f"Reviewer Edit Event #{idx+1} (Prior Accuracy Score: {data['reward_signal']}):\n"
                        f"  - What you generated: {json.dumps(data['original_draft_snippet'])}\n"
                        f"  - What the clinician demanded: {json.dumps(data['corrected_version_snippet'])}\n"
                    )
            return "\n".join(memory_context)
        except Exception:
            return "Failed to extract active memory buffers."