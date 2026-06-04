import os
import json
import yaml
from openai import OpenAI
from dotenv import load_dotenv
from src.schemas import AgentDecision, DischargeSummaryDraft
from src.tools import read_pdf_content, drug_interaction_lookup, escalate_to_clinician

load_dotenv()

class DischargeAgentController:
    """
    An autonomous ReAct AI Agent Controller that evaluates unstructured medical notes 
    and designs a safe, non-fabricated discharge summary draft.
    """
    def __init__(self, patient_id: str, pdf_paths: list):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.patient_id = patient_id
        self.pdf_paths = pdf_paths
        
        # Core State Machine Store Variables
        self.extracted_text_store = {}
        self.trace_logs = []
        self.iteration_count = 0
        self.max_iterations = 12  # Hard requirements rule 9 (Control)
        self.correction_memory_cache = "No prior corrective feedback logged in context."

    def load_prompts(self) -> dict:
        """Loads non-hardcoded foundational system prompts and instruction matrices."""
        with open("config/prompts.yaml", "r") as f:
            return yaml.safe_load(f)

    def update_agent_memory_buffer(self, critic_engine):
        """Injects accumulated historical doctor edits (Part 2 Feedback loop) into current prompt run context."""
        self.correction_memory_cache = critic_engine.compile_few_shot_context_buffer()

    def run_loop(self) -> dict:
        """
        Executes the autonomous planning, reading, tool execution, and reconciliation 
        loop. Returns the full step trace and finalized clinical JSON schema payload.
        """
        prompts = self.load_prompts()
        
        # Setup foundation baseline state injection environment
        system_base = prompts["agent_system_prompt"].format(
            file_list=str(self.pdf_paths),
            correction_memory=self.correction_memory_cache
        )
        
        messages = [
            {"role": "system", "content": system_base},
            {
                "role": "user", 
                "content": "Begin analysis. Read files sequentially using the appropriate tools. Carefully check for conflicting diagnoses or unverified medication alterations."
            }
        ]
        
        print(f"🏁 Starting Agentic ReAct Engine Loop for {self.patient_id}...")

        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            
            # --- STEP A: GENERATE REASONING THOUGHT AND ACTION PLAN ---
            try:
                # Force structured JSON format parsing based on our AgentDecision schema bounds
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1 # Low variance to enforce strict clinical focus paths
                )
                
                decision_raw = json.loads(response.choices[0].message.content)
                decision = AgentDecision(**decision_raw)
                
            except Exception as e:
                # Rule 8: Robust Failure Handling (LLM parse errors cannot cause system failure crashes)
                error_trace = {
                    "step": self.iteration_count,
                    "thought": "Internal parse error detected during LLM response interpretation. Actively generating dynamic retry boundary.",
                    "tool_name": "retry_fallback",
                    "tool_input": {"error": str(e)}
                }
                self.trace_logs.append(error_trace)
                messages.append({"role": "user", "content": f"Formatting parse error encountered: {str(e)}. Please correct your JSON layout structure immediately."})
                continue

            # --- STEP B: LOG DETAILED OBSERVABILITY TRACE (Rule 10) ---
            step_log = {
                "step": self.iteration_count,
                "thought": decision.thought,
                "tool_chosen": decision.tool_name,
                "inputs_passed": decision.tool_input
            }
            self.trace_logs.append(step_log)
            
            print(f"👉 Step {self.iteration_count} | Tool: {decision.tool_name} | Thought Snippet: {decision.thought[:65]}...")

            # Sync current action decision step execution to conversational history store context
            messages.append({"role": "assistant", "content": json.dumps(decision_raw)})

            # --- STEP C: DYNAMIC TOOL ROUTING BRANCH SELECTION ---
            if decision.tool_name == "read_pdf":
                file_path = decision.tool_input.get("file_path")
                
                # Rule 2 & 8: Process layout and tables via custom pdfplumber tool infrastructure safely
                tool_output = read_pdf_content(file_path)
                self.extracted_text_store[file_path] = tool_output
                
                # Inject findings back to prompt state as observations
                messages.append({
                    "role": "user", 
                    "content": f"Observation result for read_pdf tool: {json.dumps(tool_output)}"
                })

            elif decision.tool_name == "lookup_drugs":
                med_list = decision.tool_input.get("medications", [])
                
                # Rule 7: Consult registry system directly and assert warnings
                tool_output = drug_interaction_lookup(med_list)
                messages.append({
                    "role": "user", 
                    "content": f"Observation result for lookup_drugs database verification: {json.dumps(tool_output)}"
                })

            elif decision.tool_name == "escalate":
                reason = decision.tool_input.get("reason")
                field = decision.tool_input.get("missing_field")
                
                # Rule 3 & 4: Stop auto-completion paths and tag anomalies to manual reviews
                tool_output = escalate_to_clinician(reason, field)
                messages.append({
                    "role": "user", 
                    "content": f"Human-In-The-Loop Verification Requested. Escalation confirmed: {json.dumps(tool_output)}"
                })

            elif decision.tool_name == "finalize":
                # Final evaluation checkpoint verification assertion
                draft_data = decision.tool_input.get("draft_payload")
                
                try:
                    # Enforce rigid type constraints on output document using Pydantic Validation schemas
                    validated_draft = DischargeSummaryDraft(**draft_data)
                    return {
                        "status": "SUCCESS",
                        "trace_history": self.trace_logs,
                        "final_draft": validated_draft.model_dump()
                    }
                except Exception as schema_err:
                    messages.append({
                        "role": "user", 
                        "content": f"Structure validation mapping error encountered on final draft initialization: {str(schema_err)}. Correct the properties format."
                    })
                    continue
            
            else:
                messages.append({
                    "role": "user", 
                    "content": f"Unknown tool execution request code block '{decision.tool_name}'. Select an active systemic tool resource asset."
                })

        # --- STEP D: CRITICAL BOUNDARY BLOWOUT SAFE FALLBACK RESOLUTION ---
        print("⚠️ Safety Threshold: Maximum orchestration iteration limit exceeded without resolving final boundaries safely.")
        emergency_safety_payload = {
            "patient_demographics": {"status": "[[MISSING - ESCALATED FOR REVIEW]]"},
            "principal_diagnosis": "[[UNRESOLVED CRITICAL CONTEXT ERROR - MANUAL ASSIGNMENT REQ]]",
            "secondary_diagnoses": [],
            "hospital_course": "Automated generation timeline context loop exceeded standard execution safety steps without arriving at a verified conclusion. Manual assembly required.",
            "procedures_performed": [],
            "discharge_medications": [],
            "allergies": "Not Known",
            "pending_results": ["All pending diagnostics deferred to manual physician tracking protocols."],
            "discharge_condition": "Unverified",
            "clinical_flags_for_review": ["CRITICAL STATE BLOWOUT: Controller process execution loop terminated due to iteration threshold limits rule."]
        }
        
        return {
            "status": "TIMEOUT_CRITICAL_ESC",
            "trace_history": self.trace_logs,
            "final_draft": emergency_safety_payload
        }