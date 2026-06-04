import Levenshtein

def calculate_edit_distance_reward(agent_draft: str, doctor_edited: str) -> float:
    """
    Calculates a normalized reward signal based on character and token edit distance.
    Formula: Reward = 1.0 - (Levenshtein_Distance / Max_Length)
    A higher reward means less editing was required by the clinician.
    """
    if not agent_draft and not doctor_edited:
        return 1.0
        
    # Standardize whitespace to ensure formatting tweaks don't skew the medical accuracy signal
    draft_clean = " ".join(agent_draft.strip().split())
    edited_clean = " ".join(doctor_edited.strip().split())
    
    max_len = max(len(draft_clean), len(edited_clean))
    if max_len == 0:
        return 1.0
        
    distance = Levenshtein.distance(draft_clean, edited_clean)
    
    # Calculate normalized score bounded strictly between 0.0 and 1.0
    reward = 1.0 - (distance / max_len)
    return round(max(0.0, reward), 4)

def calculate_section_match_rates(draft_dict: dict, edited_dict: dict) -> dict:
    """
    Computes a granular breakdown of alignment across critical schema sections.
    """
    section_rewards = {}
    for key in edited_dict.keys():
        draft_sec = str(draft_dict.get(key, ""))
        edited_sec = str(edited_dict.get(key, ""))
        section_rewards[key] = calculate_edit_distance_reward(draft_sec, edited_sec)
    return section_rewards