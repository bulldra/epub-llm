import os
import json
from typing import List, Dict, Union, Optional

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "../cache/history")


def save_history(session_id: str, history: List[Dict[str, Union[str, None]]]) -> None:
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_history(
    session_id: str,
) -> Optional[List[Dict[str, Union[str, None]]]]:
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_histories() -> List[str]:
    return [f[:-5] for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]
