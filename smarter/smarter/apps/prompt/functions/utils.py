
import os


def should_log(level: int) -> bool:
    enabled = os.getenv("SMARTER_PROMPT_FUNCTION_LOGGING", "1")
    return enabled.strip().lower() not in {"0", "false", "off", "no"}
