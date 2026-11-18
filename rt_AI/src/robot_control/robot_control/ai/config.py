"""Configuration manager for the Autonomous AI multi-robot platform.

Loads configuration from environment variables and optional .env file.
Supports hybrid modes:
- cloud: OpenAI / Anthropic / Gemini API
- local: Local Ollama server (e.g. http://localhost:11434)
- deterministic: Offline rule-based kinematic intent parser and trajectory generator
- hybrid: Auto-detects cloud/local server and falls back transparently to deterministic
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _parse_env_file(filepath: str) -> dict:
    """Parse key=value pairs from a .env file without external dependencies."""
    config = {}
    if not os.path.exists(filepath):
        return config
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    config[key] = val
    except Exception:
        pass
    return config


@dataclass
class AIConfig:
    # Operating Mode: 'hybrid', 'local', 'cloud', 'deterministic'
    mode: str = "hybrid"

    # Execution Policy: 'advisory', 'simulation', 'hardware'
    execution_policy: str = "advisory"

    # Cloud API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Model Endpoints & Settings
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "llama3"
    api_base_url: Optional[str] = None
    request_timeout_s: float = 8.0

    # Safety Gating
    collision_gate_mm: float = 45.0
    require_hardware_confirmation: bool = True

    # Named Workspace Locations
    named_locations: dict = field(
        default_factory=lambda: {
            "nw bay": [-200.0, 150.0, 150.0],
            "nw table": [-200.0, 150.0, 150.0],
            "north bay": [0.0, 150.0, 180.0],
            "center buffer": [0.0, -150.0, 150.0],
            "se table": [150.0, 100.0, 180.0],
            "east bay": [150.0, 100.0, 100.0],
            "east table": [150.0, 100.0, 100.0],
        }
    )


def load_ai_config(workspace_root: Optional[str] = None) -> AIConfig:
    """Load AI Configuration from .env and environment variables."""
    if workspace_root is None:
        workspace_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )

    env_path = os.path.join(workspace_root, ".env")
    env_vars = _parse_env_file(env_path)

    def get_var(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(key, env_vars.get(key, default))

    mode = (get_var("AI_MODE", "hybrid") or "hybrid").lower()
    policy = (get_var("AI_EXECUTION_POLICY", "advisory") or "advisory").lower()
    ollama_url = get_var("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"
    model = get_var("AI_MODEL_NAME", "llama3") or "llama3"
    openai_key = get_var("OPENAI_API_KEY")
    anthropic_key = get_var("ANTHROPIC_API_KEY")
    gemini_key = get_var("GEMINI_API_KEY")

    return AIConfig(
        mode=mode,
        execution_policy=policy,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        gemini_api_key=gemini_key,
        ollama_base_url=ollama_url,
        model_name=model,
    )

