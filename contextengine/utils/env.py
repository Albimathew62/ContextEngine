import os
from pathlib import Path
from typing import Optional


def load_env(
    path: Optional[str] = None,
    override: bool = False,
) -> None:
    """
    Optional .env loader.

    - Does nothing if python-dotenv is not installed
    - Does nothing if .env file is missing
    - Never required for ContextEngine to work

    Parameters:
    - path: custom path to .env (default: project root)
    - override: override existing environment variables
    """

    try:
        from dotenv import load_dotenv
    except ImportError:
        # dotenv is optional
        return

    if path:
        env_path = Path(path)
    else:
        env_path = Path.cwd() / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=override)
