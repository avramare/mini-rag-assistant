"""Check that the local environment is ready: .env, Ollama, models, Langfuse keys.

Usage: uv run python scripts/check_env.py
Prints one line per check and exits 1 if anything is missing. Never prints secret values.
"""

import sys

import httpx

from mini_rag.config import PROJECT_ROOT, Settings

OK, FAIL = "[ok]  ", "[fail]"


def main() -> int:
    problems = 0

    def report(passed: bool, msg: str, hint: str = "") -> None:
        nonlocal problems
        print(f"{OK if passed else FAIL} {msg}")
        if not passed and hint:
            print(f"       -> {hint}")
        problems += not passed

    env_path = PROJECT_ROOT / ".env"
    report(env_path.exists(), ".env file exists", "copy .env.example to .env and fill it in")

    s = Settings()
    models = {"GEN_MODEL": s.gen_model, "EMBED_MODEL": s.embed_model, "JUDGE_MODEL": s.judge_model}
    for name, value in models.items():
        report(bool(value), f"{name} is set", f"set {name} in .env (a model from `ollama list`)")

    try:
        resp = httpx.get(f"{s.ollama_host}/api/tags", timeout=3.0)
        resp.raise_for_status()
        pulled = {m["name"] for m in resp.json().get("models", [])}
        report(True, f"Ollama reachable at {s.ollama_host}")
    except (httpx.HTTPError, ValueError) as exc:
        pulled = None
        report(False, f"Ollama reachable at {s.ollama_host}",
               f"start Ollama (`ollama serve`) or fix OLLAMA_HOST ({type(exc).__name__})")

    if pulled is not None:
        for name, value in models.items():
            if value:
                # `ollama list` shows "llama3.2:latest" for a model configured as "llama3.2"
                found = value in pulled or f"{value}:latest" in pulled
                report(found, f"{name} '{value}' is pulled", f"run `ollama pull {value}`")

    for name, secret in {
        "LANGFUSE_PUBLIC_KEY": s.langfuse_public_key,
        "LANGFUSE_SECRET_KEY": s.langfuse_secret_key,
    }.items():
        report(bool(secret.get_secret_value()), f"{name} is set",
               "create keys in Langfuse project settings (needed from Phase 2)")

    print("\nAll checks passed." if problems == 0 else f"\n{problems} problem(s) found.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
