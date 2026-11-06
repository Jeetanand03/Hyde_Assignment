"""Minimal GLM client (OpenAI-compatible chat completions), stdlib only.

Config via environment variables (or a local .env file, never submitted):
  GLM_API_KEY   - your key (required)
  GLM_BASE_URL  - e.g. https://.../v1   (from the GLM dashboard)
  GLM_MODEL     - the provided model name
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
PROJECT = ROOT.parent
CACHE_DIR = ROOT / ".cache"


def _load_dotenv():
    env = next((p for p in (ROOT / ".env", PROJECT / ".env") if p.exists()), ROOT / ".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
API_KEY = os.environ.get("GLM_API_KEY", "")
BASE_URL = os.environ.get("GLM_BASE_URL", "").rstrip("/")
MODEL = os.environ.get("GLM_MODEL", "")


def chat(messages, temperature=0.0, max_tokens=2048, retries=4, timeout=120, use_cache=True, **extra):
    """Send chat messages, return the assistant text. Responses are cached on disk."""
    if not (API_KEY and BASE_URL and MODEL):
        raise RuntimeError("Set GLM_API_KEY, GLM_BASE_URL and GLM_MODEL (env vars or .env)")
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    body = {"model": MODEL, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, **extra}

    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["text"]
    if os.environ.get("GLM_CACHE_ONLY"):  # rebuild outputs from cached answers only
        raise RuntimeError("cache miss (GLM_CACHE_ONLY set)")

    url = BASE_URL if BASE_URL.endswith("/chat/completions") else BASE_URL + "/chat/completions"
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            text = out["choices"][0]["message"]["content"] or ""
            if not text.strip():
                raise RuntimeError("empty content (raise max_tokens)")
            CACHE_DIR.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps({"text": text}), encoding="utf-8")
            return text
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}"
            if e.code in (400, 401, 403, 404):
                break  # not retryable
        except Exception as e:  # timeouts, connection resets
            last_err = repr(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GLM call failed: {last_err}")


def extract_json(text):
    """Pull the first JSON object/array out of a model reply (handles ```json fences)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return json.loads(text)  # raise a clear error


def chat_json(messages, **kw):
    """chat() + parse JSON, with one repair retry if parsing fails."""
    text = chat(messages, **kw)
    try:
        return extract_json(text)
    except Exception:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        fix = messages + [{"role": "assistant", "content": text},
                          {"role": "user", "content": "Return ONLY valid JSON, no prose."}]
        return extract_json(chat(fix, **kw))
