import os
import tomllib
from google import genai
from openai import OpenAI

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SECRETS_CANDIDATES = [
    os.path.normpath(os.path.join(_BASE_DIR, ".streamlit", "secrets.toml")),
    os.path.normpath(os.path.join(_BASE_DIR, "..", ".streamlit", "secrets.toml")),
]


def _load_keys():
    keys = {"gemini": [], "openai": None}
    
    # 1. Try Streamlit Secrets (for Cloud Deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                keys["gemini"].append(st.secrets["GEMINI_API_KEY"])
            if "GEMINI_API_KEY_2" in st.secrets:
                keys["gemini"].append(st.secrets["GEMINI_API_KEY_2"])
            if "OPENAI_API_KEY" in st.secrets:
                keys["openai"] = st.secrets["OPENAI_API_KEY"]
    except ImportError:
        pass

    # 2. Try Local Secrets File
    for secrets in _SECRETS_CANDIDATES:
        if not os.path.exists(secrets):
            continue
        with open(secrets, "rb") as f:
            data = tomllib.load(f)
        for k in ["GEMINI_API_KEY", "GEMINI_API_KEY_2"]:
            if k in data and data[k] and data[k] not in keys["gemini"]:
                keys["gemini"].append(data[k])
        if "OPENAI_API_KEY" in data and data["OPENAI_API_KEY"] and not keys["openai"]:
            keys["openai"] = data["OPENAI_API_KEY"]
            
    # 3. Try Environment Variables
    for k in ["GEMINI_API_KEY", "GEMINI_API_KEY_2"]:
        v = os.environ.get(k)
        if v and v not in keys["gemini"]:
            keys["gemini"].append(v)
    if not keys["openai"]:
        keys["openai"] = os.environ.get("OPENAI_API_KEY")
        
    return keys


def call_llm(prompt: str, max_tokens: int = 4000) -> str:
    keys = _load_keys()
    for gkey in keys["gemini"]:
        try:
            client = genai.Client(api_key=gkey)
            resp = client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
            return resp.text
        except Exception:
            continue
    if keys["openai"]:
        try:
            client = OpenAI(api_key=keys["openai"])
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"All LLM providers failed. Last: {e}")
    raise RuntimeError("No API keys configured")
