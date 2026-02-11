import os
import time
import tomllib
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage


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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.normpath(os.path.join(base_dir, ".streamlit", "secrets.toml")),
        os.path.normpath(os.path.join(base_dir, "..", ".streamlit", "secrets.toml")),
    ]
    for secrets in candidates:
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


def _get_model_candidates():
    """Return list of (provider, model_class, kwargs) to try in order."""
    keys = _load_keys()
    candidates = []

    for gkey in keys["gemini"]:
        candidates.append(("gemini", gkey))

    if keys["openai"]:
        candidates.append(("openai", keys["openai"]))

    return candidates


def _create_model(provider, api_key):
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.3,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.3,
        )


def run_agent(system_prompt: str, tools: list, max_iterations: int = 5) -> tuple[str, int]:
    """
    Run a ReAct agent with automatic provider fallback.
    Tries each Gemini key, then OpenAI. Retries on 429 with backoff.
    """
    candidates = _get_model_candidates()
    if not candidates:
        raise RuntimeError("No API keys configured")

    last_error = None
    for provider, api_key in candidates:
        try:
            return _run_agent_with_model(
                _create_model(provider, api_key),
                system_prompt, tools, max_iterations
            )
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "rate_limit" in err_str.lower():
                time.sleep(2)
                continue
            raise

    raise RuntimeError(f"All providers exhausted. Last error: {last_error}")


def _run_agent_with_model(llm, system_prompt, tools, max_iterations):
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Analyze the dataset and provide detailed findings. "
                     "Use the available tools to gather data before forming conclusions. "
                     "Do not guess -- call tools first. No emoji in output."),
    ]

    iterations = 0
    response = None

    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        iterations += 1

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            fn = tool_map.get(tc["name"])
            if fn:
                try:
                    result = fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tc['name']}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    final = response.content if response and response.content else "Analysis complete."
    return final, iterations
