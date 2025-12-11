# core/llm_client.py
from langchain_community.llms import Ollama
from app.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_NUM_PREDICT

def init_llm(model_name: str = None, base_url: str = None, num_predict: int = None):
    m = model_name or OLLAMA_LLM_MODEL
    b = base_url or OLLAMA_BASE_URL
    npred = num_predict or OLLAMA_NUM_PREDICT
    llm = Ollama(base_url=b, model=m, temperature=0.2, num_ctx=4096, num_predict=npred)
    # Optionally small warm-up
    _ = llm.invoke("hello")[:1]
    return llm
