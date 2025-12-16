# core/llm_client.py
import os
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from app.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_NUM_PREDICT

groq_api_key = os.getenv("GROQ_API_KEY")



def init_llm(model_name: str = None, base_url: str = None, num_predict: int = None):
    m = model_name or OLLAMA_LLM_MODEL
    b = base_url or OLLAMA_BASE_URL
    npred = num_predict or OLLAMA_NUM_PREDICT
    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=groq_api_key
)
    llm = Ollama(base_url=b, model=m, temperature=0.2, num_ctx=4096, num_predict=npred)
    # Optionally small warm-up
    _ = llm.invoke("hello")[:1]
    return llm
