from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
from time import sleep

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, lang: str = "English", max_retries: int = 5):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.lang = lang
        self.max_retries = max_retries
        self._local_llm = None  # Lazy initialization for fallback
        
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )

    def _get_local_llm(self):
        """Lazily initialize local LLM for fallback."""
        if self._local_llm is None:
            logger.info("Initializing local LLM as fallback...")
            self._local_llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
            logger.info("Local LLM initialized successfully.")
        return self._local_llm

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            for attempt in range(self.max_retries):
                try:
                    response = self.llm.chat.completions.create(messages=messages, temperature=0, model=self.model)
                    return response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        # All OpenAI retries failed, fall back to local LLM
                        logger.warning("All OpenAI API attempts failed. Falling back to local LLM...")
                        try:
                            local_llm = self._get_local_llm()
                            response = local_llm.create_chat_completion(messages=messages, temperature=0)
                            logger.info("Successfully generated response using local LLM fallback.")
                            return response["choices"][0]["message"]["content"]
                        except Exception as fallback_error:
                            logger.error(f"Local LLM fallback also failed: {fallback_error}")
                            raise Exception(f"Both OpenAI API and local LLM failed. OpenAI error: {e}, Local LLM error: {fallback_error}")
                    # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    delay = 2 ** attempt
                    logger.info(f"Retrying in {delay} seconds...")
                    sleep(delay)
        else:
            response = self.llm.create_chat_completion(messages=messages, temperature=0)
            return response["choices"][0]["message"]["content"]

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English", max_retries: int = 5):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang, max_retries=max_retries)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM