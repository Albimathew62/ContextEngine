import warnings
from typing import Optional

class TiktokenEstimator:
    
    def __init__(self,model:str="gpt-4"):
        self.model=model
        self._encoder=None
        self._available=self._load(model)
        
    def _load(self,model:str)->bool:
        try:
            import tiktoken
            self._encoder=tiktoken.encoding_for_model(model)
            return True
        except ImportError:
            warnings.warn(
                "tiktoken is not installed. Falling back to chars//4 estimate. "
                "Run: pip install contextengine-ai[tiktoken]",
                UserWarning,
                stacklevel=2,
            )
            return False
        
        except KeyError:
            import tiktoken
            self._encoder=tiktoken.get_encoding("cl100k_base")
            return True
        
        
    def count(self,text:str)->int:
        if not self._available or self._encoder is None:
            return max(1,len(text)//4)
        return max(1,len(self._encoder.encode(text)))
    