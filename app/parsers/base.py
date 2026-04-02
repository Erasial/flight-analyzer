from abc import ABC, abstractmethod
import pandas as pd

class DataParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> pd.DataFrame:
        pass
    