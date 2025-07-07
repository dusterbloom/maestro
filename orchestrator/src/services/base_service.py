
from abc import ABC, abstractmethod
from typing import Any, Dict

class ServiceResult:
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error

class BaseService(ABC):
    @abstractmethod
    async def process(self, input: Any, context: Dict) -> ServiceResult:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass

    @abstractmethod
    def get_metrics(self) -> Dict:
        pass
