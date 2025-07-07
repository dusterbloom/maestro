
import asyncio
from typing import Dict, Any

class RequestDeduplicator:
    def __init__(self):
        self.active_requests: Dict[str, asyncio.Task] = {}

    async def process_or_join(self, request_id: str, coro):
        if request_id in self.active_requests:
            return await self.active_requests[request_id]

        task = asyncio.create_task(coro)
        self.active_requests[request_id] = task
        try:
            return await task
        finally:
            if request_id in self.active_requests:
                del self.active_requests[request_id]
