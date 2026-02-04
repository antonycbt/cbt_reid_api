from pydantic import BaseModel
from typing import Generic, Optional, TypeVar

T = TypeVar("T")

class MessageResponse(BaseModel, Generic[T]):
    message: str
    data: Optional[T] = None
