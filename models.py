from pydantic import BaseModel
from typing import List

class Task(BaseModel):
    id: int
    description: str

class Plan(BaseModel):
    tasks: List[Task]
