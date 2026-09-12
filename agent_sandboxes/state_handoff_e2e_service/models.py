from pydantic import BaseModel, ConfigDict, Field

class ItemPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    embedding: list[float] = Field(default_factory=list)
