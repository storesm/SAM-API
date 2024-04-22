from typing import List, Optional, Union
from pydantic import BaseModel, field_validator


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


class SAMModel(BaseModel):
    value: str
    label: str


class RectInput(BaseModel):
    startX: float
    startY: float
    width: float
    height: float


class PointInput(BaseModel):
    x: int
    y: int


class SAMInput(BaseModel):
    model: Optional[SAMModel]
    rectangles: Optional[List[RectInput]]
    positive_points: Optional[List[PointInput]]
    negative_points: Optional[List[PointInput]]
