from typing import List, Optional, Union
from fastapi import File, UploadFile
from pydantic import BaseModel, field_validator


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


class RectInput(BaseModel):
    startX: Union[float, int]
    startY: Union[float, int]
    width: Union[float, int]
    height: Union[float, int]


class PointInput(BaseModel):
    x: Union[float, int]
    y: Union[float, int]


class SAMInput(BaseModel):
    model: Optional[str]
    rectangles: Optional[List[RectInput]]
    positive_points: Optional[List[PointInput]]
    negative_points: Optional[List[PointInput]]
