from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Point(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class Box(BaseModel):
    start_x: float = Field(ge=0)
    start_y: float = Field(ge=0)
    width: float
    height: float

    def to_corners(self) -> tuple:
        """Absolute XYXY corners, which is the layout SAM expects for a box prompt."""
        x_min = min(self.start_x, self.start_x + self.width)
        y_min = min(self.start_y, self.start_y + self.height)
        x_max = max(self.start_x, self.start_x + self.width)
        y_max = max(self.start_y, self.start_y + self.height)
        return x_min, y_min, x_max, y_max


class SegmentationRequest(BaseModel):
    model_type: Optional[str] = None
    boxes: List[Box] = []
    positive_points: List[Point] = []
    negative_points: List[Point] = []

    @model_validator(mode="after")
    def require_at_least_one_prompt(self):
        if not self.boxes and not self.positive_points and not self.negative_points:
            raise ValueError("at least one point or box is required")
        return self


class SegmentationJob(BaseModel):
    job_id: str
    created_at: str
    model_type: Optional[str]
    image_width: int
    image_height: int
    prompt_counts: dict
