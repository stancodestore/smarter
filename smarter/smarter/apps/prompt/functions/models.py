"""Sample models for prompt function modules."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FilterVehiclesRequestModel(BaseModel):
    """Validate input arguments for the ``filter_vehicles`` tool."""

    budget: int = Field(..., gt=0, description="Maximum price in USD, e.g. 45000")
    body_type: str = Field(..., min_length=1, description="Vehicle body style, e.g. 'SUV'")
    year_min: Optional[int] = Field(
        default=None,
        ge=1900,
        le=datetime.now().year + 1,
        description="Earliest model year (inclusive)",
    )

    @field_validator("body_type")
    @classmethod
    def strip_body_type(cls, value: str) -> str:
        """Strip whitespace and normalize body type to title case."""
        return value.strip().title()
