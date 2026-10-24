"""Pydantic request models for landmark JSON.

Video / frame payload shapes are REJECTED, not ignored:
``extra="forbid"`` on every model turns a request carrying ``video``,
``image``, ``frames_base64`` or any other unlisted field into a 422,
and the gateway independently rejects image/video content parts.
Landmark JSON is the only accepted input shape.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

FORBIDDEN_HINT = "video/frame payloads are not accepted; send landmark JSON only"


class Landmark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    z: float | None = None
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)


class Frame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t: float
    landmarks: dict[str, Landmark]


class TugMarkers(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float
    stand: float
    turn_start: float
    turn_end: float
    sit: float


class TugRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str = Field(min_length=1, max_length=120)
    frames: list[Frame] = Field(min_length=5)
    markers: TugMarkers | None = None


class GaitPass(BaseModel):
    model_config = ConfigDict(extra="forbid")
    view: str = Field(pattern="^(front|side)$")
    frames: list[Frame] = Field(min_length=3)


class ArthritisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str = Field(min_length=1, max_length=120)
    standing_frames: list[Frame] = Field(min_length=1)
    gait_passes: list[GaitPass] = Field(default_factory=list)
