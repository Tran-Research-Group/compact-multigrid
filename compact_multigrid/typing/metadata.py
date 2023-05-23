from typing import Any, Literal, TypeAlias
from typing_extensions import TypedDict, NotRequired

RenderMode: TypeAlias = Literal["human", "rgb_array", "ansi", "ascii"]


class Metadata(TypedDict):
    render_modes: list[RenderMode]
    render_fps: NotRequired[int]
    options: NotRequired[dict[str, Any]]
