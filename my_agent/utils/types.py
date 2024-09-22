from typing_extensions import TypedDict, Literal

class Turn(TypedDict):
    speaker: Literal["examiner", "candidate"]
    text: str

class Suggestion(TypedDict):
    text: str
    explanation: str