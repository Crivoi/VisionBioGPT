from enum import Enum


class ModelCheckpoint(Enum):
    """
    Checkpoints of used models
    """
    BioGPT = "microsoft/biogpt"


class NoteEventCategory(Enum):
    """
    Categories of MIMIC-III note events
    """
    DischargeSummary = "Discharge summary"
