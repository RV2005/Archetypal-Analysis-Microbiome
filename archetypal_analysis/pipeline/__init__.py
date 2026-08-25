"""Config-driven, resumable archetypal-analysis pipeline."""

from .config import load_config

__all__ = ["load_config", "PipelineRunner"]


def __getattr__(name):
    if name == "PipelineRunner":
        from .runner import PipelineRunner
        return PipelineRunner
    raise AttributeError(name)