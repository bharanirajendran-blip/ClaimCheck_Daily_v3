"""
ClaimCheck Daily — Agent package
Architecture:
  - Director  : GPT-4o orchestrates the daily pipeline (claim selection, verdict synthesis)
  - Researcher : Claude does the deep research + source verification per claim
  - Publisher  : Renders findings to docs/ for GitHub Pages
"""

__all__ = ["Director", "Researcher", "Publisher", "run_pipeline"]


def __getattr__(name: str):
    """Lazy attribute loading keeps lightweight entrypoints importable."""
    if name == "Director":
        from .director import Director

        return Director
    if name == "Researcher":
        from .researcher import Researcher

        return Researcher
    if name == "Publisher":
        from .publisher import Publisher

        return Publisher
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
