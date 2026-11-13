"""Milestone 1: badminton singles analysis on the BFMD annotations.

Modules
    config    YAML config loading and path resolution
    coords    unit conversions between BFMD coordinate conventions and pixels
    court     court model (metres), homographies image <-> court
    raw       parsers for the raw BFMD JSON files (one function per annotation type)
    schema    `MatchData`: the documented, validated per-match container
    build     raw JSON -> MatchData, plus the on-disk cache
    validate  data-quality and alignment checks with a written report
    viz       rally visualisation (image-space canvas + top-down court)
"""

__all__ = ["config", "coords", "court", "raw", "schema", "build", "validate", "viz"]
