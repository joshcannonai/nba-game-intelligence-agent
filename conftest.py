"""Puts the repo root on sys.path so `pytest` works, not just `python -m pytest`.

Bare `pytest` does not add the current directory to sys.path; `python -m pytest`
does. Without this file, a fresh clone following the README's `pytest` hits
ModuleNotFoundError: No module named 'agent'.
"""
