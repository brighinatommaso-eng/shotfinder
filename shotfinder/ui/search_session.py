"""Query generation tracking for the search window.

The window debounces typing and runs searches on a thread pool, so two queries
can be in flight at once. Each search carries the generation it was started
for; a result whose generation is no longer current is dropped instead of
overwriting newer results.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


class SearchSession:
    """Hands out generation tokens and rejects results from older queries."""

    def __init__(self) -> None:
        self._generation = 0
        self._query = ""

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def query(self) -> str:
        return self._query

    def begin(self, query: str) -> int:
        """Start a new query and return its generation token."""
        self._generation += 1
        self._query = query
        return self._generation

    def is_current(self, generation: int) -> bool:
        return generation == self._generation

    def accept(self, generation: int, results: T) -> T | None:
        """Return *results* only when no newer query has started."""
        return results if self.is_current(generation) else None
