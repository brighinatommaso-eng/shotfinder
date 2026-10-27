"""Stale search responses are discarded by generation, not by arrival order."""

from __future__ import annotations

from shotfinder.ui.search_session import SearchSession


def test_a_newer_query_invalidates_an_in_flight_result() -> None:
    session = SearchSession()
    first = session.begin("spiaggia")
    second = session.begin("spider robot")

    assert session.accept(first, ["old"]) is None
    assert session.accept(second, ["new"]) == ["new"]


def test_repeating_the_same_query_still_advances_the_generation() -> None:
    session = SearchSession()
    first = session.begin("proxy")
    second = session.begin("proxy")

    assert first != second
    assert session.accept(first, ["first"]) is None
    assert session.accept(second, ["second"]) == ["second"]


def test_empty_results_are_still_accepted() -> None:
    session = SearchSession()
    generation = session.begin("nothing matches")

    assert session.accept(generation, []) == []
    assert session.is_current(generation)


def test_query_is_kept_for_labelling() -> None:
    session = SearchSession()
    assert session.query == ""

    session.begin("  passphrase  ")

    assert session.query == "  passphrase  "
    assert session.generation == 1
