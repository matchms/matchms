#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the matchms module.
"""
import pytest


def test_something():
    assert True


def test_with_error():
    with pytest.raises(ValueError):
        # Do something that raises a ValueError
        raise ValueError


# Fixture example
@pytest.fixture
def an_object():
    return {}


def test_matchms(an_object):
    assert an_object == {}
