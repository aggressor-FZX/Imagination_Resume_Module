import pytest

from city_geo_mapper import get_geo_id


def test_get_geo_id_known_city():
    assert get_geo_id("San Francisco") == "31000US41860"
    assert get_geo_id("SF") == "31000US41860"
    assert get_geo_id("san francisco, ca") == "31000US41860"


def test_get_geo_id_unknown_city():
    assert get_geo_id("Unknown City") is None
    assert get_geo_id("") is None
