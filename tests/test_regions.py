import numpy as np

from thermal import regions


def test_box_nswe_string():
    s = regions.box_nswe({"north": 66.8, "south": 63.1, "west": -25.0, "east": -13.0})
    assert s == [66.8, 63.1, -25.0, -13.0]


def test_reykjanes_mask():
    lat = np.array([[63.9, 64.5], [63.8, np.nan]])
    lon = np.array([[-22.4, -22.4], [-21.8, -21.0]])
    m = regions.region_mask("reykjanes", lat, lon)
    assert m[0, 0]  # Svartsengi area inside
    assert not m[0, 1]  # north of the peninsula
    assert m[1, 0]
    assert not m[1, 1]  # nan excluded
