from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr

from thermal import geo


def test_solar_zenith_equinox_noon():
    # Sun close to overhead at the equator at the March 2024 equinox, solar noon ~12:07 UTC at 0E
    zen, _ = geo.solar_position(datetime(2024, 3, 20, 12, 7, tzinfo=UTC), 0.0, 0.0)
    assert zen < 1.0


def test_solar_zenith_midnight_reykjanes_winter():
    zen, az = geo.solar_position(datetime(2024, 1, 15, 0, 0, tzinfo=UTC), 63.9, -22.4)
    assert zen > 130
    assert 300 < az or az < 60  # sun below the horizon roughly to the north


def test_solar_arrays():
    lat = np.array([[63.0, 64.0], [65.0, 66.0]])
    lon = np.array([[-22.0, -22.0], [-18.0, -18.0]])
    zen, az = geo.solar_position(datetime(2024, 6, 21, 12, 0, tzinfo=UTC), lat, lon)
    assert zen.shape == (2, 2)
    assert np.all((zen > 35) & (zen < 50))


def test_satellite_angles_subpoint_and_iceland():
    zen, _ = geo.satellite_angles(0.0, 0.0, 0.0)
    assert zen == pytest.approx(0.0, abs=1e-6)
    zen, az = geo.satellite_angles(63.9, -22.4, 0.0)
    assert 72 < zen < 76  # standard geostationary geometry gives ~74 deg
    assert 145 < az < 165  # towards the SSE


def test_pixel_area_regular_grid_equator():
    lat, lon = np.meshgrid(np.arange(-1.0, 1.01, 0.5), np.arange(-1.0, 1.01, 0.5), indexing="ij")
    area = geo.pixel_area_m2(lat, lon)
    expected = (0.5 * 111.2e3) ** 2
    assert area[2, 2] == pytest.approx(expected, rel=0.02)


def test_geolocate_latlon_variables():
    lat = np.linspace(63, 67, 5)
    lon = np.linspace(-25, -13, 7)
    ds = xr.Dataset(coords={"lat": ("y", lat), "lon": ("x", lon)})
    la, lo, method = geo.geolocate(ds)
    assert la.shape == (5, 7) and lo.shape == (5, 7)
    assert "lat/lon" in method


def test_geolocate_geos_projection():
    from pyproj import CRS, Transformer

    h = 35785831.0
    crs = CRS.from_proj4(f"+proj=geos +h={h} +lon_0=0 +sweep=y +a=6378169 +b=6356583.8 +units=m")
    fwd = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    x0, y0 = fwd.transform(-22.0, 64.0)
    x = x0 + np.arange(-3, 4) * 3000.0
    y = y0 + np.arange(-2, 3) * 3000.0
    ds = xr.Dataset(
        {
            "IR_039": (("y", "x"), np.zeros((5, 7)), {"grid_mapping": "geos"}),
            "geos": (
                (),
                0,
                {
                    "grid_mapping_name": "geostationary",
                    "perspective_point_height": h,
                    "longitude_of_projection_origin": 0.0,
                    "sweep_angle_axis": "y",
                    "semi_major_axis": 6378169.0,
                    "semi_minor_axis": 6356583.8,
                },
            ),
        },
        coords={"x": ("x", x, {"units": "m"}), "y": ("y", y, {"units": "m"})},
    )
    lat, lon, method = geo.geolocate(ds)
    assert "geos" in method
    assert lat[2, 3] == pytest.approx(64.0, abs=1e-3)
    assert lon[2, 3] == pytest.approx(-22.0, abs=1e-3)
