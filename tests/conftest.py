from dataclasses import dataclass
from pathlib import Path

import pytest
import xarray as xr

import lenapy


def pytest_addoption(parser):
    parser.addoption(
        "--overwrite_references",
        action="store_true",
        default=False,
        help="Set to True to overwrite existing reference data with new values during this test run. "
        "Useful when updating expected outputs",
    )


@pytest.fixture
def overwrite_references(request):
    return request.config.getoption("--overwrite_references")


@dataclass
class LenapyTestsPath:
    project_dir: Path

    def __post_init__(self):
        self.data = self.project_dir / "data"
        self.ref_data = self.project_dir / "tests" / "ref_data"


@pytest.fixture
def lenapy_paths(request) -> LenapyTestsPath:
    return LenapyTestsPath(Path(__file__).parent.parent)


@pytest.fixture(scope="session")
def air_temperature_data():
    """Fixture session to load et rename data from xarray tutorials."""
    dataset = xr.tutorial.open_dataset("air_temperature")
    return lenapy.utils.geo.rename_data(dataset)


@pytest.fixture(scope="session")
def ohc_data():
    return xr.open_dataset(
        LenapyTestsPath(Path(__file__).parent.parent).data / "ecco.nc",
        engine="lenapyNetcdf",
    )


@pytest.fixture(scope="session")
def ersstv5_data():
    """Fixture session to load et rename data from xarray tutorials."""
    dataset = xr.tutorial.open_dataset("ersstv5")
    return lenapy.utils.geo.rename_data(dataset)
