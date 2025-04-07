from dataclasses import dataclass
from pathlib import Path

import pytest
import xarray as xr

import lenapy


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
