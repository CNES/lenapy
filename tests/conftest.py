from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class LenapyTestsPath:
    project_dir: Path

    def __post_init__(self):
        self.data = self.project_dir / "data"


@pytest.fixture
def lenapy_paths(request) -> LenapyTestsPath:
    return LenapyTestsPath(Path(__file__).parent.parent)
