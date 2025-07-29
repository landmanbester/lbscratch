from pathlib import Path
import tarfile

import pytest
import requests

MS_NAME = "test_ascii_1h60.0s.MS"
MS_TAR_NAME = f"{MS_NAME}.tar.gz"

# https://drive.google.com/file/d/1rfGXGjjJ2XtF26LImlyJzCJMCNQZgEFT/view?usp=sharing

gdrive_id = "1rfGXGjjJ2XtF26LImlyJzCJMCNQZgEFT"
url = f"https://drive.google.com/uc?id={gdrive_id}"


def download_test_ms(path: Path) -> Path:
    ms_path = path / MS_NAME

    # Download and untar if the ms doesn't exist
    if not ms_path.exists():
        ms_tar_path = path / MS_TAR_NAME

        download = requests.get(url)
        with open(ms_tar_path, "wb") as f:
            f.write(download.content)

        with tarfile.open(ms_tar_path, "r:gz") as tar:
            tar.extractall(path=path)

        ms_tar_path.unlink()

    return ms_path


@pytest.fixture(scope="session")
def ms_name():
    from appdirs import user_cache_dir

    cache_dir = Path(user_cache_dir("radiomesh")) / "test-data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return download_test_ms(cache_dir)
