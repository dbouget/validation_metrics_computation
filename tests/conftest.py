import os
import shutil
import pytest
import logging
from tests.download_resources import download_resources


@pytest.fixture(scope="session")
def test_dir():
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.listdir(test_dir):
        download_resources(test_dir=test_dir)
    yield test_dir
    logging.info(f"Removing the temporary directory for tests.")
    shutil.rmtree(test_dir)