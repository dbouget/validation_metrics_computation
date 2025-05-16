import os
import shutil
import requests
import logging
import traceback
import zipfile


def download_resources(test_dir: str):
    dest_dir = ""
    try:
        test_case1 = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsValLib_UnitTest1-v1.1.zip'
        dest_dir = os.path.join(test_dir)
        os.makedirs(dest_dir, exist_ok=True)

        archive_dl_dest = os.path.join(dest_dir, 'test_case1.zip')
        headers = {}
        response = requests.get(test_case1, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

    except Exception as e:
        logging.error(f"Error during input data download with: {e} \n {traceback.format_exc()}\n")
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        raise ValueError("Error during input data download.\n")
