import os
import json
import requests

def download_github_raw_json(raw_url, local_save_dir, filename, overwrite=False):
    """
    Download un file JSON da GitHub (raw content URL),
    salvarlo in locale, e restituirne il contenuto come dict/list.
    Restituisce None in caso di errore.
    """
    os.makedirs(local_save_dir, exist_ok=True)
    local_file_path = os.path.join(local_save_dir, filename)

    if os.path.exists(local_file_path) and not overwrite:
        try:
            with open(local_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e_load:
            print(f"  ERROR loading existing local file {local_file_path}: {e_load}. Re-downloading...")

    try:
        response = requests.get(raw_url, stream=True)
        response.raise_for_status()

        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with open(local_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    except requests.exceptions.RequestException as e_req:
        print(f"  ERROR downloading from GitHub {raw_url}: {e_req}")
    except json.JSONDecodeError as e_json:
        print(f"  ERROR decoding JSON from {local_file_path}: {e_json}")
    except Exception as e_generic:
        print(f"  ERROR processing {filename}: {e_generic}")

    return None
import os
import json
import requests

def download_github_raw_json(raw_url, local_save_dir, filename, overwrite=False):
    """
    Download un file JSON da GitHub (raw content URL),
    salvarlo in locale, e restituirne il contenuto come dict/list.
    Restituisce None in caso di errore.
    """
    os.makedirs(local_save_dir, exist_ok=True)
    local_file_path = os.path.join(local_save_dir, filename)

    if os.path.exists(local_file_path) and not overwrite:
        try:
            with open(local_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e_load:
            print(f"  ERROR loading existing local file {local_file_path}: {e_load}. Re-downloading...")

    try:
        response = requests.get(raw_url, stream=True)
        response.raise_for_status()

        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with open(local_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    except requests.exceptions.RequestException as e_req:
        print(f"  ERROR downloading from GitHub {raw_url}: {e_req}")
    except json.JSONDecodeError as e_json:
        print(f"  ERROR decoding JSON from {local_file_path}: {e_json}")
    except Exception as e_generic:
        print(f"  ERROR processing {filename}: {e_generic}")

    return None
