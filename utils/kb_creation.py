import os
import ast
import time
import random
from tqdm.auto import tqdm

MAX_PY_FILE_SIZE_BYTES = 2_000_000  # 2 MB
MAX_SOURCE_CHARS_FOR_AST_PARSE = 100_000
AST_WALK_TIMEOUT_S = 2
SKIP_KNOWN_PROBLEMATIC_FILES_CONFIG = {}

def extract_code_units_from_file(py_file_path, fallback_encoding_to_use="latin-1"):
    """Estrae funzioni/classi da un file .py tramite AST parsing."""
    units = []
    source_code = None
    try:
        if os.path.getsize(py_file_path) > MAX_PY_FILE_SIZE_BYTES:
            return units
        with open(py_file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except UnicodeDecodeError:
        try:
            with open(py_file_path, 'r', encoding=fallback_encoding_to_use) as f:
                source_code = f.read()
        except Exception:
            return units
    except Exception:
        return units

    if source_code and len(source_code) <= MAX_SOURCE_CHARS_FOR_AST_PARSE:
        try:
            parsed_tree = ast.parse(source_code, filename=os.path.basename(py_file_path))
            can_get_segment = hasattr(ast, 'get_source_segment')
            walk_start = time.time()
            for node in ast.walk(parsed_tree):
                if time.time() - walk_start > AST_WALK_TIMEOUT_S:
                    break
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    segment = None
                    if can_get_segment:
                        try:
                            segment = ast.get_source_segment(source_code, node)
                        except Exception:
                            pass
                    if segment is None:
                        try:
                            if hasattr(ast, 'unparse'):
                                segment = ast.unparse(node)
                            else:
                                segment = ast.dump(node)
                        except Exception:
                            pass
                    if segment:
                        units.append(segment)
        except Exception:
            pass
    return units


def generate_kb_for_library_sources(source_root_dir, lib_key, max_snippets_cfg=1000, use_tqdm_cfg=True, fallback_enc_cfg="latin-1"):
    """
    Cammina dentro la directory sorgente, estrae snippet da file .py,
    e restituisce una lista di snippet per la KB.
    """
    print(f"  Building KB for '{lib_key}' from: {source_root_dir}")
    kb_snippets_list = []
    py_file_paths_to_process = []

    for root, _, files in os.walk(source_root_dir):
        if any(excluded in root for excluded in ['.git', 'docs', 'tests', 'examples', '__pycache__', '.venv']):
            continue
        for file_name in files:
            if file_name.endswith(".py"):
                py_file_paths_to_process.append(os.path.join(root, file_name))

    iterator = tqdm(py_file_paths_to_process, desc=f"Snippets: {lib_key}", unit="file", leave=False) if use_tqdm_cfg else py_file_paths_to_process

    for py_file_path in iterator:
        relative_file_path = os.path.relpath(py_file_path, source_root_dir)
        if lib_key in SKIP_KNOWN_PROBLEMATIC_FILES_CONFIG and relative_file_path in SKIP_KNOWN_PROBLEMATIC_FILES_CONFIG[lib_key]:
            print(f"    Skipping problematic file: {relative_file_path} for {lib_key}")
            continue

        snippets = extract_code_units_from_file(py_file_path, fallback_enc_cfg)
        kb_snippets_list.extend(snippets)

    total = len(kb_snippets_list)
    print(f"    Extracted {total} snippets for '{lib_key}'.")
    if total > max_snippets_cfg:
        print(f"    Sampling {max_snippets_cfg} snippets...")
        kb_snippets_list = random.sample(kb_snippets_list, max_snippets_cfg)

    if not kb_snippets_list:
        print(f"    WARNING: No snippets extracted for library '{lib_key}'.")

    return kb_snippets_list
