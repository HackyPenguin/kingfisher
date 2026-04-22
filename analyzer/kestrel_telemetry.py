"""Kingfisher telemetry shim.

Telemetry is intentionally disabled during the Kingfisher rebrand.
This module preserves the existing call surface so the rest of the app can
continue to import it safely, but every outbound send function is a no-op.
"""

from __future__ import annotations

import json
import os
import platform
import uuid
from typing import Any, Dict, List, Optional

_MAX_LOG_ENTRIES = 50
_MAX_RUNTIME_LOG_LINES = 200
_MAX_RUNTIME_LOG_CHARS = 40_000
_MAX_RUNTIME_LOG_TOTAL_CHARS = 60_000


def _read_version() -> str:
    try:
        for candidate in [
            os.path.join(os.path.dirname(__file__), 'VERSION.txt'),
            os.path.join(os.path.dirname(__file__), '..', 'VERSION.txt'),
        ]:
            if os.path.isfile(candidate):
                with open(candidate, 'r', encoding='utf-8') as f:
                    for line in f:
                        value = line.strip()
                        if not value:
                            continue
                        if value.lower().startswith('version:'):
                            return value.split(':', 1)[1].strip() or 'unknown'
                        return value
    except Exception:
        pass
    return 'unknown'


def _get_os_info() -> str:
    try:
        return f"{platform.system()}-{platform.release()}-{platform.machine()}"
    except Exception:
        return 'unknown'


def get_machine_id(settings: dict) -> str:
    try:
        mid = settings.get('machine_id')
        if mid and isinstance(mid, str) and len(mid) > 8:
            return mid
        mid = str(uuid.uuid4())
        settings['machine_id'] = mid
        return mid
    except Exception:
        return 'unknown'


def _disabled(*_args, **_kwargs) -> None:
    return None


def send_feedback(*args, **kwargs) -> None:
    _disabled(*args, **kwargs)


def send_crash_report(*args, **kwargs) -> None:
    _disabled(*args, **kwargs)


def send_installation_telemetry(*args, **kwargs) -> None:
    _disabled(*args, **kwargs)


def send_analysis_completion_telemetry(*args, **kwargs) -> None:
    _disabled(*args, **kwargs)


def send_folder_analytics(*args, **kwargs) -> None:
    _disabled(*args, **kwargs)


def get_recent_log_tail(folder: Optional[str] = None, max_entries: int = _MAX_LOG_ENTRIES, runtime_log_files: int = 1) -> str:
    try:
        try:
            from kestrel_analyzer.config import KESTREL_DIR_NAME, LOG_FILENAME_PREFIX, LOG_FILE_EXTENSION
        except ImportError:
            from analyzer.kestrel_analyzer.config import KESTREL_DIR_NAME, LOG_FILENAME_PREFIX, LOG_FILE_EXTENSION
    except Exception:
        KESTREL_DIR_NAME = '.kingfisher'
        LOG_FILENAME_PREFIX = 'kingfisher_error'
        LOG_FILE_EXTENSION = 'json'

    try:
        candidates = []
        if folder:
            candidates.append(os.path.join(folder, KESTREL_DIR_NAME))
        candidates.append(os.path.join(os.path.expanduser('~'), KESTREL_DIR_NAME))

        payload: Dict[str, Any] = {}
        analysis_path = None
        analysis_mtime = 0.0
        for log_dir in candidates:
            if not os.path.isdir(log_dir):
                continue
            for fname in os.listdir(log_dir):
                if fname.startswith(LOG_FILENAME_PREFIX) and fname.endswith(f'.{LOG_FILE_EXTENSION}'):
                    fp = os.path.join(log_dir, fname)
                    try:
                        mt = os.path.getmtime(fp)
                    except OSError:
                        continue
                    if mt > analysis_mtime:
                        analysis_mtime = mt
                        analysis_path = fp
        if analysis_path:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                payload['analysis_entries'] = data[-max_entries:]

        runtime_files = []
        runtime_file_limit = max(1, min(int(runtime_log_files or 1), 5))
        for base_dir in candidates:
            runtime_dir = os.path.join(base_dir, 'logs')
            if not os.path.isdir(runtime_dir):
                continue
            for fname in os.listdir(runtime_dir):
                if fname.startswith('kingfisher_runtime_') and fname.endswith('.log'):
                    fp = os.path.join(runtime_dir, fname)
                    try:
                        runtime_files.append((os.path.getmtime(fp), fp))
                    except OSError:
                        continue
        runtime_files.sort(key=lambda x: x[0], reverse=True)
        per_file_char_budget = max(2_000, int(_MAX_RUNTIME_LOG_TOTAL_CHARS / max(1, runtime_file_limit)))
        per_file_char_budget = min(per_file_char_budget, _MAX_RUNTIME_LOG_CHARS)
        runtime_tails = []
        for _, runtime_path in runtime_files[:runtime_file_limit]:
            try:
                with open(runtime_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                runtime_tail = ''.join(lines[-_MAX_RUNTIME_LOG_LINES:])
                if len(runtime_tail) > per_file_char_budget:
                    runtime_tail = runtime_tail[-per_file_char_budget:]
                if runtime_tail:
                    runtime_tails.append({'file': os.path.basename(runtime_path), 'tail': runtime_tail})
            except Exception:
                continue
        if runtime_tails:
            payload['runtime_output_tails'] = runtime_tails
            payload['runtime_output_tail'] = runtime_tails[0].get('tail', '')
        return json.dumps(payload, indent=2, default=str) if payload else ''
    except Exception:
        return ''


def collect_folder_stats(item_path: str, files_this_session: int, total_files: int) -> Dict[str, Any]:
    try:
        try:
            from kestrel_analyzer.config import RAW_EXTENSIONS, JPEG_EXTENSIONS
        except ImportError:
            from analyzer.kestrel_analyzer.config import RAW_EXTENSIONS, JPEG_EXTENSIONS
        all_exts = {str(e).lower() for e in set(RAW_EXTENSIONS) | set(JPEG_EXTENSIONS)}
        file_sizes_kb: List[float] = []
        file_formats: Dict[str, int] = {}
        for fname in os.listdir(item_path):
            if len(file_sizes_kb) >= 1000:
                break
            fpath = os.path.join(item_path, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in all_exts:
                continue
            try:
                file_sizes_kb.append(round(os.path.getsize(fpath) / 1024.0, 1))
            except OSError:
                continue
            file_formats[ext] = file_formats.get(ext, 0) + 1
        return {'file_sizes_kb': file_sizes_kb, 'file_formats': file_formats}
    except Exception:
        return {'file_sizes_kb': [], 'file_formats': {}}
