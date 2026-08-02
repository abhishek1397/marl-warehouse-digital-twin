"""CSVLogger recording training metrics to structured CSV files."""

import csv
import os
from typing import Any, Dict, List, Optional


class CSVLogger:
    """Logs structured metrics dictionary rows to CSV files."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path: str = os.path.abspath(csv_path)
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        self._headers: Optional[List[str]] = None
        self._file = None
        self._writer = None

    def log_row(self, data: Dict[str, Any]) -> None:
        """Writes a row dictionary to CSV file."""
        if not data:
            return

        current_keys = list(data.keys())

        # Initialize or update CSV headers
        if self._headers is None or set(current_keys) != set(self._headers):
            self._headers = current_keys
            file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

            if self._file:
                self._file.close()

            mode = "a" if file_exists else "w"
            self._file = open(self.csv_path, mode, newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._headers)

            if not file_exists:
                self._writer.writeheader()

        if self._writer:
            self._writer.writerow(data)
            self._file.flush()

    def close(self) -> None:
        """Closes file handle."""
        if self._file and not self._file.closed:
            self._file.close()
            self._file = None
