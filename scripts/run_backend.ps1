$ErrorActionPreference = "Stop"
$env:NO_COLOR = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
python backend/run_dev.py
