#!/bin/bash
# Run the bot using the project's virtual environment
cd "$(dirname "$0")"
exec .venv/bin/python main.py "$@"
