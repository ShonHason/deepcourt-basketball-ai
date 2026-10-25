#!/bin/bash
# Start DeepCourt.ai — API server (8510) + Streamlit dashboard (8502)
cd "$(dirname "$0")"

pkill -f "streamlit run dashboard/app.py" 2>/dev/null
pkill -f "landing/api.py" 2>/dev/null
sleep 1

# FastAPI auth + static server (Google OAuth lives here)
.venv/bin/python landing/api.py &
sleep 2

# Streamlit dashboard
.venv/bin/streamlit run dashboard/app.py --server.port 8502 --server.headless true --server.fileWatcherType none &
sleep 3

open http://localhost:8510/
