#!/bin/bash
# Start FastAPI app with Uvicorn on Render's assigned port
uvicorn api_app:app --host 0.0.0.0 --port $PORT
