# VidAI — AI Video Generator

A full-stack web app that turns any topic into a complete AI-generated video.
Built with Python, Flask, MoviePy, Edge TTS, Pexels API, and Ollama.

## Files
- `frontend/index.html` — The website (deploy on GitHub Pages)
- `frontend/deploy-guide.html` — Step-by-step deployment guide
- `backend/app.py` — Python Flask server (deploy on Render.com)
- `backend/requirements.txt` — Python dependencies
- `backend/render.yaml` — Render.com auto-deploy config

## Quick Start
1. Push this folder to a GitHub repo
2. Deploy `backend/` on Render.com (free)
3. Set up Cloudflare R2 for video storage (free)
4. Enable GitHub Pages for `frontend/` folder
5. Update BACKEND_URL in index.html with your Render URL
6. Open deploy-guide.html in your browser for full instructions

## Rename
Search and replace "VidAI" in index.html with your chosen name (Ctrl+H in VS Code)

## Stack
- Frontend: HTML/CSS/JS (GitHub Pages)
- Backend: Python + Flask + Gunicorn (Render.com)
- AI Script: Ollama / OpenAI GPT-3.5
- Voice: Microsoft Edge TTS (free, 300+ voices)
- Images: Pexels API (free)
- Video: MoviePy + FFmpeg
- Storage: Cloudflare R2 (free 10GB)
