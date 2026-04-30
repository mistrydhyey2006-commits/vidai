"""
VidAI Backend — Flask server for AI video generation
Deploy this on Render.com (free tier)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import uuid
import os
import requests
from groq import Groq
from gtts import gTTS
from moviepy.editor import (
    ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
)
from PIL import Image

# Fix for newer Pillow versions - patch ANTIALIAS
from PIL import Image as PILImage
if not hasattr(PILImage, 'ANTIALIAS'):
    PILImage.ANTIALIAS = PILImage.LANCZOS

import boto3  # For Cloudflare R2 storage
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

# ─── CONFIG ─────────────────────────────────────────────────────────────────
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "your_pexels_key_here")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Backblaze B2 storage (S3-compatible, free 10GB, no card needed)
R2_ENDPOINT = os.environ.get("B2_ENDPOINT", "")       # e.g. https://s3.us-west-004.backblazeb2.com
R2_ACCESS_KEY = os.environ.get("B2_KEY_ID", "")       # keyID from Backblaze
R2_SECRET_KEY = os.environ.get("B2_APP_KEY", "")      # applicationKey from Backblaze
R2_BUCKET = os.environ.get("B2_BUCKET", "vidai-videos")
R2_PUBLIC_URL = os.environ.get("B2_PUBLIC_URL", "")   # public URL of your bucket

# In-memory job store (use Redis in production)
jobs = {}

# ─── ROUTES ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "VidAI backend running"})


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    topic = data.get("topic", "").strip()
    duration = int(data.get("duration", 60))
    voice = data.get("voice", "en-IN-NeerjaNeural")
    style = data.get("style", "educational")

    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    if len(topic) > 300:
        return jsonify({"error": "Topic too long (max 300 chars)"}), 400
    if duration < 30 or duration > 3600:
        return jsonify({"error": "Duration must be between 30 seconds and 1 hour"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "step": "queued",
        "topic": topic,
        "video_url": None,
        "error": None,
    }

    # Run generation in background thread
    thread = threading.Thread(
        target=run_pipeline,
        args=(job_id, topic, duration, voice, style),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>", methods=["GET"])
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/video/<job_id>", methods=["GET"])
def serve_video(job_id):
    from flask import send_file
    video_path = os.path.join(VIDEO_DIR, f"{job_id}.mp4")
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404
    return send_file(
        video_path,
        mimetype="video/mp4",
        as_attachment=False,
        download_name=f"vidai_{job_id[:8]}.mp4",
    )


# ─── PIPELINE ───────────────────────────────────────────────────────────────

def run_pipeline(job_id, topic, duration, voice, style):
    """Main video generation pipeline."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:

            # STEP 1: Write script
            update_job(job_id, step="writing_script")
            script = write_script(topic, duration, style)
            logger.info(f"[{job_id}] Script written: {len(script)} chars")

            # STEP 2: Fetch images
            update_job(job_id, step="fetching_images")
            # Scale media count with duration
            media_count = max(6, min(40, duration // 10))

            # Try to get video clips first (more engaging)
            video_clips = fetch_pexels_videos(topic, count=max(3, media_count//3), tmpdir=tmpdir)
            logger.info(f"[{job_id}] Fetched {len(video_clips)} video clips")

            # Get images to fill the rest
            image_count = max(6, media_count - len(video_clips))
            image_paths = fetch_images(topic, count=image_count, tmpdir=tmpdir)
            logger.info(f"[{job_id}] Fetched {len(image_paths)} images")

            # STEP 3: Generate voiceover
            update_job(job_id, step="generating_voice")
            audio_path = os.path.join(tmpdir, "narration.mp3")
            generate_voice(script, voice, audio_path)
            logger.info(f"[{job_id}] Voice generated")

            # STEP 4: Render video
            update_job(job_id, step="rendering")
            output_path = os.path.join(tmpdir, f"{job_id}.mp4")
            render_video(image_paths, audio_path, output_path, script=script, video_clips=video_clips)
            logger.info(f"[{job_id}] Video rendered")

            # STEP 5: Upload to R2
            video_url = upload_to_r2(output_path, job_id)
            logger.info(f"[{job_id}] Uploaded: {video_url}")

            jobs[job_id].update({
                "status": "done",
                "step": "done",
                "video_url": video_url,
            })

    except Exception as e:
        logger.error(f"[{job_id}] Pipeline error: {e}")
        jobs[job_id].update({
            "status": "error",
            "error": str(e),
        })


def update_job(job_id, step):
    jobs[job_id]["step"] = step
    jobs[job_id]["status"] = "running"


# ─── AI SCRIPT ──────────────────────────────────────────────────────────────

def write_script(topic, duration, style):
    word_count = int(duration * 2.2)  # ~2.2 words/second narration pace

    style_prompts = {
        "educational": "informative and clear, like a teacher explaining to students",
        "storytelling": "narrative and engaging, like telling a fascinating story",
        "documentary": "serious and authoritative, like a BBC documentary narrator",
        "casual": "friendly and conversational, like explaining to a friend",
    }
    style_desc = style_prompts.get(style, "informative and clear")

    prompt = f"""Write a video narration script about: "{topic}"

Style: {style_desc}
Target length: approximately {word_count} words (for a {duration}-second video)

Rules:
- Write ONLY the narration text, no stage directions or scene descriptions
- No [IMAGE:] tags, no [MUSIC:] notes, no speaker labels
- Just the words that will be spoken out loud
- Start with an engaging opening sentence
- End with a memorable closing sentence
- Keep sentences relatively short for natural speech

Write the narration now:"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq failed ({e}), using fallback script")
        return f"Welcome to this video about {topic}. {topic} is a fascinating subject that has shaped our world in many ways. Let's explore the key aspects, history, and significance of this topic together. Throughout this video, we'll discover what makes {topic} so important and interesting. Thank you for watching."



# ─── IMAGES ─────────────────────────────────────────────────────────────────

def fetch_pexels_videos(topic, count, tmpdir):
    """Fetch short video clips from Pexels."""
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/videos/search?query={topic}&per_page={min(count,15)}&orientation=landscape"
    clips = []
    try:
        res = requests.get(url, headers=headers, timeout=10).json()
        videos = res.get("videos", [])
        for i, video in enumerate(videos[:count]):
            # Get smallest SD file
            files = sorted(video.get("video_files", []), key=lambda x: x.get("width", 9999))
            sd_files = [f for f in files if f.get("width", 0) <= 1280 and f.get("link")]
            if not sd_files:
                continue
            vid_url = sd_files[0]["link"]
            path = os.path.join(tmpdir, f"clip_{i}.mp4")
            try:
                vid_data = requests.get(vid_url, timeout=30).content
                with open(path, "wb") as f:
                    f.write(vid_data)
                clips.append(path)
                logger.info(f"Downloaded video clip {i}")
            except Exception as e:
                logger.warning(f"Failed to download video {i}: {e}")
    except Exception as e:
        logger.warning(f"Pexels video search failed: {e}")
    return clips


def fetch_images(topic, count, tmpdir):
    headers = {"Authorization": PEXELS_API_KEY}
    photos = []

    per_page = min(count, 80)
    pages_needed = (count + per_page - 1) // per_page

    for page in range(1, pages_needed + 1):
        remaining = count - len(photos)
        fetch = min(remaining, 80)
        url = f"https://api.pexels.com/v1/search?query={topic}&per_page={fetch}&page={page}&orientation=landscape"
        try:
            res = requests.get(url, headers=headers, timeout=10).json()
            photos.extend(res.get("photos", []))
        except Exception as e:
            logger.warning(f"Pexels page {page} failed: {e}")
        if len(photos) >= count:
            break

    paths = []
    downloaded = []
    for i, photo in enumerate(photos[:count]):
        img_url = photo["src"]["large"]
        path = os.path.join(tmpdir, f"img_{i}.jpg")
        try:
            img_data = requests.get(img_url, timeout=15).content
            with open(path, "wb") as f:
                f.write(img_data)
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize((854, 480), Image.LANCZOS)
                img.save(path, "JPEG", quality=90)
            paths.append(path)
            downloaded.append(path)
        except Exception as e:
            logger.warning(f"Failed to download image {i}: {e}")

    if downloaded and len(paths) < count:
        while len(paths) < count:
            paths.append(downloaded[len(paths) % len(downloaded)])

    if not paths:
        colors = [(30,35,60),(20,50,80),(40,20,60),(20,60,40),(60,30,20),(50,50,20)]
        for i in range(count):
            color = colors[i % len(colors)]
            path = os.path.join(tmpdir, f"placeholder_{i}.jpg")
            img = Image.new("RGB", (854, 480), color)
            img.save(path, "JPEG")
            paths.append(path)

    return paths


# ─── VOICE ──────────────────────────────────────────────────────────────────

def generate_voice(text, voice, output_path):
    # Map voice names to gTTS language codes
    lang_map = {
        'en-IN-NeerjaNeural': 'en',
        'en-IN-PrabhatNeural': 'en',
        'en-US-JennyNeural': 'en',
        'en-US-GuyNeural': 'en',
        'en-GB-SoniaNeural': 'en',
    }
    lang = lang_map.get(voice, 'en')
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)


# ─── RENDER ─────────────────────────────────────────────────────────────────
# Version 7 - Added zoom/pan effects, subtitles, background music

def make_zoom_clip(path, duration, zoom_direction="in"):
    """Create a zoom in or pan effect on an image clip."""
    import numpy as np
    W, H = 854, 480
    clip = ImageClip(path).set_duration(duration)

    def zoom_in(get_frame, t):
        frame = get_frame(t)
        progress = t / duration
        scale = 1.0 + 0.08 * progress  # zoom from 1x to 1.08x
        new_w = int(W * scale)
        new_h = int(H * scale)
        resized = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
        x = (new_w - W) // 2
        y = (new_h - H) // 2
        cropped = resized.crop((x, y, x + W, y + H))
        return np.array(cropped)

    def pan_right(get_frame, t):
        frame = get_frame(t)
        progress = t / duration
        scale = 1.08
        new_w = int(W * scale)
        new_h = int(H * scale)
        resized = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
        x = int((new_w - W) * progress)
        y = (new_h - H) // 2
        cropped = resized.crop((x, y, x + W, y + H))
        return np.array(cropped)

    if zoom_direction == "in":
        return clip.fl(zoom_in).resize((W, H))
    else:
        return clip.fl(pan_right).resize((W, H))


def make_subtitle_clip(text, duration, W=854, H=480):
    """Create a subtitle text overlay."""
    from moviepy.editor import TextClip, CompositeVideoClip
    try:
        txt = TextClip(
            text[:80],
            fontsize=20,
            color="white",
            stroke_color="black",
            stroke_width=1.5,
            method="caption",
            size=(W - 60, None),
            font="DejaVu-Sans",
        ).set_duration(duration)
        txt = txt.set_position(("center", H - txt.h - 20))
        return txt
    except Exception as e:
        logger.warning(f"Subtitle failed: {e}")
        return None


def render_video(image_paths, audio_path, output_path, script=None, video_clips=None):
    from moviepy.editor import CompositeVideoClip, AudioFileClip as AFC
    import numpy as np

    audio = AudioFileClip(audio_path)
    total_duration = audio.duration

    # Each image shows for 5 seconds, loop images to fill full audio duration
    secs_per_image = 5
    total_slots = max(len(image_paths), int(total_duration / secs_per_image) + 1)
    looped_paths = [image_paths[i % len(image_paths)] for i in range(total_slots)]
    clip_duration = total_duration / total_slots

    # Split script into chunks for subtitles
    subtitle_chunks = []
    if script:
        words = script.split()
        words_per_clip = max(1, len(words) // total_slots)
        for i in range(total_slots):
            chunk = words[i * words_per_clip:(i + 1) * words_per_clip]
            subtitle_chunks.append(" ".join(chunk))

    clips = []
    effects = ["in", "pan", "in", "pan", "in", "pan"]

    for i, path in enumerate(looped_paths):
        effect = effects[i % len(effects)]
        try:
            base = make_zoom_clip(path, clip_duration, effect)
        except Exception:
            base = ImageClip(path).set_duration(clip_duration).resize((854, 480))

        base = base.fadein(0.4).fadeout(0.4)

        # Add subtitle overlay
        layers = [base]
        if subtitle_chunks and i < len(subtitle_chunks):
            sub = make_subtitle_clip(subtitle_chunks[i], clip_duration)
            if sub:
                layers.append(sub)

        if len(layers) > 1:
            final_clip = CompositeVideoClip(layers, size=(854, 480)).set_duration(clip_duration)
        else:
            final_clip = base

        clips.append(final_clip)

    video = concatenate_videoclips(clips, method="compose")

    # Mix background music at low volume
    try:
        music_path = os.path.join(os.path.dirname(audio_path), "bg_music.mp3")
        bg_url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
        music_data = requests.get(bg_url, timeout=10).content
        with open(music_path, "wb") as f:
            f.write(music_data)
        bg = AudioFileClip(music_path).subclip(0, total_duration).volumex(0.08)
        from moviepy.editor import CompositeAudioClip
        mixed_audio = CompositeAudioClip([audio.volumex(1.0), bg])
        video = video.set_audio(mixed_audio)
        logger.info("Background music added!")
    except Exception as e:
        logger.warning(f"Background music failed: {e}, using voice only")
        video = video.set_audio(audio)

    video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=output_path.replace(".mp4", "_temp_audio.m4a"),
        remove_temp=True,
        logger=None,
        preset="ultrafast",
        threads=1,
        bitrate="600k",
    )

    audio.close()
    video.close()


# ─── UPLOAD ─────────────────────────────────────────────────────────────────

# Store videos on disk for serving
VIDEO_DIR = "/tmp/vidai_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

def upload_to_r2(file_path, job_id):
    """Copy video to persistent disk location and return backend URL."""
    dest = os.path.join(VIDEO_DIR, f"{job_id}.mp4")
    import shutil
    shutil.copy2(file_path, dest)
    logger.info(f"Video saved to disk: {dest} ({os.path.getsize(dest)} bytes)")
    backend_url = os.environ.get("BACKEND_URL", "https://vidai.onrender.com")
    return f"{backend_url}/api/video/{job_id}"


# ─── ENTRY ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
