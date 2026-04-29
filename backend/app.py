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
            # Scale image count with duration: ~1 image per 10 seconds, min 6, max 80
            image_count = max(6, min(80, duration // 10))
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
            render_video(image_paths, audio_path, output_path)
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

def fetch_images(topic, count, tmpdir):
    headers = {"Authorization": PEXELS_API_KEY}
    photos = []

    # Pexels max per_page is 80 — paginate if we need more
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

    # If we still don't have enough, loop existing photos to fill
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
                img = img.resize((1920, 1080), Image.LANCZOS)
                img.save(path, "JPEG", quality=90)
            paths.append(path)
            downloaded.append(path)
        except Exception as e:
            logger.warning(f"Failed to download image {i}: {e}")

    # If we need more images than Pexels returned, loop what we have
    if downloaded and len(paths) < count:
        while len(paths) < count:
            paths.append(downloaded[len(paths) % len(downloaded)])

    # Fallback: generate solid-color placeholder images if none downloaded
    if not paths:
        colors = [(30,35,60),(20,50,80),(40,20,60),(20,60,40),(60,30,20),(50,50,20)]
        for i in range(count):
            color = colors[i % len(colors)]
            path = os.path.join(tmpdir, f"placeholder_{i}.jpg")
            img = Image.new("RGB", (1920, 1080), color)
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

def render_video(image_paths, audio_path, output_path):
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration
    clip_duration = total_duration / len(image_paths)

    clips = []
    for path in image_paths:
        clip = (
            ImageClip(path)
            .set_duration(clip_duration)
            .fadein(0.5)
            .fadeout(0.5)
            .resize((1920, 1080))
        )
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")
    video = video.set_audio(audio)

    video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=output_path.replace(".mp4", "_temp_audio.m4a"),
        remove_temp=True,
        logger=None,  # Suppress moviepy output
        preset="fast",
    )

    audio.close()
    video.close()


# ─── UPLOAD ─────────────────────────────────────────────────────────────────

def upload_to_r2(file_path, job_id):
    """Upload to Backblaze B2 (S3-compatible) and return public URL."""
    if not R2_ENDPOINT or not R2_ACCESS_KEY:
        logger.warning("B2 not configured — returning placeholder")
        return f"/static/videos/{job_id}.mp4"

    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )

    key = f"videos/{job_id}.mp4"
    s3.upload_file(
        file_path,
        R2_BUCKET,
        key,
        ExtraArgs={"ContentType": "video/mp4", "ACL": "public-read"},
    )

    return f"{R2_PUBLIC_URL}/file/{R2_BUCKET}/{key}"


# ─── ENTRY ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
