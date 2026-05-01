"""
VidAI Backend v11
Fixes over v10:
  1. Video duration now driven by actual audio length (not slot count)
  2. Pexels search uses enriched topic keywords for relevant visuals
  3. Zoom reduced from 1.08 → 1.04 to eliminate blur on low-res images
  4. Subtitle stroke_width increased to 2.5 + font size 24 for readability
  5. Image resolution fetched at "large2" (1880px) instead of "large" (940px)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import uuid
import os
import requests
import shutil
from groq import Groq
from gtts import gTTS
from moviepy.editor import (
    ImageClip, AudioFileClip, VideoFileClip,
    concatenate_videoclips, CompositeVideoClip, CompositeAudioClip
)
from PIL import Image as PILImage
import numpy as np
import tempfile
import logging

if not hasattr(PILImage, 'ANTIALIAS'):
    PILImage.ANTIALIAS = PILImage.LANCZOS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
BACKEND_URL    = os.environ.get("BACKEND_URL", "https://vidai.onrender.com")

VIDEO_DIR = "/tmp/vidai_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

jobs = {}

W, H = 854, 480
SECS_PER_SLOT = 5


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "VidAI v11 running"})


@app.route("/api/generate", methods=["POST"])
def generate():
    data     = request.json or {}
    topic    = data.get("topic", "").strip()
    duration = int(data.get("duration", 60))
    voice    = data.get("voice", "en")
    style    = data.get("style", "educational")

    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    if len(topic) > 300:
        return jsonify({"error": "Topic too long (max 300 chars)"}), 400
    if duration < 30 or duration > 300:
        return jsonify({"error": "Duration must be 30–300 seconds"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued", "step": "queued",
        "topic": topic, "video_url": None, "error": None,
    }

    thread = threading.Thread(
        target=run_pipeline,
        args=(job_id, topic, duration, voice, style),
        daemon=True,
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
        video_path, mimetype="video/mp4",
        as_attachment=False, download_name=f"vidai_{job_id[:8]}.mp4",
    )


# ─── PIPELINE ────────────────────────────────────────────────────────────────

def run_pipeline(job_id, topic, duration, voice, style):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:

            update_job(job_id, "writing_script")
            script = write_script(topic, duration, style)
            logger.info(f"[{job_id}] Script: {len(script.split())} words")

            update_job(job_id, "fetching_media")
            total_slots = max(8, duration // 5)
            video_want  = max(3, total_slots // 3)
            image_want  = total_slots - video_want

            video_clips = fetch_pexels_videos(topic, count=video_want, tmpdir=tmpdir)
            logger.info(f"[{job_id}] Got {len(video_clips)} video clips")

            image_paths = fetch_images(topic, count=max(6, image_want), tmpdir=tmpdir)
            logger.info(f"[{job_id}] Got {len(image_paths)} images")

            update_job(job_id, "generating_voice")
            audio_path = os.path.join(tmpdir, "narration.mp3")
            generate_voice(script, voice, audio_path)
            logger.info(f"[{job_id}] Voice generated")

            update_job(job_id, "rendering")
            output_path = os.path.join(tmpdir, f"{job_id}.mp4")
            render_video(
                image_paths=image_paths,
                video_clips=video_clips,
                audio_path=audio_path,
                output_path=output_path,
                script=script,
            )
            logger.info(f"[{job_id}] Render done")

            dest = os.path.join(VIDEO_DIR, f"{job_id}.mp4")
            shutil.copy2(output_path, dest)
            video_url = f"{BACKEND_URL}/api/video/{job_id}"

            jobs[job_id].update({
                "status": "done", "step": "done", "video_url": video_url,
            })

    except Exception as e:
        logger.error(f"[{job_id}] Pipeline error: {e}", exc_info=True)
        jobs[job_id].update({"status": "error", "error": str(e)})


def update_job(job_id, step):
    jobs[job_id]["step"]   = step
    jobs[job_id]["status"] = "running"


# ─── SCRIPT ──────────────────────────────────────────────────────────────────

def write_script(topic, duration, style):
    target_words = int(duration * 2.5)
    min_words    = int(target_words * 0.85)

    style_prompts = {
        "educational":  "informative and clear, like a teacher explaining to students",
        "storytelling": "narrative and engaging, like telling a fascinating story",
        "documentary":  "serious and authoritative, like a BBC documentary narrator",
        "casual":       "friendly and conversational, like explaining to a friend",
    }
    style_desc = style_prompts.get(style, "informative and clear")

    def build_prompt(words_needed, extra=""):
        return f"""Write a video narration script about: "{topic}"

Style: {style_desc}
Required word count: {words_needed} words MINIMUM. This is MANDATORY.
{extra}

Rules:
- Write ONLY the narration text that will be spoken out loud
- No stage directions, no [IMAGE:] tags, no speaker labels
- Short sentences work best for natural speech
- Start with an engaging opening sentence
- End with a memorable closing sentence
- Keep writing until you reach {words_needed} words — do NOT stop early

Begin the narration now:"""

    client = Groq(api_key=GROQ_API_KEY)

    def call_groq(prompt):
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()

    try:
        script = call_groq(build_prompt(target_words))
        actual = len(script.split())
        logger.info(f"Script attempt 1: {actual} words (target {target_words})")

        if actual < min_words:
            shortage  = target_words - actual
            extra     = f"IMPORTANT: Your previous attempt was too short by {shortage} words. Write more detail, more examples, and expand every section."
            extension = call_groq(build_prompt(shortage + 50, extra))
            script    = script + " " + extension
            logger.info(f"Script after extension: {len(script.split())} words")

        return script

    except Exception as e:
        logger.warning(f"Groq failed ({e}), using fallback script")
        base = (
            f"Welcome to this exploration of {topic}. "
            f"{topic} is one of the most fascinating subjects you can study. "
            f"Let us dive deep into everything there is to know about {topic}. "
            f"From its origins to its modern significance, {topic} continues to shape our world. "
            f"Experts have long studied {topic} and uncovered remarkable insights. "
            f"As we continue, you will discover why {topic} matters so much today. "
            f"The history of {topic} stretches back further than most people realise. "
            f"Understanding {topic} opens up a whole new way of seeing things around us. "
            f"Thank you for joining this journey through the world of {topic}."
        )
        words_per_repeat = len(base.split())
        repeats = max(1, target_words // words_per_repeat + 1)
        return " ".join([base] * repeats)


# ─── SEARCH QUERY ENRICHMENT ─────────────────────────────────────────────────

def enrich_query(topic):
    """
    FIX 2: Raw topic titles give bad Pexels results.
    Ask Groq for 3 specific visual search terms related to the topic.
    Falls back to simple keyword extraction if Groq fails.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{
                "role": "user",
                "content": (
                    f'Give me 3 short Pexels image search queries (2-4 words each) '
                    f'for a video about: "{topic}". '
                    f'Return ONLY a comma-separated list, nothing else. '
                    f'Example format: space galaxy stars, black hole NASA, cosmic nebula'
                )
            }],
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip()
        queries = [q.strip() for q in raw.split(",") if q.strip()]
        if queries:
            logger.info(f"Enriched queries: {queries}")
            return queries
    except Exception as e:
        logger.warning(f"Query enrichment failed: {e}")

    # Fallback: use topic as-is
    return [topic]


# ─── MEDIA FETCH ─────────────────────────────────────────────────────────────

def fetch_pexels_videos(topic, count, tmpdir):
    if not PEXELS_API_KEY:
        return []

    queries = enrich_query(topic)
    headers = {"Authorization": PEXELS_API_KEY}
    clips   = []

    for query in queries:
        if len(clips) >= count:
            break
        url = f"https://api.pexels.com/videos/search?query={query}&per_page=5&orientation=landscape"
        try:
            res = requests.get(url, headers=headers, timeout=10).json()
            for video in res.get("videos", []):
                if len(clips) >= count:
                    break
                files = sorted(video.get("video_files", []), key=lambda x: x.get("width", 9999))
                sd    = [f for f in files if 0 < f.get("width", 0) <= 1280 and f.get("link")]
                if not sd:
                    continue
                path = os.path.join(tmpdir, f"clip_{len(clips)}.mp4")
                try:
                    data = requests.get(sd[0]["link"], timeout=30).content
                    with open(path, "wb") as f:
                        f.write(data)
                    clips.append(path)
                except Exception as e:
                    logger.warning(f"Video clip download failed: {e}")
        except Exception as e:
            logger.warning(f"Pexels video search failed for '{query}': {e}")

    return clips


def fetch_images(topic, count, tmpdir):
    if not PEXELS_API_KEY:
        return _placeholder_images(count, tmpdir)

    queries = enrich_query(topic)
    headers = {"Authorization": PEXELS_API_KEY}
    photos  = []

    for query in queries:
        if len(photos) >= count:
            break
        want = min(count - len(photos), 30)
        url  = (f"https://api.pexels.com/v1/search"
                f"?query={query}&per_page={want}&page=1&orientation=landscape")
        try:
            res   = requests.get(url, headers=headers, timeout=10).json()
            batch = res.get("photos", [])
            photos.extend(batch)
        except Exception as e:
            logger.warning(f"Pexels image search failed for '{query}': {e}")

    paths = []
    for i, photo in enumerate(photos[:count]):
        # FIX 3: Use large2 (1880px wide) for sharper images before downscale
        img_url = photo["src"].get("large2", photo["src"]["large"])
        path    = os.path.join(tmpdir, f"img_{i}.jpg")
        try:
            img_data = requests.get(img_url, timeout=15).content
            with open(path, "wb") as f:
                f.write(img_data)
            with PILImage.open(path) as img:
                img = img.convert("RGB").resize((W, H), PILImage.LANCZOS)
                img.save(path, "JPEG", quality=92)
            paths.append(path)
        except Exception as e:
            logger.warning(f"Image {i} failed: {e}")

    if not paths:
        return _placeholder_images(count, tmpdir)

    while len(paths) < count:
        paths.append(paths[len(paths) % len(paths)])

    return paths


def _placeholder_images(count, tmpdir):
    colors = [
        (30, 35, 60), (20, 50, 80), (40, 20, 60),
        (20, 60, 40), (60, 30, 20), (50, 50, 20),
    ]
    paths = []
    for i in range(count):
        path = os.path.join(tmpdir, f"placeholder_{i}.jpg")
        img  = PILImage.new("RGB", (W, H), colors[i % len(colors)])
        img.save(path, "JPEG")
        paths.append(path)
    return paths


# ─── VOICE ───────────────────────────────────────────────────────────────────

def generate_voice(text, voice, output_path):
    lang_map = {
        "en-IN-NeerjaNeural":  "en",
        "en-IN-PrabhatNeural": "en",
        "en-US-JennyNeural":   "en",
        "en-US-GuyNeural":     "en",
        "en-GB-SoniaNeural":   "en",
        "en": "en",
    }
    lang = lang_map.get(voice, "en")
    tts  = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)


# ─── RENDER ──────────────────────────────────────────────────────────────────

def make_zoom_clip(path, duration, direction="in"):
    """
    FIX 3: Zoom scale reduced from 1.08 → 1.04 to prevent blur on low-res images.
    """
    clip = ImageClip(path).set_duration(duration)

    def zoom_in(get_frame, t):
        frame    = get_frame(t)
        progress = t / max(duration, 0.01)
        scale    = 1.0 + 0.04 * progress   # was 0.08 — halved to reduce blur
        nw, nh   = int(W * scale), int(H * scale)
        resized  = PILImage.fromarray(frame).resize((nw, nh), PILImage.LANCZOS)
        x, y     = (nw - W) // 2, (nh - H) // 2
        return np.array(resized.crop((x, y, x + W, y + H)))

    def pan_right(get_frame, t):
        frame    = get_frame(t)
        progress = t / max(duration, 0.01)
        scale    = 1.04                     # was 1.08
        nw, nh   = int(W * scale), int(H * scale)
        resized  = PILImage.fromarray(frame).resize((nw, nh), PILImage.LANCZOS)
        x        = int((nw - W) * progress)
        y        = (nh - H) // 2
        return np.array(resized.crop((x, y, x + W, y + H)))

    fn = zoom_in if direction == "in" else pan_right
    return clip.fl(fn).resize((W, H))


def make_subtitle_clip(text, duration):
    """
    FIX 4: Larger font (24→28), thicker stroke (1.5→2.5), white fill.
    Makes subtitles readable over both dark and light footage.
    """
    from moviepy.editor import TextClip
    try:
        txt = TextClip(
            text[:90],
            fontsize=28,            # was 20
            color="white",
            stroke_color="black",
            stroke_width=2.5,       # was 1.5
            method="caption",
            size=(W - 80, None),
            font="DejaVu-Sans-Bold", # Bold variant for more contrast
        ).set_duration(duration)
        return txt.set_position(("center", H - txt.h - 25))
    except Exception as e:
        logger.warning(f"Subtitle render failed: {e}")
        return None


def render_video(image_paths, video_clips, audio_path, output_path, script=None):
    """
    FIX 1: total_duration comes from actual audio file, not slot arithmetic.
    Video is always trimmed/extended to match audio exactly.
    """
    audio          = AudioFileClip(audio_path)
    total_duration = audio.duration   # SOURCE OF TRUTH — drives everything
    total_slots    = max(4, int(total_duration / SECS_PER_SLOT) + 1)

    logger.info(f"Render: audio={total_duration:.1f}s → {total_slots} slots")

    # ── Build media schedule ─────────────────────────────────────────────────
    schedule = []
    vid_idx  = 0
    img_idx  = 0

    for i in range(total_slots):
        if video_clips and vid_idx < len(video_clips) and i % 3 == 1:
            schedule.append(("video", video_clips[vid_idx]))
            vid_idx += 1
        else:
            schedule.append(("image", image_paths[img_idx % len(image_paths)]))
            img_idx += 1

    # ── Subtitle chunks ──────────────────────────────────────────────────────
    subtitle_chunks = []
    if script:
        words          = script.split()
        words_per_slot = max(1, len(words) // total_slots)
        for i in range(total_slots):
            chunk = words[i * words_per_slot:(i + 1) * words_per_slot]
            subtitle_chunks.append(" ".join(chunk))

    # ── Build clips ──────────────────────────────────────────────────────────
    clips   = []
    effects = ["in", "pan", "in", "pan", "in", "pan"]

    for i, (media_type, path) in enumerate(schedule):
        # Last slot gets remaining duration so video matches audio exactly
        if i == len(schedule) - 1:
            slot_dur = total_duration - (SECS_PER_SLOT * (len(schedule) - 1))
            slot_dur = max(1.0, slot_dur)
        else:
            slot_dur = SECS_PER_SLOT

        try:
            if media_type == "video":
                vc       = VideoFileClip(path)
                clip_dur = min(slot_dur, vc.duration)
                base     = vc.subclip(0, clip_dur).resize((W, H)).set_duration(slot_dur)
            else:
                direction = effects[i % len(effects)]
                base      = make_zoom_clip(path, slot_dur, direction)

        except Exception as e:
            logger.warning(f"Slot {i} ({media_type}) failed: {e} — fallback image")
            fallback = image_paths[i % len(image_paths)]
            base     = ImageClip(fallback).set_duration(slot_dur).resize((W, H))

        base = base.fadein(0.3).fadeout(0.3)

        layers = [base]
        if subtitle_chunks and i < len(subtitle_chunks) and subtitle_chunks[i]:
            sub = make_subtitle_clip(subtitle_chunks[i], slot_dur)
            if sub:
                layers.append(sub)

        final = CompositeVideoClip(layers, size=(W, H)).set_duration(slot_dur) if len(layers) > 1 else base
        clips.append(final)

    # ── Concatenate ───────────────────────────────────────────────────────────
    video = concatenate_videoclips(clips, method="compose")

    # Safety trim: ensure video never exceeds audio
    if video.duration > total_duration:
        video = video.subclip(0, total_duration)

    # ── Background music ──────────────────────────────────────────────────────
    try:
        music_path = os.path.join(os.path.dirname(audio_path), "bg_music.mp3")
        bg_data    = requests.get(
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
            timeout=15,
        ).content
        with open(music_path, "wb") as f:
            f.write(bg_data)
        bg          = AudioFileClip(music_path).subclip(0, total_duration).volumex(0.07)
        mixed_audio = CompositeAudioClip([audio.volumex(1.0), bg])
        video       = video.set_audio(mixed_audio)
        logger.info("Background music mixed in")
    except Exception as e:
        logger.warning(f"BG music skipped: {e}")
        video = video.set_audio(audio)

    # ── Write ─────────────────────────────────────────────────────────────────
    video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=output_path.replace(".mp4", "_tmp_audio.m4a"),
        remove_temp=True,
        logger=None,
        preset="ultrafast",
        threads=2,
        bitrate="700k",
    )

    audio.close()
    video.close()
    logger.info(f"Video written: {output_path}")


# ─── ENTRY ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
