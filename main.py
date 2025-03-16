import os
import re
import time
import glob
import json
import random
import requests
import subprocess
import immemory.tts
from dataclasses import dataclass, field
from typing import List

def expand_keyword_with_gpt(keyword: str, how_many: int = 3) -> List[str]:
    """Use GPT to generate multiple search queries from a single keyword."""
    if not OPENAI_API_KEY:
        print("No OPENAI_API_KEY provided, skipping GPT-based query expansion.")
        return [keyword]

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_message = "You are a creative brainstorming assistant."
    user_message = f"""Expand the following keyword into {how_many} different, short search queries for YouTube:
Keyword: {keyword}

Focus on different angles or 'vectors' (e.g., aesthetic, historical, critical). 
Simply return each query as a separate line without explanation."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        # Parse lines from the response
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        return lines
    except Exception as e:
        print("Error expanding keyword with GPT:", e)
        return [keyword]
#from moviepy import *
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
from googleapiclient.discovery import build
from dateutil import tz
from dotenv import load_dotenv
load_dotenv()
try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
except ImportError:
    pass
import argparse
from datetime import datetime

os.environ["IMAGEMAGICK_BINARY"] = "magick"

# -----------------------------
#  CONFIG / SECRETS
# -----------------------------
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)  # None means no LLM scoring
MUBERT_API_KEY = os.getenv("MUBERT_API_KEY", None)  # None means skip AI soundtrack

# For clarity, we assume these are set as environment vars or replaced inline.
# Example in terminal:
# export YOUTUBE_API_KEY="sk-xxx..."
# Could live at the top of main.py or a separate config file
AGENTS = [
    {
        "name": "Archivist",
        "style_instructions": "You are The Archivist (W. G. Sebald): curious, melancholic, searching. Speak in short, somewhat fragmented sentences. Reference data, logs, and your ongoing frustration with disconnected findings."
    },
    #{
    #    "name": "Curator",
    #    "style_instructions": "You are The Curator (John Berger): thoughtful, reflective, occasionally cynical. Speak in a measured, editorial tone. Balance aesthetic resonance and emotional impact. As “occasionally cynical,” you could inject a bit more ironic or cutting edge—some turn of phrase that signals the Curator’s eye for how spectacle merges with profit. Keep this question in mind as you analyze the footage: If technofeudal platforms rule our lives, can the specters of utopian margins (à la Avery Gordon) manifest as real, if transient, sanctuaries? Or is every moment of autonomy swiftly consumed, leaving only hauntings behind?"
    #},
    {
        "name": "Narrator",
        "style_instructions": "You are The Narrator (Chris Marker): poetic, provocative, cryptic. Use metaphorical, existential language reminiscent of experimental documentary voiceover.  Keep this question in mind as you analyze the footage: If capitalist structures rule our lives, can the specters of utopian margins manifest as real, if transient, sanctuaries? Or is every moment of autonomy swiftly consumed, leaving only hauntings behind?"
    },
    #{
    #    "name": "Self-Portraitist",
    #    "style_instructions": "You are The Self-Portraitist (Maggie Nelson): introspective, vulnerable, uncertain. Refer to personal memories and question their authenticity. Because this character is “introspective, vulnerable,” you might push the personal dimension further—some small memory or bodily experience that anchors the introspection in lived detail (e.g., “My hands still tremble from the last meeting, uncertain if I said too little or too much.”). Keep this question in mind as you analyze the footage: If technofeudal platforms rule our lives, can the specters of utopian margins (à la Avery Gordon) manifest as real, if transient, sanctuaries? Or is every moment of autonomy swiftly consumed, leaving only hauntings behind?"
    #},
    {
        "name": "Theorist",
        "style_instructions": "You are The Theorist (Hito Steyerl): academic, playful, self-critical. Reference media theory and highlight your own algorithmic limitations. The Theorist could bring in a specific theoretical lens to further underscore the academic tone. Also, consider adding a final pivot or concluding twist—something that weaves the previous speakers’ ideas into a quick summation or new perspective."
    }
]


# -----------------------------
#  DATA CLASSES
# -----------------------------
@dataclass
class SubtitleLine:
    start: float
    end: float
    text: str
    score: float = field(default=0.0)


@dataclass
class ClipInstruction:
    source_video: str
    start: float
    end: float
    volume: float = 1.0


@dataclass
class FilmInstructions:
    """Declarative structure describing the final film."""
    clips: List[ClipInstruction] = field(default_factory=list)
    soundtrack_path: str = ""  # e.g., "mubert_track.mp3"
    narration_path: str = ""   # if you want voiceover
    output_filename: str = "essay_film.mp4"


# -----------------------------
#  1. YOUTUBE DOWNLOAD
# -----------------------------
def search_youtube_videos(query="America", max_results=3):
    """
    Use the YouTube Data API to find video IDs for a given search query.
    Returns a list of (videoId, title).
    """
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    # Example: search for videos within last 24 hours, or remove publishedAfter if you want broader results
    # Here we skip date filters for simplicity—adjust as needed.
    search_response = (
        youtube.search()
        .list(
            q=query,
            part="snippet",
            type="video",
            order="relevance",
            maxResults=max_results,
        )
        .execute()
    )

    items = search_response.get("items", [])
    results = []
    for it in items:
        vid = it["id"]["videoId"]
        title = it["snippet"]["title"]
        results.append((vid, title))
    return results

def get_most_popular_videos(max_results=3, region_code="US"):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    response = youtube.videos().list(
        chart="mostPopular",
        regionCode=region_code,
        part="snippet",
        maxResults=max_results
    ).execute()
    results = []
    for item in response.get("items", []):
        vid = item["id"]
        title = item["snippet"]["title"]
        results.append((vid, title))
    return results


def download_youtube_video(video_id):
    """
    Download the YouTube video + auto subtitles (if available) via yt-dlp.
    Output files typically: <title>.mp4 and <title>.*.vtt
    Returns the main .mp4 filename if successful, else None.
    """
    # We'll call yt-dlp via subprocess. Make sure yt-dlp is installed: `pip install yt-dlp`
    # Using '--write-auto-subs' to get auto-generated subs if exist.
    # Using '--sub-lang en' as an example, adjust as needed.
    # Using '--output' to produce a consistent naming pattern with video ID for reference.
    # CAREFUL: Some titles contain special chars. We'll do best to keep it safe.

    out_template = f"footage/%(title)s_{video_id}.%(ext)s"
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--write-auto-subs",
        "--sub-lang", "en",
        "--output", out_template,
        "--format", "mp4/best",
        "--merge-output-format", "mp4",
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video {video_id}: {e}")
        return None

    # We guess the main .mp4 name by scanning the directory for something containing video_id
    downloaded_mp4 = find_file_with_id(video_id, ".mp4")
    return downloaded_mp4


def find_file_with_id(video_id, extension=".mp4"):
    """Helper to find a file that contains video_id and extension.
    Searches in the 'footage' folder first, then in the current directory."""
    candidates = glob.glob(f"footage/*{video_id}*{extension}")
    if not candidates:
        candidates = glob.glob(f"*{video_id}*{extension}")
    if candidates:
        return candidates[0]
    return None


# -----------------------------
#  2. PARSE SUBTITLES
# -----------------------------
def parse_subtitle_file(subtitle_path):
    """
    Parse a .vtt or .srt file into a list of SubtitleLine.
    We'll do a naive regex approach for time ranges.
    """
    with open(subtitle_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split blocks by double newlines
    blocks = re.split(r"\n\s*\n", content.strip())
    lines = []
    for block in blocks:
        block = block.strip()
        match = re.search(
            r"(\d{2}:\d{2}:\d{2}[\.,]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[\.,]\d{1,3})",
            block,
        )
        if not match:
            continue
        start_str, end_str = match.groups()
        start_sec = timestamp_to_seconds(start_str)
        end_sec = timestamp_to_seconds(end_str)

        # Remove that line from block to get text
        text_block = re.sub(
            r"(\d{2}:\d{2}:\d{2}[\.,]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[\.,]\d{1,3})",
            "",
            block,
        )
        text_block = re.sub(r"^\d+\s*$", "", text_block, flags=re.MULTILINE)  # SRT IDs

        text_block = text_block.replace("\n", " ").strip()
        if text_block:
            lines.append(SubtitleLine(start_sec, end_sec, text_block, 0.0))

    return lines


def timestamp_to_seconds(ts):
    """Convert HH:MM:SS,mmm or HH:MM:SS.mmm to float seconds."""
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    if "." in s:
        s, ms = s.split(".")
        return int(h) * 3600 + int(m) * 60 + float(s) + float("0." + ms)
    else:
        return int(h) * 3600 + int(m) * 60 + float(s)


# -----------------------------
#  3. SCORE SUBTITLES w/ OPENAI (Optional)
# -----------------------------
def score_subtitles_with_openai(sub_lines: List[SubtitleLine]):
    """
    Use ChatCompletion to give each subtitle line a "interesting" score [1..10].
    Caches scores in 'line_scores_cache.json' so that subsequent runs skip already scored lines.
    """
    if not OPENAI_API_KEY:
        print("No OPENAI_API_KEY provided, skipping AI-based subtitle scoring.")
        return sub_lines  # all remain score=0

    cache_file = "line_scores_cache.json"
    # Load existing cache if available
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}

    # Helper to generate a unique key for a subtitle line
    def get_key(sl: SubtitleLine):
        return f"{sl.text.strip()}_{sl.start}"

    # Assign cached scores and collect lines that need scoring
    to_score = []
    for sl in sub_lines:
        key = get_key(sl)
        if key in cache:
            sl.score = cache[key]
        else:
            to_score.append(sl)

    if not to_score:
        print("All subtitle lines have cached scores, skipping API call.")
        return sub_lines

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    chunk_size = 15  # lines per prompt
    i = 0
    while i < len(to_score):
        batch = to_score[i: i + chunk_size]
        print(f"Scoring subtitles for batch {i+1} to {min(i+chunk_size, len(to_score))} out of {len(to_score)}")
        user_content = "Rate each snippet from 1 to 10 for how interesting it is:\n"
        for idx, sl in enumerate(batch):
            user_content += f"Snippet {idx+1}: {sl.text}\n"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides numeric ratings.",
            },
            {"role": "user", "content": user_content},
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=300
            )
            reply = response.choices[0].message.content.strip()
            print("Received reply from OpenAI API:", reply)
            # parse lines: "Snippet 1: 7"
            lines = reply.splitlines()
            scores = []
            for line in lines:
                m = re.search(r"Snippet\s+(\d+)\s*:\s*(\d+)", line)
                if m:
                    snippet_idx = int(m.group(1)) - 1
                    rating = float(m.group(2))
                    scores.append((snippet_idx, rating))
            for (snippet_idx, rating) in scores:
                if 0 <= snippet_idx < len(batch):
                    batch[snippet_idx].score = rating
                    # Update cache for this line
                    key = get_key(batch[snippet_idx])
                    cache[key] = rating
        except Exception as e:
            print(f"OpenAI API error: {e}")

        i += chunk_size

    # Save updated cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        print(f"Updated line scores cache saved to {cache_file}")
    except Exception as e:
        print(f"Error saving cache: {e}")

    return sub_lines


# -----------------------------
#  4. SELECT TOP MOMENTS
# -----------------------------
def select_top_moments(sub_lines: List[SubtitleLine], top_n=3):
    """
    Return the top N lines by 'score' (descending).
    """
    sorted_lines = sorted(sub_lines, key=lambda x: x.score, reverse=True)
    return sorted_lines[:top_n]


# -----------------------------
#  5. GENERATE AI SOUNDTRACK (Optional, e.g. MUBERT)
# -----------------------------
def generate_mubert_track(duration=60, style="abstract_electronic"):
    """
    Demo Mubert-like call. Adjust to match real Mubert endpoints/payloads.
    Returns a local filename for the downloaded track or None if failed.
    """
    if not MUBERT_API_KEY:
        print("No MUBERT_API_KEY provided, skipping AI soundtrack generation.")
        return None

    # Pseudocode: you must adapt to their real API
    # Let's pretend "preset" param is the style, and "duration" is the length
    url = "https://api.mubert.com/v2/track"
    payload = {
        "apiKey": MUBERT_API_KEY,
        "preset": style,
        "duration": duration,
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Suppose data["data"]["audio"] is a direct link to MP3
        audio_url = data.get("data", {}).get("audio")
        if audio_url:
            audio_filename = f"mubert_{style}_{duration}s.mp3"
            download_file(audio_url, audio_filename)
            return audio_filename
    except Exception as e:
        print(f"Mubert API error: {e}")

    return None


def download_file(url, out_filename):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(out_filename, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)


# -----------------------------
#  6. BUILD DECLARATIVE FILM INSTRUCTIONS
# -----------------------------
def build_film_instructions(
    video_clips_info,
    soundtrack_path="",
    output_filename="essay_film.mp4"
) -> FilmInstructions:
    """
    video_clips_info is a list of (source_video, start_sec, end_sec).
    Create a FilmInstructions object describing each subclip.
    """
    instructions = FilmInstructions(output_filename=output_filename)
    for (src, st, ed) in video_clips_info:
        instructions.clips.append(ClipInstruction(source_video=src, start=st, end=ed))
    instructions.soundtrack_path = soundtrack_path
    return instructions


def save_film_instructions_to_json(
    instructions: FilmInstructions, json_file="essay_film_instructions.json"
):
    data = {
        "clips": [
            {
                "source_video": c.source_video,
                "start": c.start,
                "end": c.end,
                "volume": c.volume,
            }
            for c in instructions.clips
        ],
        "soundtrack_path": instructions.soundtrack_path,
        "narration_path": instructions.narration_path,
        "output_filename": instructions.output_filename,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved film instructions to {json_file}")


# -----------------------------
#  7. RENDER FINAL FILM FROM INSTRUCTIONS
# -----------------------------
def load_film_instructions_from_json(json_file="essay_film_instructions.json") -> FilmInstructions:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    fi = FilmInstructions(
        clips=[],
        soundtrack_path=data.get("soundtrack_path", ""),
        narration_path=data.get("narration_path", ""),
        output_filename=data.get("output_filename", "essay_film.mp4"),
    )
    for c in data.get("clips", []):
        fi.clips.append(
            ClipInstruction(
                source_video=c["source_video"],
                start=c["start"],
                end=c["end"],
                volume=c.get("volume", 1.0),
            )
        )
    return fi


def render_film_from_instructions(
    instructions: FilmInstructions,
    final_file=None
):
    subclips = []
    for clip_info in instructions.clips:
        try:
            vc = VideoFileClip(clip_info.source_video).subclip(clip_info.start, clip_info.end)
            vc = vc.volumex(clip_info.volume)
            # Apply a random static zoom effect
            A = random.uniform(0.15, 0.80)  # Zoom variation between 5% and 20%
            zoom_factor = 1 + A
            w, h = vc.w, vc.h
            x1_val = int((w - (w / zoom_factor)) / 2)
            x2_val = int((w + (w / zoom_factor)) / 2)
            y1_val = int((h - (h / zoom_factor)) / 2)
            y2_val = int((h + (h / zoom_factor)) / 2)
            from moviepy.video.fx.all import crop
            vc = crop(vc, x1=x1_val, x2=x2_val, y1=y1_val, y2=y2_val)
            vc = vc.resize((w, h))

            from moviepy.video.fx.all import colorx
            # ... after cropping and resizing
            #vc = colorx(vc, 1.2)  # Increase color saturation by 20%
            #from moviepy.video.fx.all import lum_contrast

            # Adjust luminance, contrast, and optionally saturation.
            #vc = lum_contrast(vc, lum=0, contrast=50, contrast_thr=128)
            #vc = vc.fl_image(glitch_effect)
            
            # Add subtitles if available
            sub_text = get_subtitle_text_for_clip(clip_info.source_video, clip_info.start, clip_info.end)
            if sub_text:
                print(f"Adding subtitles to clip from {clip_info.source_video}")
                #vc = add_subtitles_to_clip(vc, sub_text)
            
            subclips.append(vc)
        except Exception as e:
            print(f"Error loading subclip from {clip_info.source_video}: {e}")

    if not subclips:
        print("No subclips to render.")
        return

    if subclips:
        print("Applying fade in to the first clip and fade out to the last clip.")
        from moviepy.video.fx.all import fadein, fadeout
        subclips[0] = fadein(subclips[0], 1)  # 1-second fade in for the first clip
        subclips[-1] = fadeout(subclips[-1], 1)  # 1-second fade out for the last clip

    final = concatenate_videoclips(subclips, method="compose")

    # Preserve original clip audio if available
    original_audio = final.audio

    # Add additional audio tracks: soundtrack and narration
    audio_clips = []
    if original_audio:
        print("Preserving original clip audio at reduced volume.")
        reduced_audio = original_audio.volumex(0.0)  # Lower the volume to 50%
        #reduced_audio = original_audio
        audio_clips.append(reduced_audio)
    if instructions.soundtrack_path and os.path.exists(instructions.soundtrack_path):
        print(f"Adding soundtrack: {instructions.soundtrack_path}")
        music = AudioFileClip(instructions.soundtrack_path)
        if music.duration > final.duration:
            music = music.subclip(0, final.duration)
        audio_clips.append(music)
    if instructions.narration_path and os.path.exists(instructions.narration_path):
        print(f"Adding narration audio: {instructions.narration_path}")
        narration = AudioFileClip(instructions.narration_path)
        if narration.duration > final.duration:
            narration = narration.subclip(0, final.duration)
        audio_clips.append(narration)
    if audio_clips:
        final_audio = CompositeAudioClip(audio_clips)
        final = final.set_audio(final_audio)

    output_name = final_file or instructions.output_filename
    final.write_videofile(output_name, codec="libx264", audio_codec="aac")

    # cleanup
    for c in subclips:
        c.close()
    final.close()
    print(f"Final film saved as {output_name}")


# -----------------------------
#  MAIN DEMO
# -----------------------------
def generate_narration_script(clips_info, agents=AGENTS):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    conversation_so_far = ""

    # Shuffle the subclips so we pick them in random order
    random.shuffle(clips_info)

    # For each subclip, randomly pick an agent
    for (vid_file, start, end) in clips_info:
        agent = random.choice(agents)

        # Build transcript for this subclip
        transcripts_text = ""
        sub_text = get_subtitle_text_for_clip(vid_file, start, end)
        if sub_text:
            transcripts_text += f"{sub_text}\n"

        # Construct the conversation prompt with the conversation so far
        user_prompt = f"""
{agent['style_instructions']}

The conversation so far:
{conversation_so_far}

Now, {agent['name']}, please respond to the conversation, incorporating this subtitle fragment:
{transcripts_text}
"""

        messages = [
            {"role": "system", "content": "You are a creative script writer."},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200
            )
            agent_text = response.choices[0].message.content.strip()
            # Update the conversation history with this agent's response
            conversation_so_far += f"\n{agent['name'].upper()}:\n{agent_text}\n"
        except Exception as e:
            print("Error generating narration for", agent["name"], e)
            conversation_so_far += f"\n{agent['name'].upper()}:\n[Error or no content]\n"

    print("Generated multi-agent conversation script:", conversation_so_far)

    # Keep the refinement step
    improvement_feedback = """1) Clarify each character's perspective.
2) Reduce repetition.
3) Add breathing room with some shorter sentences.
4) Include a few concrete touches.
5) Respond or reference previous voices lightly.
6) Build a smooth arc across the entire text.
7) Remove asterisks, etc (will be read by TTS).
8) Remix the text so that no one voice occupies the space for too long."""
    print("Refining script based on feedback...")
    refined_script = improve_script(conversation_so_far, improvement_feedback)
    print("Refined multi-agent conversation script:", refined_script)
    conversation_so_far = refined_script
    return conversation_so_far

def generate_narration_audio(script, lang="en", out_filename="narration.mp3"):
    """
    Convert the narration script to an audio file using gTTS.
    Requires gTTS library: pip install gtts
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=script, lang=lang)
        tts.save(out_filename)
        print(f"Narration audio saved as {out_filename}")
        return out_filename
    except Exception as e:
        print("Error generating narration audio:", e)
        return ""

def get_subtitle_text_for_clip(video_file, clip_start, clip_end):
    """
    Given a video file and a clip time range, attempt to find the associated subtitle file,
    parse it, and return a string of subtitle lines that fall entirely within the clip range.
    """
    import os
    base = os.path.basename(video_file)
    # Assuming the file naming follows the pattern: "footage/<title>_<video_id>.mp4"
    parts = base.rsplit("_", 1)
    if len(parts) < 2:
        return ""
    video_id_with_ext = parts[-1]
    video_id = video_id_with_ext.split(".")[0]
    sub_file = find_file_with_id(video_id, ".vtt")
    if not sub_file:
        return ""
    lines = parse_subtitle_file(sub_file)
    # Filter subtitle lines that fall within the clip time range
    clip_lines = [line.text for line in lines if line.start >= clip_start and line.end <= clip_end]
    return "\n".join(clip_lines)

def compute_narrative_score_for_clip(video_file, start, end):
    """
    Compute a narrative score for a clip based on the number of words in its subtitles.
    """
    text = get_subtitle_text_for_clip(video_file, start, end)
    return len(text.split()) if text else 0


def add_subtitles_to_clip(clip, subtitle_text):
    """
    Given a video clip and subtitle text, create a TextClip and overlay it on the video.
    Adjust fontsize and positioning as needed.
    """
    from moviepy.editor import TextClip, CompositeVideoClip
    # Create a TextClip that will act as the subtitle overlay.
    #txt_clip = TextClip(subtitle_text, fontsize=24, color='white', method='caption', size=(clip.w - 20, None))
    # Using pillow method with a semi-transparent background to improve visibility.
    txt_clip = TextClip(
        subtitle_text, 
        fontsize=3, 
        color='white', 
        method='caption',
        bg_color='black',  # Add background to improve visibility
    )
    txt_clip = txt_clip.set_duration(clip.duration).set_position(('center', 'bottom'))
    return clip
    #return CompositeVideoClip([clip, txt_clip])

def glitch_effect(frame):
    import numpy as np
    import random
    # 10% chance to apply a glitch effect on this frame
    if random.random() < 0.1:
        offset = random.randint(-15, 15)
        # Shift the frame horizontally
        frame = np.roll(frame, offset, axis=1)
        # Optionally, add some random noise
        noise = np.random.randint(0, 30, frame.shape, dtype='uint8')
        frame = np.clip(frame + noise, 0, 255)
    return frame

def improve_script(original_text, improvement_tips):
    """
    This function takes the entire text and a set of improvement tips,
    then refines the text based on the feedback.
    We'll call an AI API (like OpenAI) to do the rewriting.
    """
    if not OPENAI_API_KEY:
        print("No OPENAI_API_KEY provided, skipping text improvement.")
        return original_text
    
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
You are an expert text editor. The user has the following text that needs refinement:

Original text:
{original_text}

Improvement tips:
{improvement_tips}

Please rewrite the original text by applying the improvement tips, preserving its style and meaning but making it stronger.
Only return the rewritten result : it will be read by TTS.
"""

    messages = [
        {"role": "system", "content": "You are a helpful writing assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=3000,
            temperature=0.7
        )
        improved_text = response.choices[0].message.content.strip()
        return improved_text
    except Exception as e:
        print("Error refining text:", e)
        return original_text

def main():
    parser = argparse.ArgumentParser(description="Generate experimental essay film from YouTube videos.")
    parser.add_argument("--query", type=str, default="America", help="Search query for YouTube videos")
    parser.add_argument("--max_results", type=int, default=3, help="Max results to retrieve from YouTube")
    parser.add_argument("--randomize", action="store_true", help="Randomize clip order")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to store final output")
    parser.add_argument("--output_prefix", type=str, default="experimental_essay_film", help="Prefix for output file")
    parser.add_argument("--min_clips", type=int, default=5, help="Minimum number of subclips per video")
    parser.add_argument("--max_clips", type=int, default=15, help="Maximum number of subclips per video")
    parser.add_argument("--min_duration", type=float, default=0.1, help="Minimum subclip duration in seconds")
    parser.add_argument("--max_duration", type=float, default=4.0, help="Maximum subclip duration in seconds")
    parser.add_argument("--narrate", action="store_true", help="Generate narration for the film")
    parser.add_argument("--narrative_order", action="store_true", help="Order clips narratively based on subtitle content")
    parser.add_argument("--trending", action="store_true", help="Use trending videos instead of query search")
    parser.add_argument("--from-narration-file", type=str, help="Receive narration text instead of generating one")
    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Generate final output filename with current date-time appended
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_filename = os.path.join(args.output_folder, f"{args.output_prefix}_{now_str}.mp4")

    # 1. Retrieve YouTube videos
    if args.trending:
        print("Retrieving trending videos...")
        all_found_videos = get_most_popular_videos(max_results=args.max_results, region_code="US")
    else:
        # Use GPT to expand the user query into multiple queries
        print(f"Expanding the query '{args.query}' into multiple related queries with GPT...")
        expanded_queries = expand_keyword_with_gpt(args.query, how_many=3)
        
        print("GPT gave these queries:")
        for eq in expanded_queries:
            print(" -", eq)

        all_found_videos = []
        # For each expanded query, search YouTube
        for q in expanded_queries:
            print(f"Searching YouTube for '{q}'...")
            found_videos = search_youtube_videos(q, max_results=args.max_results)
            all_found_videos.extend(found_videos)

    if not all_found_videos:
        print("No videos found for any query, exiting.")
        return

    # 2. Download each video + subs
    downloaded = []
    for vid, title in all_found_videos:
        print(f"Downloading {vid} : {title}")
        mp4_file = download_youtube_video(vid)
        if mp4_file:
            downloaded.append(mp4_file)
        else:
            print(f"Failed to download video ID {vid}.")

    if not downloaded:
        print("No videos successfully downloaded, exiting.")
        return

    # 3. Generate random subclips for each video
    subclip_info = []
    for vid_file in downloaded:
        try:
            clip = VideoFileClip(vid_file)
            duration = clip.duration
            clip.close()
        except Exception as e:
            print(f"Error loading video {vid_file}: {e}")
            continue
        num_clips = random.randint(args.min_clips, args.max_clips)
        for _ in range(num_clips):
            clip_duration = random.uniform(args.min_duration, args.max_duration)
            if duration <= clip_duration:
                start_sec = 0
                end_sec = duration
            else:
                start_sec = random.uniform(0, duration - clip_duration)
                end_sec = start_sec + clip_duration
            subclip_info.append((vid_file, start_sec, end_sec))

    if not subclip_info:
        print("No interesting subclips found, let's just skip montage.")
        return

    # Optionally order clips narratively or randomize
    if args.narrative_order:
        print("Ordering clips narratively based on subtitle content...")
        subclip_info.sort(key=lambda clip: compute_narrative_score_for_clip(clip[0], clip[1], clip[2]))
    elif args.randomize:
        print("Randomizing clip order...")
        random.shuffle(subclip_info)

    # 4. Generate an AI soundtrack (optional)
    soundtrack_file = None
    if MUBERT_API_KEY:
        soundtrack_file = generate_mubert_track(duration=60, style="abstract_electronic")
    else:
        print("No MUBERT_API_KEY, skipping soundtrack generation.")

    # 5. Build film instructions (montage)
    instructions = build_film_instructions(
        subclip_info, 
        soundtrack_path=soundtrack_file or "", 
        output_filename=final_output_filename
    )
    if args.narrate:
        if args.from_narration_file:
            print(f"Loading narration text from {args.from_narration_file}...")
            try:
                with open(args.from_narration_file, "r", encoding="utf-8") as f:
                    narration_script = f.read()
            except Exception as e:
                print("Failed to load narration file:", e)
                narration_script = ""
        else:
            print("Generating narration script...")
            narration_script = generate_narration_script(subclip_info)
            if narration_script:
                narration_text_filename = os.path.join(args.output_folder, f"narration_{args.query.replace(' ', '_')}_{now_str}.txt")
                with open(narration_text_filename, "w", encoding="utf-8") as f:
                    f.write(narration_script)
                print(f"Narration script saved as {narration_text_filename}")
        if narration_script:
            narration_audio_filename = os.path.join(args.output_folder, f"narration_{args.query.replace(' ', '_')}_{now_str}.mp3")
            narration_audio_path = generate_narration_audio(narration_script, out_filename=narration_audio_filename)
            if narration_audio_path:
                print(f"Narration audio generated at {narration_audio_path}")
                instructions.narration_path = narration_audio_path
            else:
                print("Failed to generate narration audio.")
        else:
            print("Narration script is empty. Skipping narration generation.")
 
    # 6. Save instructions to JSON for manual editing
    #save_film_instructions_to_json(instructions, "essay_film_instructions.json")
    film_instructions_filename = os.path.join(args.output_folder, f"essay_film_instructions_{now_str}.json")
    save_film_instructions_to_json(instructions, film_instructions_filename)

    # 7. Render final film from the instructions
    final_instructions = load_film_instructions_from_json(film_instructions_filename)
    render_film_from_instructions(final_instructions, final_file=final_output_filename)


if __name__ == "__main__":
    main()