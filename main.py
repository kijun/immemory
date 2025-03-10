import os
import re
import time
import glob
import json
import random
import requests
import subprocess
from dataclasses import dataclass, field
from typing import List
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
from googleapiclient.discovery import build
from dateutil import tz
from dotenv import load_dotenv
load_dotenv()
import argparse
from datetime import datetime

# -----------------------------
#  CONFIG / SECRETS
# -----------------------------
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)  # None means no LLM scoring
MUBERT_API_KEY = os.getenv("MUBERT_API_KEY", None)  # None means skip AI soundtrack

# For clarity, we assume these are set as environment vars or replaced inline.
# Example in terminal:
# export YOUTUBE_API_KEY="sk-xxx..."


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
            subclips.append(vc)
        except Exception as e:
            print(f"Error loading subclip from {clip_info.source_video}: {e}")

    if not subclips:
        print("No subclips to render.")
        return

    final = concatenate_videoclips(subclips, method="compose")

    # Soundtrack
    if instructions.soundtrack_path and os.path.exists(instructions.soundtrack_path):
        print(f"Adding soundtrack: {instructions.soundtrack_path}")
        music = AudioFileClip(instructions.soundtrack_path)
        # If music is shorter, let it just end; if longer, we'll subclip to final duration
        if music.duration > final.duration:
            music = music.subclip(0, final.duration)
        # If you want to mix multiple audio tracks or do voiceover, you'd use CompositeAudioClip
        final_audio = CompositeAudioClip([music])
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
    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Generate final output filename with current date-time appended
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_filename = os.path.join(args.output_folder, f"{args.output_prefix}_{now_str}.mp4")

    # 1. Search YouTube
    query = args.query
    max_results = args.max_results
    print(f"Searching YouTube for '{query}'...")
    found_videos = search_youtube_videos(query, max_results=max_results)

    if not found_videos:
        print("No videos found, exiting.")
        return

    # 2. Download each video + subs
    downloaded = []
    for vid, title in found_videos:
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

    # Optionally randomize clip order if flag provided
    if args.randomize:
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

    # 6. Save instructions to JSON for manual editing
    save_film_instructions_to_json(instructions, "essay_film_instructions.json")

    # 7. Render final film from the instructions
    final_instructions = load_film_instructions_from_json("essay_film_instructions.json")
    render_film_from_instructions(final_instructions, final_file=final_output_filename)


if __name__ == "__main__":
    main()