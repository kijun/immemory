import os
import sys
from datetime import datetime, timedelta
from dateutil import tz
from dotenv import load_dotenv

from googleapiclient.discovery import build
from yt_dlp import YoutubeDL

load_dotenv()

# ------------------------------------------------
#  Configuration
# ------------------------------------------------
API_KEY = os.getenv("API_KEY")   # Replace with your actual key
SEARCH_QUERY = "America"
VIDEO_COUNT = 2

# ------------------------------------------------
# 1. Build YouTube client using the API key
# ------------------------------------------------
youtube = build("youtube", "v3", developerKey=API_KEY)

# ------------------------------------------------
# 2. Calculate publishedAfter = 24 hours ago (UTC)
# ------------------------------------------------
utc_now = datetime.utcnow()
utc_24h_ago = utc_now - timedelta(hours=24)
published_after = utc_24h_ago.replace(tzinfo=tz.UTC).isoformat()

# ------------------------------------------------
# 3. Call the YouTube Data API to find top videos
# ------------------------------------------------
try:
    search_response = youtube.search().list(
        q=SEARCH_QUERY,
        part="snippet",
        type="video",
        order="viewCount",           # Sort results by descending view count
        publishedAfter=published_after,
        maxResults=VIDEO_COUNT
    ).execute()
except Exception as e:
    print(f"Error fetching data from YouTube API: {e}")
    sys.exit(1)

items = search_response.get("items", [])
if not items:
    print("No videos found. Try adjusting the query/time range.")
    sys.exit(0)

# ------------------------------------------------
# 4. Build a list of video URLs
# ------------------------------------------------
video_urls = [
    f"https://www.youtube.com/watch?v={item['id']['videoId']}"
    for item in items
]

print(f"Found {len(video_urls)} videos. Starting downloads via yt-dlp...\n")

# ------------------------------------------------
# 5. Download each video using yt-dlp
# ------------------------------------------------
# ydl_opts = {
#     # For example, to specify format, output path, etc.:
#     # 'format': 'bestvideo+bestaudio/best',
#     # 'outtmpl': '%(title)s.%(ext)s',
#     # 'quiet': True,
#     # ...
# }

ydl_opts = {
    # General video download config
    'outtmpl': 'footage/%(title)s.%(ext)s',  # Save files in the 'footage' folder
    # 'format': 'bestvideo+bestaudio/best',  # Uncomment to choose best video+audio format

    # Subtitles/transcripts config:
    'writesubtitles': True,           # download subtitles if available
    'writeautomaticsub': True,        # also download auto-generated subs
    # 'allsubtitles': True,          # uncomment to get *every* language
    'subtitleslangs': ['en'],         # which language(s) to download
    'subtitlesformat': 'vtt/best',    # prefer .vtt if multiple formats
    # 'convertsubtitles': 'srt',     # uncomment if you want .srt instead of .vtt
}

with YoutubeDL(ydl_opts) as ydl:
    for url in video_urls:
        print(f"Downloading: {url}")
        try:
            ydl.download([url])
        except Exception as e:
            print(f"Failed to download {url}: {e}")
        print()

print("All downloads complete (where successful)!")
