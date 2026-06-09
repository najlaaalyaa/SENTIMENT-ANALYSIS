"""
scraper.py — YouTube comment collector using YouTube Data API v3
Run standalone: python scraper.py --url "https://youtube.com/watch?v=VIDEO_ID" --max 200
"""

import os
import re
import csv
import json
import argparse
import requests
from datetime import datetime

# ── Get API key from env or .streamlit/secrets ────────────────
def get_api_key():
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("YOUTUBE_API_KEY")
        except Exception:
            pass
    if not key:
        raise ValueError(
            "YouTube API key not found.\n"
            "Set YOUTUBE_API_KEY in your environment or Streamlit secrets."
        )
    return key


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def get_video_info(video_id: str, api_key: str) -> dict:
    """Fetch video title and channel name."""
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": video_id,
        "key": api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        return {"title": "Unknown", "channel": "Unknown"}
    snippet = items[0]["snippet"]
    return {
        "title":   snippet.get("title", "Unknown"),
        "channel": snippet.get("channelTitle", "Unknown"),
    }


def fetch_comments(video_id: str, api_key: str, max_comments: int = 200) -> list:
    """
    Fetch YouTube comments using the Comments.list API.
    Returns list of dicts with keys: text, author, likes, published_at.
    """
    url      = "https://www.googleapis.com/youtube/v3/commentThreads"
    comments = []
    page_token = None

    while len(comments) < max_comments:
        params = {
            "part":            "snippet",
            "videoId":         video_id,
            "maxResults":      min(100, max_comments - len(comments)),
            "textFormat":      "plainText",
            "key":             api_key,
            "order":           "relevance",
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment":      top.get("textDisplay", ""),
                "author":       top.get("authorDisplayName", ""),
                "likes":        top.get("likeCount", 0),
                "published_at": top.get("publishedAt", ""),
            })

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return comments[:max_comments]


def scrape(video_url: str, max_comments: int = 200, save_csv: bool = True) -> list:
    """
    Main entry point. Fetches comments for a YouTube video URL.
    Returns list of comment dicts. Optionally saves to CSV.
    """
    api_key  = get_api_key()
    video_id = extract_video_id(video_url)
    info     = get_video_info(video_id, api_key)

    print(f"📹 Video  : {info['title']}")
    print(f"📺 Channel: {info['channel']}")
    print(f"🔄 Fetching up to {max_comments} comments…")

    comments = fetch_comments(video_id, api_key, max_comments)
    print(f"✅ Fetched {len(comments)} comments.")

    if save_csv:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comments_{video_id}_{ts}.csv"
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["comment", "author", "likes", "published_at"])
            writer.writeheader()
            writer.writerows(comments)
        print(f"💾 Saved to {filename}")

    return comments


# ── CLI usage ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Comment Scraper")
    parser.add_argument("--url",  required=True, help="YouTube video URL")
    parser.add_argument("--max",  type=int, default=200, help="Max comments to fetch")
    parser.add_argument("--no-save", action="store_true", help="Don't save CSV")
    args = parser.parse_args()

    scrape(args.url, max_comments=args.max, save_csv=not args.no_save)
