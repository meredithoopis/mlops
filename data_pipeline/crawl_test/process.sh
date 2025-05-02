#!/bin/bash

# === SETTINGS ===
IMG_KEYWORDS="$1"
VID_KEYWORDS="$2"
MAX_FILESIZE_MB=500
DOWNLOAD_DIR="videos"
IMAGE_OUTPUT_DIR="downloaded_images/gg_images"
FRAME_OUTPUT_DIR="downloaded_images/from_youtube"
MAX_FILESIZE="${MAX_FILESIZE_MB}M"  

# === CHECK ARGS ===
if [ -z "$IMG_KEYWORDS" ] || [ -z "$VID_KEYWORDS" ]; then
    echo "❌ Usage: ./process.sh \"<image_keywords>\" \"<video_keywords>\""
    exit 1
fi

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$IMAGE_OUTPUT_DIR"
mkdir -p "$FRAME_OUTPUT_DIR"

# === STEP 1: Download Images ===
echo "📷 Step 1: Downloading Google Images for: '$IMG_KEYWORDS'"
python3 download_images.py "$IMG_KEYWORDS"

# === STEP 2: Search YouTube and Download Videos ===
echo "🎬 Step 2: Searching YouTube for: '$VID_KEYWORDS'..."
mapfile -t URLS < <(yt-dlp --get-id "ytsearch5:$VID_KEYWORDS" | sed 's/^/https:\/\/www.youtube.com\/watch?v=/')  #can be ytsearch5/ytsearch10

echo "📄 Found video URLs:"
for URL in "${URLS[@]}"; do
    echo "$URL"
done

# === Download each video with constraints ===
COUNTER=1
for VIDEO_URL in "${URLS[@]}"; do
    OUTPUT_NAME="source${COUNTER}.mp4"
    OUTPUT_PATH="$DOWNLOAD_DIR/$OUTPUT_NAME"

    echo "⏳ Downloading #$COUNTER: $VIDEO_URL"

    yt-dlp \
        -f "bestvideo[ext=mp4]" \
        --max-filesize "$MAX_FILESIZE" \
        -o "$OUTPUT_PATH" \
        "$VIDEO_URL" \
    || {
        echo "⚠️  $VIDEO_URL is too large or missing .mp4 format — downloading first 10 minutes instead"
        yt-dlp \
            -f "bestvideo[ext=mp4]" \
            --download-sections "*00:00:00-00:10:00" \
            -o "$OUTPUT_PATH" \
            "$VIDEO_URL" \
        || echo "❌ Failed to download $VIDEO_URL"
    }

    ((COUNTER++))
done

# === STEP 3: Extract Key Frames from Videos ===
echo "🖼️ Step 3: Extracting key frames from downloaded videos..."
for VIDEO_FILE in "$DOWNLOAD_DIR"/*.mp4; do
    echo "🔍 Processing: $VIDEO_FILE"
    python3 extract_video_frames.py "$VIDEO_FILE" "$FRAME_OUTPUT_DIR"
done

echo "🎉 All tasks completed!"

# Beware of live videos, usually they make the process longer (more than 10 mins for both images and videos); if you accidentally download a live video -> wait 4 mins and ctrl c
