#!/bin/bash

# === SETTINGS ===
IMG_KEYWORDS="$1"
VID_KEYWORDS="$2"
MAX_FILESIZE_MB=500
RESULT_FILE="video_urls.txt"
#BASE_DIR="crawl_test"
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

echo "📷 Step 1: Downloading Google Images for: '$IMG_KEYWORDS'"
python3 download_images.py "$IMG_KEYWORDS"

echo "🎬 Step 2: Searching YouTube for: '$VID_KEYWORDS'..."
yt-dlp --get-id "ytsearch5:$VID_KEYWORDS" | sed 's/^/https:\/\/www.youtube.com\/watch?v=/' > "$RESULT_FILE"
#yt-dlp --match-filter "is_live = false" --get-id "ytsearch5:$VID_KEYWORDS" | sed 's/^/https:\/\/www.youtube.com\/watch?v=/' > "$RESULT_FILE"

echo "📄 Found video URLs:"
cat "$RESULT_FILE"

mapfile -t URLS < "$RESULT_FILE"

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

echo "🖼️ Step 3: Extracting key frames from downloaded videos..."
for VIDEO_FILE in "$DOWNLOAD_DIR"/*.mp4; do
    echo "🔍 Processing: $VIDEO_FILE"
    python3 extract_video_frames.py "$VIDEO_FILE" "$FRAME_OUTPUT_DIR"
done

echo "🎉 All tasks completed!"


##How to 
# chmod +x process.sh
# ./process.sh "street webcam car" "traffic camera video"  - image keywords, video keywords
# Change ytsearch5/10 for num of videos 
# Beware of live videos, usually they make the process longer (more than 10 mins for both images and videos); if you accidentally download a live video -> wait 4 mins and ctrl c
# Another option: yt-dlp --match-filter "is_live = false" --get-id "ytsearch5:$VID_KEYWORDS" | sed 's/^/https:\/\/www.youtube.com\/watch?v=/' > "$RESULT_FILE"