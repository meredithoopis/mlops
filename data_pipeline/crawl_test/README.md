# 🖼️ Image & Video Crawler with Frame Extraction

A simple, script-based pipeline to crawl images using **SerpAPI**, download videos from **YouTube**, and extract key frames using **ffmpeg**.

---

## 📦 Features

- 🔍 **Image Crawling** from Google Images via SerpAPI
- 🎥 **Video Downloading** via `yt-dlp` (YouTube)
- 🖼️ **Key Frame Extraction** from videos using OpenCV
- 🛠️ Shell automation via `process.sh`

---

## ⚙️ How-to
### 1. Configure API key
Create an .env file in the folder: 
```ini 
API_KEY=your_serpapi_key_here
```
Paste your API Key from [SerpApi](https://serpapi.com/users/sign_in)

### 2. Running the script 
```bash 
chmod +x process.sh
./process.sh "street webcam car" "traffic camera video"
```



