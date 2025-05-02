import os
import requests
import time
from serpapi import GoogleSearch
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('API_KEY')

SAVE_DIR = os.path.join('downloaded_images', 'gg_images')
os.makedirs(SAVE_DIR, exist_ok=True)


def get_query_and_limit_from_cli():
    """
    Parse search query and optional image download limit from command line.
    Returns:
        tuple: (query string, limit integer)
    """
    parser = argparse.ArgumentParser(description="Search and download images from Google Images via SerpAPI.")
    parser.add_argument('query', type=str, help='Search query for images')
    parser.add_argument('--limit', type=int, default=50, help='Maximum number of images to download (default: 50)')
    args = parser.parse_args()
    return args.query, args.limit


def search_images(query):
    """
    Search images using SerpAPI Google Images engine.
    Parameters:
        query (str): The search term.
    Returns:
        list: List of image URLs from the search results.
    """
    params = {
        "api_key": API_KEY,
        "engine": "google_images",
        "google_domain": "google.com",
        "q": query,
        "hl": "en",
        "gl": "us",
        "num": 50  
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return [item['original'] for item in results.get("images_results", [])]


def download_image(url, save_path):
    """
    Downloads an image from a URL and saves it to the given path.
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            file.write(response.content)
        return True
    except requests.RequestException as e:
        print(f"❌ Failed to download {url}: {e}")
        return False


def main():
    """
    Main pipeline: parse args, search images, and download them.
    """
    query, limit = get_query_and_limit_from_cli()
    image_links = search_images(query)

    if not image_links:
        print("❌ No image links found.")
        return

    print(f"📸 Found {len(image_links)} image links. Downloading up to {limit}...")

    downloaded = 0
    for i, url in enumerate(tqdm(image_links, desc="Downloading images")):
        if downloaded >= limit:
            break
        filename = f"image_{downloaded + 1}.png"
        save_path = os.path.join(SAVE_DIR, filename)
        if download_image(url, save_path):
            downloaded += 1
        time.sleep(0.2)  

    print(f"\n✅ Downloaded {downloaded} image(s).")


if __name__ == "__main__":
    main()
