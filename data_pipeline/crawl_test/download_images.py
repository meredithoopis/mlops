import os
import requests
import json
import time
from serpapi import GoogleSearch
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')
SAVE_DIR = os.path.join('downloaded_images', 'gg_images')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def get_query_from_cli():
    parser = argparse.ArgumentParser(description="Search and download images from Google Images via SerpAPI")
    parser.add_argument('query', type=str, help='Search query for images')
    args = parser.parse_args()
    return args.query

def search_images(query): 
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
    with open('res.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Results saved to res.json")
    return results

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"✅ Image saved to {save_path}")
    except requests.RequestException as e:
        print(f"❌ Failed to download {url}: {e}")
        return url  


def main():
    query = get_query_from_cli()  
    results = search_images(query)  

    images_link = []
    for result in results.get("images_results", []):
        images_link.append(result['original'])
    
    failed_links = []  
    for i, url in enumerate(tqdm(images_link, desc="Downloading images")):
        filename = f"image_{i+1}.png" 
        save_path = os.path.join(SAVE_DIR, filename)
        failed_url = download_image(url, save_path)
        if failed_url:
            failed_links.append(failed_url)
        time.sleep(0.2)
    if failed_links:
        with open("failed_links.txt", "w") as f:
            for link in failed_links:
                f.write(link + "\n")
        print(f"\n⚠️ Some images failed to download. See failed_links.txt for details.")
    else:
        print("\n🎉 All images downloaded successfully.")

if __name__ == "__main__":
    main()

#how to: python download_images.py "street webcam car"
#Problem: How to limit the number of images downloaded? Currently, we have 100 images downloaded.