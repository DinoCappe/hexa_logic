import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Target webpage and download directory
BASE_URL = "https://www.boardspace.net/hive/hivegames/archive-2024/"
DOWNLOAD_DIR = "hivegames_2024_zips"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def fetch_zip_links(url):
    """Fetch all .zip file URLs from the page."""
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    zip_links = []

    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.zip'):
            zip_links.append(urljoin(url, href))

    return zip_links

def download_file(url, directory):
    """Download a file and save it to the specified directory."""
    filename = os.path.join(directory, os.path.basename(url))
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    zip_urls = fetch_zip_links(BASE_URL)
    print(f"Found {len(zip_urls)} zip files.")

    for url in zip_urls:
        download_file(url, DOWNLOAD_DIR)

if __name__ == "__main__":
    main()
