from icrawler.builtin import GoogleImageCrawler
from pathlib import Path

# Base folder to save images
base_dir = Path("data/food-101/train")
base_dir.mkdir(parents=True, exist_ok=True)

# Food classes you want to download
classes = ["rice", "cheese", "hotdog", "eggroll"]

# Number of images per class
max_images = 50

for item in classes:
    folder = base_dir / item
    folder.mkdir(exist_ok=True)

    print(f"Downloading images for: {item}")
    crawler = GoogleImageCrawler(storage={"root_dir": str(folder)})
    crawler.crawl(keyword=item, max_num=max_images)
