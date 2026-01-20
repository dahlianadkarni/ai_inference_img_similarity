#!/usr/bin/env python3
"""
Create a demo dataset with intentional duplicates for testing.

Downloads sample images and creates exact + perceptual duplicates.
"""
import requests
from pathlib import Path
import shutil
from PIL import Image
import random

DEMO_DIR = Path.home() / "demo_photos"
UNSPLASH_ACCESS_KEY = "YOUR_UNSPLASH_ACCESS_KEY"  # Get free key at unsplash.com/developers

def download_unsplash_images(count=50):
    """Download random images from Unsplash."""
    DEMO_DIR.mkdir(exist_ok=True)
    
    print(f"Downloading {count} images from Unsplash...")
    
    for i in range(count):
        try:
            # Use source.unsplash.com for random images (no API key needed)
            url = f"https://source.unsplash.com/800x600/?nature,city,food,people&sig={i}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                filepath = DEMO_DIR / f"photo_{i:04d}.jpg"
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {filepath.name}")
        except Exception as e:
            print(f"Error downloading image {i}: {e}")
    
    print(f"Downloaded images to {DEMO_DIR}")

def create_duplicates():
    """Create exact and perceptual duplicates."""
    print("\nCreating duplicates...")
    
    photos = list(DEMO_DIR.glob("photo_*.jpg"))
    
    # Create exact duplicates (10% of photos)
    exact_count = max(5, len(photos) // 10)
    for photo in random.sample(photos, min(exact_count, len(photos))):
        duplicate = DEMO_DIR / f"{photo.stem}_duplicate{photo.suffix}"
        shutil.copy2(photo, duplicate)
        print(f"Created exact duplicate: {duplicate.name}")
    
    # Create perceptual duplicates (resized versions)
    perceptual_count = max(5, len(photos) // 10)
    for photo in random.sample(photos, min(perceptual_count, len(photos))):
        try:
            img = Image.open(photo)
            # Resize slightly
            new_size = (int(img.width * 0.9), int(img.height * 0.9))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            
            duplicate = DEMO_DIR / f"{photo.stem}_resized{photo.suffix}"
            resized.save(duplicate, quality=85)
            print(f"Created perceptual duplicate: {duplicate.name}")
        except Exception as e:
            print(f"Error creating perceptual duplicate for {photo.name}: {e}")
    
    print(f"\nDemo dataset ready at: {DEMO_DIR}")
    print(f"Total images: {len(list(DEMO_DIR.glob('*.jpg')))}")

if __name__ == "__main__":
    print("Creating demo dataset...")
    print("=" * 60)
    
    # Option 1: Download from Unsplash
    download_unsplash_images(50)
    
    # Option 2: Or just use existing images in a folder
    # DEMO_DIR = Path("/path/to/your/existing/photos")
    
    create_duplicates()
    
    print("\n" + "=" * 60)
    print("Demo dataset created!")
    print(f"\nTo scan this dataset, use:")
    print(f"  Source: Directory")
    print(f"  Path: {DEMO_DIR}")
