#!/bin/bash
# Quick demo setup script

echo "==================================="
echo "Photo Duplicate Finder - Demo Setup"
echo "==================================="
echo ""

# Create demo directory
DEMO_DIR="$HOME/demo_photos"
echo "Creating demo directory: $DEMO_DIR"
mkdir -p "$DEMO_DIR"

# Download a few sample images (if curl/wget available)
echo ""
echo "Downloading sample images..."

for i in {1..20}; do
    echo "Downloading image $i/20..."
    curl -s -L "https://source.unsplash.com/800x600/?nature,city&sig=$i" \
         -o "$DEMO_DIR/photo_$(printf '%03d' $i).jpg" 2>/dev/null || \
    wget -q "https://source.unsplash.com/800x600/?nature,city&sig=$i" \
         -O "$DEMO_DIR/photo_$(printf '%03d' $i).jpg" 2>/dev/null
done

# Create some exact duplicates
echo ""
echo "Creating exact duplicates..."
cp "$DEMO_DIR/photo_001.jpg" "$DEMO_DIR/photo_001_copy.jpg"
cp "$DEMO_DIR/photo_005.jpg" "$DEMO_DIR/photo_005_duplicate.jpg"
cp "$DEMO_DIR/photo_010.jpg" "$DEMO_DIR/photo_010_copy.jpg"

echo ""
echo "==================================="
echo "✓ Demo dataset created!"
echo "==================================="
echo ""
echo "Location: $DEMO_DIR"
echo "Total files: $(ls -1 "$DEMO_DIR" | wc -l | tr -d ' ')"
echo ""
echo "Next steps:"
echo "1. Open http://127.0.0.1:8080"
echo "2. Select Source: Directory"
echo "3. Enter path: $DEMO_DIR"
echo "4. Click 'Start Scan'"
echo "5. Generate embeddings"
echo "6. Review duplicates!"
echo ""
echo "In DEMO MODE:"
echo "- Duplicates will be copied to: $DEMO_DIR/duplicates/"
echo "- No Photos.app permissions needed"
echo "- Perfect for presentations!"
echo ""
