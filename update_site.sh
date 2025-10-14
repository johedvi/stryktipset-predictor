#!/bin/bash
echo "🔄 Updating Stryktipset site..."

# Get current week
WEEK=$(date +%V)
YEAR=$(date +%Y)

# Generate new predictions and viewer
python predict_stryktipset.py
python generate_viewer.py

# Add and commit
git add stryktipset_viewer.html coupons/
git commit -m "Update coupons - Week $WEEK, $YEAR"

# Push to GitHub
git push origin main

echo "✅ Pushed to GitHub!"
echo "⏳ Site will update in ~1 minute"
echo "🌐 https://YOUR_USERNAME.github.io/stryktipset-predictor/stryktipset_viewer.html"