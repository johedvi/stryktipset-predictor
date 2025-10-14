#!/bin/bash

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GITHUB_USERNAME="YOUR_USERNAME"  # Change this!
REPO_NAME="stryktipset-predictor"
BRANCH="main"

# Functions
print_header() {
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Main script starts here
print_header "Stryktipset Site Updater"
echo ""

# Get current week and year
WEEK=$(date +%V)
YEAR=$(date +%Y)
DATE=$(date +%Y-%m-%d)

print_info "Date: $DATE"
print_info "Week: $WEEK, $YEAR"
echo ""

# Step 0: Check if we're in the right directory
if [ ! -f "config.py" ] || [ ! -f "app.py" ]; then
    print_error "Not in project root directory!"
    print_info "Please run from the stryktipset-predictor directory"
    exit 1
fi

# Step 1: Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_warning "You have uncommitted changes"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Aborted by user"
        exit 0
    fi
fi

# Step 2: Pull latest changes
print_header "Updating from GitHub"
git pull origin $BRANCH

if [ $? -ne 0 ]; then
    print_error "Failed to pull from GitHub"
    print_info "Fix conflicts manually and try again"
    exit 1
fi

print_success "Up to date with remote"
echo ""

# Step 3: Generate new predictions
print_header "Generating Predictions"

# Use Python module syntax to avoid import issues
python -m src.prediction.predict

if [ $? -ne 0 ]; then
    print_error "Prediction generation failed!"
    print_info "Check error messages above"
    exit 1
fi

print_success "Predictions generated"
echo ""

# Step 4: Generate HTML viewer
print_header "Generating HTML Viewer"

# Check which viewer script exists and run with Python module syntax
if [ -f "generate_viewer.py" ]; then
    python generate_viewer.py
elif [ -f "scripts/generate_coupon_viewer.py" ]; then
    python scripts/generate_coupon_viewer.py
else
    print_error "Viewer generator not found!"
    print_info "Expected: generate_viewer.py or scripts/generate_coupon_viewer.py"
    exit 1
fi

if [ $? -ne 0 ]; then
    print_error "Viewer generation failed!"
    exit 1
fi

print_success "HTML viewer generated"
echo ""

# Step 5: Verify files exist
print_header "Verifying Output Files"

ERRORS=0

if [ -f "index.html" ]; then
    SIZE=$(du -h index.html | cut -f1)
    print_success "index.html ($SIZE)"
else
    print_error "index.html not found!"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "coupons" ]; then
    COUNT=$(ls -1 coupons/*.txt 2>/dev/null | wc -l)
    print_success "coupons/ directory ($COUNT files)"
else
    print_error "coupons/ directory not found!"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    print_error "Missing required files!"
    exit 1
fi

echo ""

# Step 6: Show what will be committed
print_header "Changes to Commit"
git status --short index.html stryktipset_viewer.html coupons/
echo ""

# Step 7: Commit changes
print_header "Committing Changes"

# Force add files (they're in .gitignore)
git add -f stryktipset_viewer.html index.html coupons/*.txt

# Check if there are changes to commit
if git diff --staged --quiet; then
    print_info "No changes to commit"
    exit 0
fi

COMMIT_MSG="Update predictions - Week $WEEK, $YEAR

- Generated on: $DATE
- Week: $WEEK
- Year: $YEAR
- Strategies: 3
"

git commit -m "$COMMIT_MSG"

if [ $? -ne 0 ]; then
    print_error "Commit failed!"
    exit 1
fi

print_success "Changes committed"
echo ""

# Step 8: Push to GitHub
print_header "Pushing to GitHub"

git push origin $BRANCH

if [ $? -ne 0 ]; then
    print_error "Push failed!"
    print_info "Try: git push origin $BRANCH --force-with-lease"
    exit 1
fi

print_success "Pushed to GitHub"
echo ""

# Step 9: Success summary
print_header "🎉 Deployment Successful!"
echo ""
echo "📊 Week: $WEEK, $YEAR"
echo "📅 Date: $DATE"
echo "⏳ GitHub Pages will update in ~1-2 minutes"
echo ""
echo "🌐 View your site at:"
echo "   https://$GITHUB_USERNAME.github.io/$REPO_NAME/"
echo "   (redirects to stryktipset_viewer.html)"
echo ""
echo "📁 Local files updated:"
echo "   - index.html"
echo "   - stryktipset_viewer.html"
echo "   - coupons/*.txt"
echo ""
print_info "To view locally: open index.html"
echo ""