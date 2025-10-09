"""
Test script to verify the setup is correct
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    packages = [
        ('requests', 'requests'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
    ]
    
    failed = []
    
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT FOUND")
            failed.append(package_name)
    
    if failed:
        print(f"\n⚠️  Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages imported successfully!")
    return True


def test_project_structure():
    """Test that directory structure exists"""
    print("\nTesting directory structure...")
    
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'data/cache',
        'models',
        'logs',
    ]
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ - NOT FOUND")
            path.mkdir(parents=True, exist_ok=True)
            print(f"    Created {directory}/")
    
    print("\n✓ Directory structure verified!")
    return True


def test_api_key():
    """Test that API key is configured"""
    print("\nTesting API key configuration...")
    
    from config import API_FOOTBALL_KEY
    
    if API_FOOTBALL_KEY == "your_api_key_here" or not API_FOOTBALL_KEY:
        print("  ✗ API key not configured")
        print("\n  Set your API key in one of these ways:")
        print("    1. Environment variable: export API_FOOTBALL_KEY='your_key'")
        print("    2. Edit config.py: API_FOOTBALL_KEY = 'your_key'")
        print("    3. Create .env file with: API_FOOTBALL_KEY=your_key")
        print("\n  Get your key from: https://www.api-football.com/")
        return False
    
    print(f"  ✓ API key configured (length: {len(API_FOOTBALL_KEY)})")
    return True


def test_modules():
    """Test that project modules can be imported"""
    print("\nTesting project modules...")
    
    modules = [
        'config',
        'utils',
        'data_fetcher',
        'data_explorer',
        'rule_based_predictor',
        'feature_engineering',
        'ml_predictor',
        'main',
    ]
    
    failed = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}.py")
        except Exception as e:
            print(f"  ✗ {module}.py - ERROR: {str(e)[:50]}")
            failed.append(module)
    
    if failed:
        print(f"\n⚠️  Failed to import: {', '.join(failed)}")
        return False
    
    print("\n✓ All project modules imported successfully!")
    return True


def test_api_connection():
    """Test API connection (optional, uses 1 API request)"""
    print("\nTesting API connection...")
    print("  (This will use 1 API request from your quota)")
    
    response = input("  Test API connection? (y/n): ")
    
    if response.lower() != 'y':
        print("  ⊘ Skipped API connection test")
        return True
    
    try:
        from src.data.data_fetcher import APIFootballFetcher
        
        fetcher = APIFootballFetcher()
        
        # Try to fetch fixtures for a small league
        fixtures = fetcher.get_fixtures(league_id=39, season=2023)  # Premier League
        
        if fixtures and len(fixtures) > 0:
            print(f"  ✓ API connection successful! (Found {len(fixtures)} fixtures)")
            return True
        else:
            print("  ✗ API returned no data")
            print("    Check your API key and subscription status")
            return False
    
    except Exception as e:
        print(f"  ✗ API connection failed: {str(e)[:100]}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Stryktipset Predictor - Setup Verification")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Directory Structure", test_project_structure()))
    results.append(("API Key", test_api_key()))
    results.append(("Project Modules", test_modules()))
    results.append(("API Connection", test_api_connection()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60 + "\n")
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    
    if all_passed:
        print("🎉 All tests passed! You're ready to go!")
        print("="*60 + "\n")
        print("Quick start guide:")
        print("  1. Fetch data: python data_fetcher.py")
        print("  2. Explore: python data_explorer.py")
        print("  3. Train: python feature_engineering.py && python ml_predictor.py")
        print("  4. Predict: python main.py")
        print("\nSee README.md for detailed instructions")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("="*60)
    
    print()


if __name__ == "__main__":
    main()