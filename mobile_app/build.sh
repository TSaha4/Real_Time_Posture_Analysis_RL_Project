#!/bin/bash

# UPRYT Mobile Build Script
# This script helps build the Android APK

set -e

echo "============================================"
echo "UPRYT Mobile - Android Build Script"
echo "============================================"
echo ""

# Check if buildozer is installed
if ! command -v buildozer &> /dev/null; then
    echo "Buildozer not found. Installing..."
    pip install buildozer cython==0.29.33
fi

# Parse arguments
BUILD_MODE="debug"
TARGET="android"

while [[ $# -gt 0 ]]; do
    case $1 in
        release)
            BUILD_MODE="release"
            shift
            ;;
        clean)
            echo "Cleaning build directory..."
            rm -rf .buildozer
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./build.sh [release|clean]"
            exit 1
            ;;
    esac
done

echo "Build mode: $BUILD_MODE"
echo ""

# Check for Android SDK
if [ -z "$ANDROID_HOME" ]; then
    echo "WARNING: ANDROID_HOME not set. Build may fail."
    echo "Set it with: export ANDROID_HOME=/path/to/android-sdk"
    echo ""
fi

# Build the APK
echo "Starting build..."
echo ""

if [ "$BUILD_MODE" == "release" ]; then
    buildozer android release
    APK_PATH="bin/*.apk"
else
    buildozer android debug
    APK_PATH="bin/*.apk"
fi

# Find and display APK
echo ""
echo "============================================"
echo "Build complete!"
echo ""

if ls bin/*.apk 1> /dev/null 2>&1; then
    APK_FILE=$(ls -la bin/*.apk | head -n 1 | awk '{print $NF}')
    APK_SIZE=$(du -h "$APK_FILE" | cut -f1)
    echo "APK Location: $APK_FILE"
    echo "APK Size: $APK_SIZE"
    echo ""
    echo "To install on device:"
    echo "  adb install -r $APK_FILE"
else
    echo "WARNING: APK not found in bin/"
    echo "Check build logs for errors."
fi

echo "============================================"
