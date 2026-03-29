# Buildozer specification for UPRYT Mobile App
# https://buildozer.readthedocs.io/

[app]

# App title and identity
title = UPRYT Posture
package.name = upryt_posture
package.domain = com.upryt

# App version
version = 1.0.0

# Requirements for Android
requirements = python3,kivy==2.2.0,opencv-python-headless,mediapipe,numpy

# Orientation
orientation = portrait

# Fullscreen mode
fullscreen = 1

# Android permissions
android.permissions = CAMERA, RECORD_AUDIO

# Icon (uncomment and set path to your icon)
# icon.filename = icon.png

# Android specific settings
android.allow_backup = True
android.api = 27
android.minapi = 24
android.ndk_api = 21

# Enable AndroidX
android.enable_androidx = True

# Bootstrap
bootstrap = sdl2

# SDL2 specific
sdl2_touch_mode = True

# Log level
log_level = 2

# Windows
window_title = UPRYT Posture

# Architecture
p4a_arch = arm64-v8a

# Recipe for mediapipe
p4a_recipe = mediapipe

[buildozer]

# Build directory
build_dir = ./.buildozer

# Bin directory for APK
bin_dir = ./bin

# Log level
log_level = 2

# Warn on root
warn_on_root = 1
