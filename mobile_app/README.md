# UPRYT Mobile - Android APK Builder

Build an Android APK from the posture analysis system.

## Contents

| File | Purpose |
|------|---------|
| `main.py` | Kivy app with camera & UI |
| `posture_detector.py` | Mobile-optimized MediaPipe detector |
| `posture_screen.kv` | Kivy UI layout |
| `buildozer.spec` | Build configuration |
| `requirements.txt` | Python dependencies |
| `build.bat` | Windows build script |
| `build.sh` | Linux/Mac build script |
| `Dockerfile` | Docker build environment |

## Quick Start

### Docker (Recommended)
```bash
docker build -t upryt-builder .
docker run -v $(pwd):/app upryt-builder
```

APK output: `bin/upryt_posture-1.0.0-*-debug.apk`

### Linux/Mac
```bash
pip install -r requirements.txt
export ANDROID_HOME=/path/to/sdk
./build.sh
```

### Windows
Windows can't run Buildozer natively. Use Docker or WSL2.

## Build Requirements

- Python 3.8+
- 4GB+ RAM
- 10GB+ disk space
- Android SDK (or Docker)
- Java JDK 11+ (or Docker)

## Building Steps

1. Install Buildozer:
   ```bash
   pip install buildozer cython==0.29.33
   ```

2. Install Android SDK:
   ```bash
   export ANDROID_HOME=/path/to/sdk
   yes | $ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager --licenses
   ```

3. Build:
   ```bash
   buildozer android debug
   ```

## Installing APK

```bash
adb install -r bin/*.apk
```

## Mobile App Features

- Real-time camera preview
- Posture score (0-100%)
- Attention tracking
- Audio alerts
- Settings & statistics

## Customization

Edit `buildozer.spec`:
```ini
title = My App
package.name = myapp
version = 1.0.0
```

## APK Size

~80-150MB (arm64-v8a + armeabi-v7a)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| buildozer not found | `pip install buildozer` |
| ANDROID_HOME not set | Set environment variable |
| Build fails | Use Docker instead |
