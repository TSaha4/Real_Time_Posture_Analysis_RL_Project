# UPRYT Mobile - Docker Build Environment
# Use this to build the Android APK on Windows/Mac

FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    openjdk-11-jdk \
    ant \
    autoconf \
    libtool \
    pkg-config \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libtinfo5 \
    cmake \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV ANDROID_HOME=/opt/android-sdk
ENV ANDROID_SDK_ROOT=/opt/android-sdk
ENV PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/tools/bin:$ANDROID_HOME/platform-tools

# Download Android SDK
RUN mkdir -p $ANDROID_HOME/cmdline-tools && \
    cd $ANDROID_HOME/cmdline-tools && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip -O cmdline-tools.zip && \
    unzip -q cmdline-tools.zip && \
    mv cmdline-tools latest && \
    rm cmdline-tools.zip

# Accept licenses and install SDK components
RUN yes | $ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager --licenses || true
RUN $ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager --install \
    "platforms;android-33" \
    "build-tools;33.0.0" \
    "platform-tools"

# Download NDK
RUN $ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager --install \
    "ndk;25.1.8937393"

# Install Buildozer
RUN pip install --upgrade pip && \
    pip install buildozer cython==0.29.33

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Build the APK
# Run with: docker run -v $(pwd):/app upryt-builder buildozer android debug
CMD ["buildozer", "android", "debug"]
