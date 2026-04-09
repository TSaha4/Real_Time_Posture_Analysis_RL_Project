from setuptools import setup, find_packages
import os

def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""

setup(
    name="upryt",
    version="2.0.0",
    author="UPRYT Team",
    author_email="info@upryt.com",
    description="Real-Time Posture Analysis with Reinforcement Learning",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/upryt/posture-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/upryt/posture-analysis/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "mediapipe>=0.10.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "upryt=main:main",
            "upryt-gui=gui_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "pose_landmarker_lite.task",
            "models/*.pth",
            "logo/*.png",
        ],
    },
    zip_safe=False,
)
