"""
Setup script para YOLO Macroinvertebrados.

Este script permite instalar el proyecto como un paquete Python.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Leer requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="yolo-macro-detect",
    version="1.0.0",
    author="Kevin Galeano",
    author_email="kevin.galeano@example.com",
    description="Pipeline de detección de macroinvertebrados acuáticos usando YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevingaleano/yolo-macro-detect",
    project_urls={
        "Bug Reports": "https://github.com/kevingaleano/yolo-macro-detect/issues",
        "Source": "https://github.com/kevingaleano/yolo-macro-detect",
        "Documentation": "https://github.com/kevingaleano/yolo-macro-detect/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "yolo-macro-detect=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "yolo",
        "object-detection",
        "macroinvertebrates",
        "aquatic",
        "computer-vision",
        "deep-learning",
        "robotics",
        "environmental-science",
    ],
    platforms=["any"],
    license="MIT",
    zip_safe=False,
) 