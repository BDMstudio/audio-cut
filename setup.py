#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: setup.py
# AI-SUMMARY: 项目安装配置文件

"""
智能人声分割器项目安装配置
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# 读取requirements文件
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="vocal-smart-splitter",
    version="2.5.1",
    description="智能人声分割器 - 基于人声内容和换气停顿的智能音频分割工具（支持多特征副歌检测）",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="BDM Team",
    author_email="bdm@example.com",
    url="https://github.com/bdm/vocal-smart-splitter",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "vocal_smart_splitter": ["config.yaml"],
    },
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="audio, voice, splitting, AI, machine learning, signal processing",
    project_urls={
        "Documentation": "https://github.com/bdm/vocal-smart-splitter/blob/main/README.md",
        "Source": "https://github.com/bdm/vocal-smart-splitter",
        "Tracker": "https://github.com/bdm/vocal-smart-splitter/issues",
    },
)
