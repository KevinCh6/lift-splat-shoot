"""
Learning copy package initializer.

The original root-level ``src/__init__.py`` imports training and exploration
modules immediately. For step-through learning we keep this copy lightweight so
``from src.models import ...`` does not require nuScenes/OpenCV dependencies.
"""
