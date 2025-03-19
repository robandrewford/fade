"""
Setup script for the fade package.
"""

from setuptools import setup, find_packages

setup(
    name="fade",
    version="0.1.0",
    description="Framework for Automated Document Entity extraction",
    author="FADE Team",
    author_email="robert.ford@wildebeest.ai",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "PyMuPDF>=1.25.3",
        "pdfplumber>=0.10.3",
        "camelot-py>=0.11.0",
        "unstructured>=0.12.0",
        "layoutparser>=0.3.4",
        "paddleocr>=2.7.0.3",
        "deepdoctection>=0.25",
        "langgraph>=0.0.15",
        "opencv-python<=4.6.0.66",
        "mcp-pdf-forms @ git+https://github.com/Wildebeest/mcp_pdf_forms.git",
        "matplotlib>=3.8.0",
        "notebook>=7.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "scipy>=1.11.0",
        "tqdm>=4.65.0",
        "ipykernel>=6.0.0",
        "jupyter>=1.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 