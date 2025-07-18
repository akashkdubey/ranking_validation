from setuptools import setup, find_packages

setup(
    author="Akash Dubey",
    url="https://github.com/akashkdubey/ranking_validation",
    author_email="akashdubey826@gmail.com",
    name="rank_validation",
    version="1.2.0",
    description="Lightweight, vectorised ranking-metric toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.2",
        "pandas>=1.4",
        "scipy>=1.8",
        "rbo>=0.1.1",          
    ],
    python_requires=">=3.8",
    classifiers=[
    "Development Status :: 4 - Beta",          
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Text Processing :: Linguistic",
    "Operating System :: OS Independent",
    "Typing :: Typed",                         
],
    entry_points={
        "console_scripts": ["rank-validation=rank_validation.cli:main"],
    },
)

