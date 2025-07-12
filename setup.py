from setuptools import setup, find_packages

setup(
    author="Akash Dubey",
    url="https://github.com/akashkdubey/ranking_validation",
    author_email="akashdubey826@gmail.com",
    name="rank_validation",
    version="1.1.8",
    description="Lightweight, vectorised ranking-metric toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scipy", "rbo"],
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

