from setuptools import setup, find_packages

setup(
    name="rank_validation",
    version="0.0.7",
    description="creates a rank validation report",
    py_modules=["validation_generator", "metrics", "normalise"],
    install_requires=["numpy", "pandas", "swifter", "rbo", "scipy", "typing"],
    author="Akash Dubey",
    url="https://github.com/akashkdubey/ranking_validation",
    author_email="akashdubey826@gmail.com",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ]
)
