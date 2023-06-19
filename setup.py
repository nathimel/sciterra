import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sciterra",
    version="0.0.1",
    author="Nathaniel Imel",
    author_email="nimel@uci.edu",
    description="Scientific literature exploration and topographic analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nathimel/sciterra",
    project_urls={"Bug Tracker": "https://github.com/nathimel/sciterra/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    # python_requires=">=3.6",
    install_requires=[
        "bibtexparser",
        "numpy",
        "torch",
        "transformers",
        "scikit-learn",
        "pandas",
        # "plotnine",
        "pytest",
    ],
)
