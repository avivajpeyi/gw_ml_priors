from setuptools import find_packages, setup

NAME = "gw_ml_priors"


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


setup(
    name=NAME,
    version=find_meta("version"),
    author=find_meta("author"),
    author_email=find_meta("email"),
    maintainer=find_meta("author"),
    maintainer_email=find_meta("email"),
    url=find_meta("uri"),
    license=find_meta("license"),
    description=find_meta("description"),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "bilby",
        "black",
        "isort",
        "tensorflow",
        "scipy",
        "pytest",
        "seaborn",
        "cloudpickle",
    ],
    keywords=["prior"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
