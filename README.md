# sciterra: a python library for similarity-based scientometrics

[![build](https://github.com/nathimel/sciterra/actions/workflows/build.yml/badge.svg)](https://github.com/nathimel/sciterra/actions/workflows/build.yml)

Sciterra is a software libary to support data-driven analyses of scientific literature, with a focus on unifying different bibliographic database APIs and document-embedding methods for systematic scientometrics research.

## Overview

The main purpose of sciterra is to perform similarity-based retrieval of scientific publications for metascience/scientometrics research. While there are many services that can make the individual steps of this simple, this software library exists to

1. Unify the different APIs and vector-based retrieval methods

2. Support scientometrics analyses of citation dynamics, especially with respect to a vectorized 'landscape' of literature.

## Installing sciterra

First, set up a virtual environment (e.g. via [miniconda](https://docs.conda.io/projects/miniconda/en/latest/), `conda create -n sciterra`, and `conda activate sciterra`).

1. Install sciterra via git:

    `python -m pip install 'sciterra @ git+https://github.com/nathimel/sciterra.git'`

2. Alternatively, download or clone this repository and navigate to the root folder, and install locally:

    `pip install -e .`

3. It is not yet recommended because sciterra is still in development, but you can also install via pip from pypi:

    `pip install sciterra`

You will also need to download a trained pipeline for [spacy](https://spacy.io/usage):

`python -m spacy download en_core_web_sm`    

Optional: If you plan on querying the NASA Astrophysical Data System (ADS), you must have an ADS API key saved at `~/.ads/dev_key`.
To generate an ADS API key navigate to [the ADS web interface](https://ui.adsabs.harvard.edu/), create and sign in to an ADS account, and navigate to [Settings > API Token](https://ui.adsabs.harvard.edu/user/settings/token).

## Tests

To run all the unit tests for sciterra, found at [src/tests](https://github.com/nathimel/sciterra/tree/main/src/tests), run the following command at the root of the repository:

`pytest`

This may take up to several hours in total, due to slow api calls in `test_cartography` and `test_tracing`.

Note: If you opted not to set up authentication for ADS during the set up, the tests in `test_librarian.TestADSLibrarian` and the test `test_tracing.TestExpansion.test_atlas_tracer_ads` will fail.
<!-- TODO: Add an `--ads` flag to the pytests that is turned off by default. Turn it on if the user is using ADS. -->


## Usage

### Atlas

The central object in sciterra is the [`Atlas`](src/sciterra/mapping/atlas.py). This is a basic data structure for containing scientific publications that are returned from calls to various bibliographic database APIs.

An Atlas minimally requires a list of [`Publications`](src/sciterra/mapping/publication.py).

#### Publication

A publication object is a minimal wrapper around publication data, and should have a string identifier. It is designed to standardize the basic metadata contained in the results from some bibliographic database API.

```python
from sciterra import Atlas, Publication

atl = Atlas([Publication({"identifier": "id"})])
```

Alternatively, you can construct an Atlas by passing in a .bib file. The entries in this bibtex file will be parsed for unique identifiers (e.g., DOIs), and sent in an API call, and returned as Publications, which then populate an Atlas.

```python
atl = crt.bibtex_to_atlas(bibtex_filepath)
```

In the line of code above, the variable `crt` is an instance of a [`Cartographer`](src/sciterra/mapping/cartography.py) object, which encapsulates the bookkeeping involved in querying a bibliographic database for publications.

### Cartographer

The Cartographer class is named because interfaces with an Atlas to build out a library of publications. Since it does so via similarity-based retrieval, the resulting Atlas can be considered a 'region' of publications.

To do this, a Cartographer needs two things: an API with which to interface, and a way of getting document embeddings. Both are encapsulated, respectively, by the [`Librarian`](src/sciterra/librarians/librarian.py) and the [`Vectorizer`](src/sciterra/vectorization/vectorizer.py) classes.

```python
from sciterra import Cartographer
from sciterra.librarians import SemanticScholarLibrarian # or ADSLibrarian
from sciterra.vectorization import SciBERTVectorizer # among others

crt = Cartographer(
    librarian=SemanticScholarLibrarian(),
    vectorizer=SciBERTVectorizer(),
)
```

#### Librarian

Each Librarian subclass is designed to be a wrapper for an existing python API service, such as the [ads](https://ads.readthedocs.io/en/latest/) package or the [semanticscholar](https://github.com/danielnsilva/semanticscholar#) client library.

A Librarian subclass also overrides two methods. The first is `get_publications`, which takes a list of identifiers, should query the specific API for that Librarian, and returns a list of Publications. Keyword arguments can be passed to specify the metadata that is kept for each publication (e.g. date, title, journal, authors, etc.) The second method is `convert_publication`, which defines how the result of an API call should be converted to a sciterra Publication object.

Contributions to sciterra in the form of new Librarian subclasses are encouraged and appreciated.

### Vectorizer

Vectorizer subclasses override one function, `embed_documents`, which takes a list of strings, representing the text of a publication (currently, just its abstract), and returns an `np.ndarray` of embeddings.

Under the hood, the `project` method of Cartographer, which is used during similarity-based retrieval, uses the vectorizer roughly as follows

```python
# Get abstracts
docs = [atlas[identifier].abstract for identifier in identifiers]

# Embed abstracts
result = vectorizer.embed_documents(docs)
embeddings = result["embeddings"]

# depending on the vectorizer, sometimes not all embeddings can be obtained due to out-of-vocab issues
success_indices = result["success_indices"] # shape `(len(embeddings),)`
fail_indices = result["fail_indices"] # shape `(len(docs) - len(embeddings))``
```

Currently, sciterra has vectorizers using [SciBERT](https://aclanthology.org/D19-1371/), [SBERT](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models), [GPT-2](https://huggingface.co/docs/transformers/en/model_doc/gpt2), [Word2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#), and a simple bag-of-words (BOW) vectorizer that uses the same vocabulary as the Word2Vec vectorizer. Contributions to sciterra in the form of new Vectorizer subclasses are also encouraged and appreciated.

### Putting it all together

The main use case for all of these ingredients is to iteratively build out a region of publications. This is done using `iterate_expand`:

```python
from sciterra.mapping.tracing import iterate_expand

# Assuming the initial atlas contains just one publication
(atl.center, ) = atl.publications.keys()
# build out an atlas to contain 10,000 publications, with increasing dissimilarity to the initial publication, saving progress in binary files to the directory named "atlas".
iterate_expand(
    atl=atl,
    crt=crt,
    atlas_dir="atlas",
    target_size=10000,
    center=atl.center,
)
```

This method has a number of keyword arguments that enable tracking the Atlas expansion, limiting the number of publications per expansion, how many times to try to get a response if there are connection issues, etc.

In practice, it may be helpful to use the [`sciterra.mapping.tracing.AtlasTracer`](src/sciterra/mapping/tracing.py) data structure to reduce most of the loading/initialization boilerplate described above. For an example, see [main.py](src/examples/scratch/main.py).

## Additional features

- The [topography](src/sciterra/mapping/topography.py) submodule contains similarity-based metrics for publications, to support scientometrics analyses.

## Acknowledgments

This software is a reimplimentation of Zachary Hafen-Saavedra's library, [cc](https://github.com/zhafen/cc).

To cite sciterra, please use the following workshop paper,

```
@inproceedings{Imel2023,
 author = {Imel, Nathaniel, and Hafen, Zachary},
 title = {Citation-similarity relationships in astrophysics},
 booktitle = {AI for Scientific Discovery: From Theory to Practice Workshop (AI4Science @ NeurIPS)},
 year = {2023},
 url = {https://openreview.net/pdf?id=mISayy7DPI},
}
```
