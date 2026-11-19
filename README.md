# StringLab Enterprise – String Search Performance Studio

An interactive Python-based application for exploring, analyzing, and benchmarking string matching and information retrieval algorithms. The project demonstrates Trie-based prefix retrieval, Knuth–Morris–Pratt (KMP), Rabin–Karp, and Bloom Filter techniques through an interactive Streamlit interface.

## Overview

StringLab Enterprise is a local-first string search and performance analysis application designed to study string matching algorithms and retrieval techniques.

The application enables users to search text corpora, explore prefix-based retrieval, upload documents, compare string matching algorithms, conduct performance experiments, and export benchmark results.

## Features

- **KMP String Matching:** Efficient exact pattern matching using the Knuth–Morris–Pratt algorithm.
- **Rabin–Karp Algorithm:** String matching using rolling hash techniques.
- **Trie Explorer:** Prefix-based retrieval and autocomplete functionality.
- **Bloom Filter:** Probabilistic membership checking as an optional pre-filter.
- **Document Ingestion:** Supports TXT, PDF, and DOCX document uploads.
- **Corpus Search:** Search across indexed documents and view exact match positions.
- **Benchmark Lab:** Analyze algorithm performance across varying text and pattern lengths.
- **Algorithm Comparison:** Compare algorithm characteristics and performance.
- **Search History:** Track previous queries and associated search metadata.
- **CSV Export:** Export benchmark measurements for further analysis.
- **SQLite Storage:** Persist documents and search metadata locally.

## Technologies Used

- Python
- Streamlit
- SQLite
- KMP (Knuth–Morris–Pratt)
- Rabin–Karp
- Trie Data Structure
- Bloom Filter
- PyPDF
- DOCX Processing
- CSV

## Algorithms Implemented

### 1. Knuth–Morris–Pratt (KMP)

KMP performs efficient string matching by preprocessing the pattern and constructing a Longest Prefix Suffix (LPS) array.

**Time Complexity:** O(n + m)

### 2. Rabin–Karp

Rabin–Karp uses rolling hash calculations to identify potential pattern matches and verifies candidate matches to handle hash collisions.

**Average Time Complexity:** O(n + m)

**Worst-case Time Complexity:** O(nm)

### 3. Trie

Trie is a tree-based data structure used for efficient prefix-based retrieval and autocomplete.

**Prefix Lookup Complexity:** O(P), excluding the time required to return suggestions.

### 4. Bloom Filter

Bloom Filter is a probabilistic data structure used to check whether an element may exist in a set.

It can indicate that an element is definitely absent or possibly present, but it cannot guarantee that an element exists.

## Project Structure

```text
StringLab-Enterprise/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── algorithms/
│   ├── __init__.py
│   ├── bloom_filter.py
│   ├── kmp.py
│   ├── rabin_karp.py
│   └── trie.py
│
├── core/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── document_manager.py
│   ├── ingestion.py
│   ├── search_engine.py
│   ├── storage.py
│   └── text_processor.py
│
├── data/
│   ├── sample_documents/
│   └── test_documents/
│
├── tests/
│   ├── test_benchmark.py
│   ├── test_bloom_filter.py
│   ├── test_document_manager.py
│   ├── test_kmp.py
│   ├── test_rabin_karp.py
│   ├── test_search_engine.py
│   ├── test_storage_ingestion.py
│   ├── test_text_processor.py
│   └── test_trie.py
│
└── utils/
    ├── __init__.py
    └── timing.py
