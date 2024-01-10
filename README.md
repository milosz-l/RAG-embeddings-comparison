# RAG (Retrieval Augmented Generation) embeddings comparison

## Overview
This project explores the influence of various embeddings on the effectiveness of Retrieval Augmented Generation (RAG).

## Key Objectives
- Submitting new model to the [mteb benchmark](https://huggingface.co/spaces/mteb/leaderboard) leaderboard for retrieval task in Polish language. In the end we chose the [silver retriever v1.1](https://huggingface.co/ipipan/silver-retriever-base-v1.1).
- Building a streamlit app showcasing RAG in action.

## Project structure explained
```
├── assets                              # folder with assets for .md files
├── mteb_benchmark                      # folder with files related to mteb benchmark
│   ├── download_leaderboard.py         # script for fetching leaderboard
│   ├── retrieval_data_pl.csv           # result of download_leaderboard.py
│   ├── retrieve_ndcg_at_10.py          # nicely prints ndcg@10 values from results folder
│   ├── run_mteb.ipynb                  # notebook used in google colab for running mteb tasks
│   └── run_mteb.py                     # basic script used for running mteb tasks locally
├── .gitignore                          # contains a list of files and folders to be ignored by git
├── dokumentacja_koncowa.md             # final documentation in markdown format (in Polish)
├── dokumentacja_wstepna.md             # preliminary documentation in markdown format (in Polish)
├── main.py                             # main file of the streamlit app
├── Pipfile                             # defines the project's python dependencies
├── Pipfile.lock                        # provides a snapshot of the entire dependency tree
├── README.md                           # the file you're reading right now
├── Trials.ipynb                        # Jupyter notebook for various trial codes and tests
└── utils.py                            # utility functions for the streamlit app
```

NOTE:
- `mteb tasks` mean mteb tasks for retrieval in Polish language