import logging

from mteb import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

retrieval_tasks = [
    "ArguAna-PL",
    "DBPedia-PL",
    "FiQA-PL",
    "HotpotQA-PL",
    "MSMARCO-PL",
    "NFCorpus-PL",
    "NQ-PL",
    "Quora-PL",
    "SCIDOCS-PL",
    "SciFact-PL",
    "TRECCOVID-PL",
]

tasks = retrieval_tasks

model_name = "ipipan/silver-retriever-base-v1.1"
model = SentenceTransformer(model_name)

evaluation = MTEB(tasks=tasks, task_langs=["pl"])
evaluation.run(model, output_folder=f"results/pl/{model_name.split('/')[-1]}")
