"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging
import functools
import numpy as np
from torch import Tensor
from sentence_transformers import SentenceTransformer

from mteb import MTEB

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")


TASK_LIST_RETRIEVAL = [
    "ArguAna",
    # "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    # "FEVER",
    "FiQA2018",
    # "HotpotQA",
    # "MSMARCO",
    "NFCorpus",
    # "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

QUANTIZATION = ['ubinary', 'int8']

TASK_LIST = TASK_LIST_RETRIEVAL

model_name = "jinaai/jina-embeddings-v2-base-en"
model = SentenceTransformer(model_name, trust_remote_code=True, device='cuda:0')

old_model_encode = model.encode

QUANT_FUNTION = {'ubinary': 'hamming', 'int8': 'cos_sim'}

for task in TASK_LIST:
    for quant in QUANTIZATION:
        logger.info(f"Running task: {task} with quantization: {quant}")
        # normalize_embeddings should be true for this model
        model.encode = functools.partial(old_model_encode, precision=quant)
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(
            tasks=[task], task_langs=["en"]
        )  # Remove "en" for running all languages
        evaluation.run(
            model, output_folder=f"results/{quant}/{model_name}", eval_splits=eval_splits,
            score_function=QUANT_FUNTION[quant], batch_size=16
        )

