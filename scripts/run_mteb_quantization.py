"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging
import functools
from sentence_transformers import SentenceTransformer

from mteb import MTEB
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel

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
model = DRESModel(SentenceTransformer(model_name, trust_remote_code=True, device='cuda:0'))

old_model_encode_queries = model.encode_queries
old_model_encode_corpus = model.encode_corpus

batch_size_map = {'CQADupstackAndroidRetrieval': 1,
                  'CQADupstackEnglishRetrieval': 1,
                  'CQADupstackGamingRetrieval': 1,
                  'CQADupstackGisRetrieval': 1, 'CQADupstackMathematicaRetrieval': 1, 'CQADupstackPhysicsRetrieval': 1,
                  'CQADupstackProgrammersRetrieval': 1, 'CQADupstackStatsRetrieval': 1, 'CQADupstackTexRetrieval': 1,
                  'CQADupstackUnixRetrieval': 1, 'CQADupstackWebmastersRetrieval': 1, 'CQADupstackWordpressRetrieval': 1}
QUANT_FUNTION = {'ubinary': 'dot', 'int8': 'cos_sim'}

for task in TASK_LIST:
    for quant in QUANTIZATION:
        logger.info(f"Running task: {task} with quantization: {quant}")
        # normalize_embeddings should be true for this model
        if quant == 'int8':
            model.encode_queries = functools.partial(old_model_encode_queries, precision=quant)
            model.encode_corpus = functools.partial(old_model_encode_corpus, precision=quant)
        else:
            model.encode_queries = functools.partial(old_model_encode_queries, precision='float32')
            model.encode_corpus = functools.partial(old_model_encode_corpus, precision=quant)
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(
            tasks=[task], task_langs=["en"]
        )  # Remove "en" for running all languages
        evaluation.run(
            model, output_folder=f"results/{quant}/{model_name}", eval_splits=eval_splits,
            score_function=QUANT_FUNTION[quant], batch_size=batch_size_map.get(task, 16)
        )

