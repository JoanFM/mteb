"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging
import functools
import numpy as np
from torch import Tensor
from unittest.mock import patch
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers import SentenceTransformer

from mteb import MTEB

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")


def my_quantize_embeddings(
        embeddings,
        precision,
        ranges,
        calibration_embeddings,
):
    """
    Quantizes embeddings to a lower precision. This can be used to reduce the memory footprint and increase the
    speed of similarity search. The supported precisions are "float32", "int8", "uint8", "binary", and "ubinary".

    :param embeddings: Unquantized (e.g. float) embeddings with to quantize to a given precision
    :param precision: The precision to convert to. Options are "float32", "int8", "uint8", "binary", "ubinary".
    :param ranges: Ranges for quantization of embeddings. This is only used for int8 quantization, where the ranges
        refers to the minimum and maximum values for each dimension. So, it's a 2D array with shape (2, embedding_dim).
        Default is None, which means that the ranges will be calculated from the calibration embeddings.
    :type ranges: Optional[np.ndarray]
    :param calibration_embeddings: Embeddings used for calibration during quantization. This is only used for int8
        quantization, where the calibration embeddings can be used to compute ranges, i.e. the minimum and maximum
        values for each dimension. Default is None, which means that the ranges will be calculated from the query
        embeddings. This is not recommended.
    :type calibration_embeddings: Optional[np.ndarray]
    :return: Quantized embeddings with the specified precision
    """
    if isinstance(embeddings, Tensor):
        embeddings = embeddings.cpu().numpy()
    elif isinstance(embeddings, list):
        if isinstance(embeddings[0], Tensor):
            embeddings = [embedding.cpu().numpy() for embedding in embeddings]
        embeddings = np.array(embeddings)
    if embeddings.dtype in (np.uint8, np.int8):
        raise Exception("Embeddings to quantize must be float rather than int8 or uint8.")

    if precision == "float32":
        return embeddings.astype(np.float32)

    if precision.endswith("int8"):
        # Either use the 1. provided ranges, 2. the calibration dataset or 3. the provided embeddings
        if ranges is None:
            if calibration_embeddings is not None:
                ranges = np.vstack((np.min(calibration_embeddings, axis=0), np.max(calibration_embeddings, axis=0)))
            else:
                if embeddings.shape[0] < 100:
                    logger.warning(
                        f"Computing {precision} quantization buckets based on {len(embeddings)} embedding{'s' if len(embeddings) != 1 else ''}."
                        f" {precision} quantization is more stable with `ranges` calculated from more embeddings "
                        "or a `calibration_embeddings` that can be used to calculate the buckets."
                    )
                ranges = np.vstack((np.min(embeddings, axis=0), np.max(embeddings, axis=0)))
        starts = ranges[0, :]
        steps = (ranges[1, :] - ranges[0, :]) / 255

        if precision == "uint8":
            return ((embeddings - starts) / steps).astype(np.uint8)
        elif precision == "int8":
            return ((embeddings - starts) / steps - 128).astype(np.int8)

    if precision == "binary":
        return (embeddings > 0).astype(np.float16)

    if precision == "ubinary":
        return (embeddings > 0).astype(np.float16)

    raise ValueError(f"Precision {precision} is not supported")


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

QUANTIZATION = ['binary', 'int8']

TASK_LIST = TASK_LIST_RETRIEVAL


def my_function_to_patch():
    # Your custom implementation here
    pass


# Patch the function in the module
with patch('sentence_transformers.quantization.quantize_embeddings', new=my_quantize_embeddings):
    model_name = "jinaai/jina-embeddings-v2-base-en"
    model = SentenceTransformer(model_name, trust_remote_code=True, device='cuda:0')

    old_model_encode = model.encode

    QUANT_FUNTION = {'binary': 'hamming', 'int8': 'cos_sim'}

    for task in TASK_LIST:
        for quant in QUANTIZATION:
            logger.info(f"Running task: {task} with quantization: {quant}")
            # normalize_embeddings should be true for this model
            model.encode = functools.partial(model.encode, precision=quant)
            eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
            evaluation = MTEB(
                tasks=[task], task_langs=["en"]
            )  # Remove "en" for running all languages
            evaluation.run(
                model, output_folder=f"results/{quant}/{model_name}", eval_splits=eval_splits,
                score_function=QUANT_FUNTION[quant], batch_size=16
            )
