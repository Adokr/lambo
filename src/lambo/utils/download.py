"""
Procedures for downloading a model from the model repository. The code is based on analogous `COMBO module <https://gitlab.clarin-pl.eu/syntactic-tools/combo/-/blob/develop/combo/utils/download.py>`_
"""

import errno
import logging
import os
from pathlib import Path

import requests
import tqdm
import urllib3
from requests import adapters, exceptions

logger = logging.getLogger(__name__)

# The types of models available and their subdirectories in the model repository
TYPE_TO_PATH = {
    "LAMBO_211_no_pretraining": "vanilla211-s",
    "LAMBO_211": "full211-s",
    "LAMBO_213": "full213-s-withunk"}

default_type = 'LAMBO_213'

# The adress of the remote repository
_URL = "http://home.ipipan.waw.pl/p.przybyla/lambo/{type}/{treebank}.{extension}"

# The adress of the remote repository for optional variants
_URL_VAR = "http://home.ipipan.waw.pl/p.przybyla/lambo/{type}/{treebank}_{variant}.{extension}"

_HOME_DIR = os.getenv("HOME", os.curdir)

# Models are stored in ~/.lambo/
_CACHE_DIR = os.getenv("LAMBO_DIR", os.path.join(_HOME_DIR, ".lambo"))


def download_model(model_name, force=False):
    """
    Retrieve a model from the model repository, either local one, or available remotely.

    :param model_name: a full model name, e.g. ``LAMBO_no_pretraining-UD_Polish-PDB``, to retrieve
    :param force: whether model-redownload should be forced
    :return: a list of locations: for the dictionary file and the Pytorch network file of the downloaded model
    """
    _make_cache_dir()
    type = model_name.split("-", 1)[0]
    treebank = model_name.split("-", 1)[1]
    locations = []
    # First download pass for the main model
    for extension in ['dict', 'pth']:
        url = _URL.format(type=TYPE_TO_PATH[type], treebank=treebank, extension=extension)
        local_filename = model_name + '.' + extension
        location = os.path.join(_CACHE_DIR, local_filename)
        locations.append(Path(location))
        if os.path.exists(location) and not force:
            logger.debug("Using cached data.")
            continue
        chunk_size = 1024
        logger.info(url)
        try:
            with _requests_retry_session(retries=2).get(url, stream=True) as r:
                pbar = tqdm.tqdm(unit="B", total=int(r.headers.get("content-length")),
                                 unit_divisor=chunk_size, unit_scale=True)
                with open(location, "wb") as f:
                    with pbar:
                        for chunk in r.iter_content(chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
        except exceptions.RetryError:
            raise ConnectionError(f"Couldn't find or download model. "
                                  "Check if model name is correct or try again later!")
    # Second download pass for the variants
    for variant in ['subwords']:
        extension = 'pth'
        url = _URL_VAR.format(type=TYPE_TO_PATH[type], treebank=treebank, variant=variant, extension=extension)
        local_filename = model_name + '_' + variant + '.' + extension
        location = os.path.join(_CACHE_DIR, local_filename)
        if os.path.exists(location) and not force:
            logger.debug("Using cached data.")
            locations.append(Path(location))
            continue
        chunk_size = 1024
        logger.info(url)
        try:
            with _requests_retry_session(retries=2).get(url, stream=True) as r:
                pbar = tqdm.tqdm(unit="B", total=int(r.headers.get("content-length")),
                                 unit_divisor=chunk_size, unit_scale=True)
                with open(location, "wb") as f:
                    with pbar:
                        for chunk in r.iter_content(chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
        except exceptions.RetryError:
            # This is normal if splitter was not trained
            print("Couldn't find or download model variant (" + variant + ") -- might be unavailable.")
            locations.append(None)
            continue
        locations.append(Path(location))
    return locations


def _make_cache_dir():
    """
    Create cache directory, unless it exists.
    
    :return: no value returned
    """
    try:
        os.makedirs(_CACHE_DIR)
        logger.info(f"Making cache dir {_CACHE_DIR}")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _requests_retry_session(
        retries=3,
        backoff_factor=0.3,
        status_forcelist=(404, 500, 502, 504),
        session=None,
):
    """
    Create a session for download with retries, based on `Peterbe.com <https://www.peterbe.com/plog/best-practice-with-retries-with-requests>`_
    
    :param retries: number of retries
    :param backoff_factor: backoff factor
    :param status_forcelist: error codes to force
    :param session: session to reuse
    :return: a session to use
    """
    session = session or requests.Session()
    retry = urllib3.Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = adapters.HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
