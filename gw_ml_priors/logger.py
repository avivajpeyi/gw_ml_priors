import logging


def get_logger():
    proj_logger = logging.getLogger("GW_ML_PRIORS")
    proj_logger.setLevel(logging.DEBUG)
    return proj_logger


logger = get_logger()
