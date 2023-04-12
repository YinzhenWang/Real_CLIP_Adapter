import logging


def create_logger(output_file, add_stream=True):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add file handler
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add stream handler
    if add_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger