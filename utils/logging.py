import logging


def get_event_logger(logger_name: str, filename: str, console_print: bool = True, level: str = 'info') -> logging.Logger:
    logging.basicConfig(
        filename=filename,
        format='[%(levelname)s] %(name)s at %(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %I:%M:%S %p',
        level=dict(
            info=logging.INFO,
            debug=logging.DEBUG,
            warning=logging.WARNING,
            error=logging.ERROR,
            critical=logging.CRITICAL
        )[level]
    )
    logger = logging.getLogger(logger_name)
    if console_print:
        logger.addHandler(logging.StreamHandler())
    return logger
