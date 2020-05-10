import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir=None):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    if log_dir is not None:
        exp_file_handler = RotatingFileHandler(log_dir / 'exp_debug.log', maxBytes=10 ** 6, backupCount=5)
        exp_file_handler.setLevel(logging.DEBUG)
        exp_file_handler.setFormatter(Formatter(log_file_format))

        exp_errors_file_handler = RotatingFileHandler(log_dir / 'exp_error.log', maxBytes=10 ** 6, backupCount=5)
        exp_errors_file_handler.setLevel(logging.WARNING)
        exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    if len(main_logger.handlers) > 0:
        # remove tensorboard-added unnecessary info handler
        main_logger.removeHandler(main_logger.handlers[0])

    main_logger.addHandler(console_handler)
    if log_dir is not None:
        main_logger.addHandler(exp_file_handler)
        main_logger.addHandler(exp_errors_file_handler)
