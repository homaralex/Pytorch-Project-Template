import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
from shutil import copyfile

import gin

from pytorch_template.utils.dirs import make_exp_dirs, CHECKPOINTS_DIR_GIN_MACRO_NAME, TBOARD_DIR_GIN_MACRO_NAME, \
    OUT_DIR_GIN_MACRO_NAME, LOG_DIR_GIN_MACRO_NAME
from pytorch_template.utils.repo import save_repo_archive


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.removeHandler(main_logger.handlers[0])
    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def _gin_add_kwargs(gin_kwargs: dict):
    """Updates the gin config by adding the passed values as gin macros."""
    for key, val in gin_kwargs.items():
        gin.bind_parameter(binding_key=f'%{key}', value=val)


def process_gin_config(config_file, gin_kwargs: dict):
    # add custom values not provided in the config file as macros
    _gin_add_kwargs(gin_kwargs)

    gin.parse_config_file(config_file=config_file)

    # create some important directories to be used for that experiment
    summary_dir, checkpoints_dir, out_dir, log_dir = make_exp_dirs(exp_name=gin.REQUIRED)

    # setup logging in the project
    setup_logging(log_dir)
    logger = logging.getLogger()

    # make important paths available through the Gin config
    _gin_add_kwargs({
        CHECKPOINTS_DIR_GIN_MACRO_NAME: checkpoints_dir,
        TBOARD_DIR_GIN_MACRO_NAME: summary_dir,
        LOG_DIR_GIN_MACRO_NAME: log_dir,
        OUT_DIR_GIN_MACRO_NAME: out_dir,
    })

    logger.info(f"The experiment name is '{gin.query_parameter('%exp_name')}'")
    logger.info("Configuration:")
    logging.info(gin.config.config_str())

    # copy the gin config file into the current run's directory
    copyfile(config_file, checkpoints_dir / 'config.gin')
    # save current repo state
    save_repo_archive(filename=checkpoints_dir / 'repo.tar')
    logger.info(f'Saved config and repo to {checkpoints_dir}.')

    logger.info("Configurations are successfully processed and dirs are created.")
    logger.info("The pipeline of the project will begin now.")
