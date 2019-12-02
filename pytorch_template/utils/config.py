import logging
from shutil import copyfile

import gin
from git import InvalidGitRepositoryError

from pytorch_template.utils.dirs import make_exp_dirs, CHECKPOINTS_DIR_GIN_MACRO_NAME, TBOARD_DIR_GIN_MACRO_NAME, \
    OUT_DIR_GIN_MACRO_NAME, LOG_DIR_GIN_MACRO_NAME
from pytorch_template.utils.logs import setup_logging
from pytorch_template.utils.repo import save_repo_archive


def _gin_add_macros(gin_macros: dict):
    """Updates the gin config by adding the passed values as gin macros."""
    for key, val in gin_macros.items():
        gin.bind_parameter(binding_key=f'%{key}', value=val)


def process_gin_config(config_file, gin_macros: dict = None):
    # add custom values not provided in the config file as macros
    _gin_add_macros(gin_macros or {})

    # TODO not the cleanest way
    gin.bind_parameter('configure_device.gpu_id', gin.query_parameter('%gpu_id'))

    gin.parse_config_file(config_file=config_file)

    # create some important directories to be used for that experiment
    summary_dir, checkpoints_dir, out_dir, log_dir = make_exp_dirs(exp_name=gin.REQUIRED)

    # setup logging in the project
    setup_logging(log_dir)
    logger = logging.getLogger()

    # make important paths available through the Gin config
    _gin_add_macros({
        CHECKPOINTS_DIR_GIN_MACRO_NAME: checkpoints_dir,
        TBOARD_DIR_GIN_MACRO_NAME: summary_dir,
        LOG_DIR_GIN_MACRO_NAME: log_dir,
        OUT_DIR_GIN_MACRO_NAME: out_dir,
    })

    logger.info(f"The experiment name is '{gin.query_parameter('%exp_name')}'")
    logger.info("Configuration:")
    logging.info(gin.config.config_str())

    # save the whole gin config into the current run's directory
    print(gin.config.config_str(), file=(checkpoints_dir / 'config.gin').open(mode='w'))
    logger.info(f'Saved config to {checkpoints_dir}.')

    # save current repo state
    try:
        save_repo_archive(filename=checkpoints_dir / 'repo.tar')
        logger.info(f'Saved repo to {checkpoints_dir}.')
    except InvalidGitRepositoryError as e:
        logger.warning(f'Could not save repo. GitPython error:\n{e}.')

    logger.info("Configurations are successfully processed and dirs are created.")
    logger.info("The pipeline of the project will begin now.")
