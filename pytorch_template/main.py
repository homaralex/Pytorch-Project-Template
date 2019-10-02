"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Parses the gin config file, creates necessary directories and
finally instantiates an Agent object which then launches its pipeline.
"""

import argparse

import gin

from pytorch_template import agents as base_agents_module
from pytorch_template.utils.config import process_gin_config


@gin.configurable
def make_agent(agent_name, agents_module):
    agent_class = getattr(agents_module or base_agents_module, agent_name)
    agent = agent_class()

    return agent


def main(agents_module=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'config',
        metavar='gin_config_file',
        help='Gin config file (see https://github.com/google/gin-config)',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        # default value specified for fast in-code changes
        default=False,
    )
    args = parser.parse_args()

    process_gin_config(config_file=args.config, gin_kwargs={'debug': args.debug})

    # create an Agent object and let it run its pipeline
    agent = make_agent(agents_module=agents_module)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
