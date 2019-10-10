from setuptools import setup, find_packages


def parse_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='pytorch_template',
    version='1.0',
    packages=find_packages(),
    install_requires=parse_requirements(),
    scripts=['scripts/clear_experiments.sh', 'scripts/launch_tboard.sh'],
)
