from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text('utf-8')

setup(
    name='siinn',
    version='0.1.0',
    packages=['siinn'],
    entry_points={
        'console_scripts': [
            'siinn = siinn:main'
        ]
    },
    python_requires='>=3.6',
    install_requires=[
        'numpy',
    ]
)
