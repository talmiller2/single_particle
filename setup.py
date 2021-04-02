from setuptools import setup

setup(
    name="single_particle",
    author="Tal Miller",
    author_email="talmiller@gmail.com",
    packages=['em_fields', 'loop_fields'],
    scripts=['em_fields/evolution_slave.py', 'em_fields/evolution_slave_fenchel.py'],
)

# install locally using: python setup.py install --user
# another option: pip install . --user --upgrade
# save exe permissions that git accepts: git update-index --chmod=+x file_name
# ~/.local/lib/python3.6/site-packages/
