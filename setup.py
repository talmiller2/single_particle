from setuptools import setup

setup(
    name="single_particle",
    author="Tal Miller",
    author_email="talmiller@gmail.com",
    packages=['em_fields', 'loop_fields'],
    # package_data={'em_fields': ['evolution_slave.py']},
    # scripts=['em_fields/evolution_slave.py', 'em_fields/evolution_slave_fenchel.py'],
)

# install locally using: python setup.py install --user
# another option: pip install . --user --upgrade
# save exe permissions that git accepts: git update-index --chmod=+x file_name
# the local install dir: ~/.local/lib/python3.6/site-packages/
# can also not install at all and add ~/.bashrc: export PYTHONPATH="$PYTHONPATH:/home/talm/code/single_particle"
