import os


def get_script_evolution_slave():
    script_path = os.path.dirname(os.path.abspath(__file__)) + '/evolution_slave.py'
    # script_path = '~/.local/bin/evolution_slave.py'
    return script_path


def get_script_evolution_slave_fenchel():
    script_path = os.path.dirname(os.path.abspath(__file__)) + '/evolution_slave_fenchel.py'
    # script_path = '~/.local/bin/evolution_slave_fenchel.py'
    return script_path
