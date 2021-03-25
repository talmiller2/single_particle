import os

tmp = 5

def get_script_evolution_slave():
    script_path = os.path.dirname(os.path.abspath(__file__)) + '/evolution_slave.py'
    return script_path
