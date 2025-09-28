import os

def get_script_evolution_slave():
    script_path = os.path.dirname(os.path.abspath(__file__)) + '/evolution_slave.py'
    return script_path


def get_compile_heatmap_slave():
    script_path = os.path.dirname(os.path.abspath(__file__)) + '/compile_heatmap_slave.py'
    return script_path


def get_compile_heatmap_v2_slave():
    script_path = os.path.dirname(os.path.abspath(__file__)) + '/compile_heatmap_v2_slave.py'
    return script_path
