import argparse
from gui.args_utils import args_to_config
from pytorch_main import run

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    cfg = args_to_config(p)
    run(cfg)
