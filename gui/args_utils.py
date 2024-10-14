import argparse
import ast

def args_to_config(parser):

    parser.add_argument("--MODEL", type=str)
    parser.add_argument("--EPOCHS", type=int)
    parser.add_argument("--BATCH_SIZE", type=int)
    parser.add_argument("--LEARNING_RATE", type=float)
    parser.add_argument("--TRAIN_PART", type=float)
    parser.add_argument("--SCHEDULER", type=str)
    parser.add_argument("--DEBUG", type=str)
    parser.add_argument("--DATASET", type=str)
    parser.add_argument("--LOSS", type=str)    
    parser.add_argument("--OPTIMIZER", type=str)
    parser.add_argument("--MOMENTUM", type=float)
    parser.add_argument("--WEIGHT_DECAY", type=float)
    parser.add_argument("--PRECISION", type=int)
    parser.add_argument("--NDWI", type=str)
    parser.add_argument("--EVAL_FREQ", type=int)
    parser.add_argument("--SCALE", type=int)
    parser.add_argument("--ABLATION", type=int)
    parser.add_argument("--INCLUDE_AMM", type=str)

    args = parser.parse_args()
    
    config = {
        'MODEL': args.MODEL,
        'EPOCHS': args.EPOCHS,
        'BATCH_SIZE': args.BATCH_SIZE,
        'LEARNING_RATE': args.LEARNING_RATE,
        'TRAIN_PART': args.TRAIN_PART,
        'SCHEDULER': args.SCHEDULER,
        'DEBUG': ast.literal_eval(args.DEBUG) if args.DEBUG in ['True', 'False'] else True,
        'DATASET': args.DATASET,
        'LOSS': args.LOSS,
        'OPTIMIZER': args.OPTIMIZER,
        'MOMENTUM': args.MOMENTUM,
        'WEIGHT_DECAY': args.WEIGHT_DECAY,
        'PRECISION': args.PRECISION,
        'NDWI': ast.literal_eval(args.NDWI) if args.NDWI in ['True', 'False'] else False,
        'EVAL_FREQ': args.EVAL_FREQ,
        'SCALE': args.SCALE,
        'ABLATION': args.ABLATION,
        'INCLUDE_AMM': ast.literal_eval(args.INCLUDE_AMM) if args.INCLUDE_AMM in ['True', 'False'] else False
    }

    return config

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    cfg = args_to_config(p)
    print(cfg)