import argparse
import logging
import torch

from loader.data_loader import DataLoader
from config.simple_gpt_config import SimpleGPTConfig
from trainer.simple_gpt_trainer import SimpleGPTTrainer
from models.simple_gpt import SimpleGPT


def main():
    # basic logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # parse command line arguments
    args, gptConfig = parse_arguments()

    # for reproducability
    torch.manual_seed(1337)

    # load data
    logging.info(f'loading data from {args.dataPath} ...')
    dataLoader = DataLoader(dataPath=args.dataPath, gptConfig=gptConfig)
    dataLoader.loadData()

    # here are all the unique characters that occur in this text
    logging.info(f'creating embeddings for loaded data ...')
    chars = sorted(list(set(dataLoader.data)))
    gptConfig.vocab_size = len(chars)
    logging.info(f'vocabulary size is {gptConfig.vocab_size}')
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # slit training and validation data
    logging.info(f'preparing data for training ...')
    dataLoader.splitData(encode=encode)

    logging.info(f'creating the model')
    model = SimpleGPT(gptConfig)
    gpt = model.to(gptConfig.device)
    # print the number of parameters in the model
    logging.info(f'{sum(p.numel() for p in gpt.parameters())/1e6} million parameters')

    gptTrainer = SimpleGPTTrainer(model, gptConfig, dataLoader)
    logging.info(f'training the model ...')
    gptTrainer.train()
    logging.info(f'training done!')

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=gptConfig.device)
    logging.info(f'samling the model')
    print(decode(gpt.generate(context, max_new_tokens=500)[0].tolist()))
    #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


def setupGPTConfig(args):
    # configuration of gpt hyperparameters
    gptConfig = SimpleGPTConfig()
    gptConfig.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gptConfig.batch_size = args.batch_size
    gptConfig.block_size = args.block_size
    gptConfig.max_iters = args.max_iters
    gptConfig.eval_interval = args.eval_interval
    gptConfig.learning_rate = args.learning_rate
    gptConfig.eval_iters = args.eval_iters
    gptConfig.n_embd = args.n_embd
    gptConfig.n_head = args.n_head
    gptConfig.n_layer = args.n_layer
    gptConfig.dropout = args.dropout
    logging.info(gptConfig)
    return gptConfig

def parse_arguments() -> argparse.Namespace:
    """
    validate command line arguments and parse them
    :return: namespace with arguments
    """
    parser = argparse.ArgumentParser("python3 gpt_run.py")
    parser.add_argument("dataPath", nargs='?', help=" param desc", type=str, default='data/DarkKnight.txt')
    parser.add_argument("batch_size", nargs='?', help=" param desc", type=int, default=16) # 64 # how many independent sequences will we process in parallel?
    parser.add_argument("block_size", nargs='?', help=" param desc", type=int, default=32)  # 256 # what is the maximum context length for predictions?    
    parser.add_argument("n_embd", nargs='?', help=" param desc", type=int, default=64) # 384
    parser.add_argument("n_head", nargs='?', help=" param desc", type=int, default=4) # 6
    parser.add_argument("n_layer", nargs='?', help=" param desc", type=int, default=4) # 6
    parser.add_argument("dropout", nargs='?', help=" param desc", type=int, default=0) # 0.2
    parser.add_argument("max_iters", nargs='?', help=" param desc", type=int, default=5000)
    parser.add_argument("eval_iters", nargs='?', help=" param desc", type=int, default=200)
    parser.add_argument("eval_interval", nargs='?', help=" param desc", type=int, default=100) # 500
    parser.add_argument("learning_rate", nargs='?', help=" param desc", type=int, default=1e-3) # 3e-4
    parser.add_argument("max_new_tokens", nargs='?', help=" param desc", type=int, default=500) # 10000
    namespace = parser.parse_args()
    return namespace, setupGPTConfig(namespace)


if __name__ == '__main__':
    main()