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
    # torch.manual_seed(1337)

    if args.train == 'true':
        # load data
        logging.info(f'loading data from {args.dataPath} ...')
        dataLoader = DataLoader(dataPath=args.dataPath, gptConfig=gptConfig)
        dataLoader.loadData()

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(dataLoader.data)))
        gptConfig.vocab_size = len(chars)
        logging.info(f'vocabulary size is {gptConfig.vocab_size}')
        logging.debug(f'voabulary - {chars}')


        logging.info(f'creating the model')
        model = SimpleGPT(gptConfig)
        model.createMappings(chars)    

        gpt = model.to(gptConfig.device)
        # print the number of parameters in the model
        logging.info(f'{sum(p.numel() for p in gpt.parameters())/1e6} million parameters')

        # slit training and validation data
        logging.info(f'preparing data for training ...')
        dataLoader.splitData(model)

        # training
        gptTrainer = SimpleGPTTrainer(model, gptConfig, dataLoader)
        logging.info(f'training the model ...')
        gptTrainer.train()
        logging.info(f'training done!')

        # save model
        logging.info(f'saving the model for future use')
        torch.save(model, 'models/gpt.model')
        logging.info(f'saved at models/gpt.model')
    
    else:
        logging.info(f'loading model from previously trained save at models/gpt.model')
        model = torch.load('models/gpt.model')
        model.eval()
        gpt = model.to(gptConfig.device)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=gptConfig.device)
    logging.info(f'sampling the model')
    print(model.decode(gpt.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()))
    #open('out.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


def setupGPTConfig(args):
    # configuration of gpt hyperparameters
    gptConfig = SimpleGPTConfig()
    if args.device is None:
        gptConfig.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        gptConfig.device = args.device
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
    return gptConfig

def parse_arguments() -> argparse.Namespace:
    """
    validate command line arguments and parse them
    :return: namespace with arguments
    """
    parser = argparse.ArgumentParser("python3 gpt_run.py")
    parser.add_argument('-t', "--train", required=False, help=" param desc", type=str, default='true')
    parser.add_argument('-d', "--dataPath", required=False, help=" param desc", type=str, default='data/input.txt')
    parser.add_argument("--device", required=False, help=" param desc", type=str, default=None)
    parser.add_argument("--batch_size", required=False, help=" param desc", type=int, default=16) # 64 # how many independent sequences will we process in parallel?
    parser.add_argument("--block_size", required=False, help=" param desc", type=int, default=32)  # 256 # what is the maximum context length for predictions?    
    parser.add_argument("--n_embd", required=False, help=" param desc", type=int, default=128) # 384
    parser.add_argument("--n_head", required=False, help=" param desc", type=int, default=6) # 6
    parser.add_argument("--n_layer", required=False, help=" param desc", type=int, default=6) # 6
    parser.add_argument("--dropout", required=False, help=" param desc", type=float, default=0.2) # 0.2
    parser.add_argument('-m', "--max_iters", required=False, help=" param desc", type=int, default=5000)
    parser.add_argument("--eval_iters", required=False, help=" param desc", type=int, default=200)
    parser.add_argument("--eval_interval", required=False, help=" param desc", type=int, default=500) # 500
    parser.add_argument('-l', "--learning_rate", required=False, help=" param desc", type=float, default=8e-5) # 3e-4
    parser.add_argument('-o', "--max_new_tokens", required=False, help=" param desc", type=int, default=1000) # 10000
    namespace = parser.parse_args()
    logging.info(namespace)
    return namespace, setupGPTConfig(namespace)


if __name__ == '__main__':
    main()