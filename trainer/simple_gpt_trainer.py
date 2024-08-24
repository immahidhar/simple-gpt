import torch
import logging

class SimpleGPTTrainer():

    def __init__(self, model, gptConfig, dataLoader):
        self.model = model
        self.gptConfig = gptConfig
        self.dataLoader = dataLoader
        self.logger = logging.getLogger(SimpleGPTTrainer.__name__)

    def train(self):
        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.gptConfig.learning_rate)

        for iter in range(self.gptConfig.max_iters):

            # every once in a while evaluate the loss on train and val sets
            if iter % self.gptConfig.eval_interval == 0 or iter == self.gptConfig.max_iters - 1:
                losses = self.estimate_loss()
                self.logger.debug(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

            # sample a batch of data
            xb, yb = self.dataLoader.get_batch('train')

            # evaluate the loss
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.gptConfig.eval_iters)
            for k in range(self.gptConfig.eval_iters):
                X, Y = self.dataLoader.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out