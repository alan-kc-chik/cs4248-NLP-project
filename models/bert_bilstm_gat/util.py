from timeit import default_timer as timer
import numpy as np
from torch import nn
from tqdm import tqdm
from model import Classify
import torch
import torch.optim as optim
import matplotlib
import gc

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Utils:
    def __init__(self, params, dl):
        self.params = params
        self.data_loader = dl

    @staticmethod
    def to_tensor(arr):
        # list -> Tensor (on GPU if possible)
        if torch.cuda.is_available():
            tensor = torch.tensor(arr).type(torch.cuda.LongTensor)
        else:
            tensor = torch.tensor(arr).type(torch.LongTensor)
        return tensor

    def get_dev_loss_and_acc(self, model, loss_fn):
        losses = []
        hits = 0
        total = 0
        model.eval()
        for sents, lens, labels in self.data_loader.dev_data_loader:
            y_batch = self.to_tensor(labels)
            logits = self.get_gcn_logits(model, sents)
            loss = loss_fn(logits, y_batch)
            hits += torch.sum(torch.argmax(logits, dim=1) == y_batch).item()
            total += len(sents)
            losses.append(loss.item())

        return np.asscalar(np.mean(losses)), hits / total

    def get_gcn_logits(self, model, docs):
        logits = []
        for i, sents in enumerate(docs):
            logit = model(sents)
            logits.append(logit)
        return torch.stack(logits)

    def train(self, pretrained_emb, train_partition, tune_from, save_plots_as):

        model = Classify(self.params, ntags=self.data_loader.ntags)
        # model = nn.DataParallel(model)
        # model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model = model.cuda()

        if tune_from != '':
            model.load_state_dict(torch.load(tune_from, map_location=lambda storage, loc: storage))
            print(f'loaded state from {tune_from}')
        gc.collect()
        torch.cuda.empty_cache()

        for name, param in model.bert_model.named_parameters():
            param.requires_grad = False

        # optimizer = optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.params.lr, weight_decay=self.params.weight_decay)

        # Variables for plotting
        train_losses = []
        dev_losses = []
        train_accs = []
        dev_accs = []
        s_t = timer()
        prev_best = 0
        patience = 0

        # Start the training loop
        for epoch in range(1, self.params.max_epochs + 1):
            print(f'epoch = {epoch}')
            model.train()
            train_loss = 0
            hits = 0
            total = 0
            # i = 0
            for sents, lens, labels in tqdm(self.data_loader.train_data_loader):
                # gc.collect()
                torch.cuda.empty_cache()

                y_batch = self.to_tensor(labels)
                logits = self.get_gcn_logits(model, sents)
                loss = loss_fn(logits, y_batch)

                # Book keeping
                train_loss += float(loss.item())
                hits += float(torch.sum(torch.argmax(logits, dim=1) == y_batch).item())
                # One can alternatively do this accuracy computation on cpu by,
                # moving the logits to cpu: logits.data.cpu().numpy(), and then using numpy argmax.
                # However, we should always avoid moving tensors between devices if possible for faster computation
                total += float(len(sents))

                # Back-prop
                optimizer.zero_grad()  # Reset the gradients
                loss.backward()  # Back propagate the gradients
                optimizer.step()  # Update the network

                # if i % 100 == 0:
                #     del logits
                #     gc.collect()
                #     torch.cuda.empty_cache()

                # i = i + 1

            # Compute loss and acc for dev set
            dev_loss, dev_acc = self.get_dev_loss_and_acc(model, loss_fn)
            train_losses.append(train_loss / len(self.data_loader.train_data_loader))
            dev_losses.append(dev_loss)
            train_accs.append(hits / total)
            dev_accs.append(dev_acc)
            tqdm.write("Epoch: {}, Train loss: {}, Train acc: {}, Dev loss: {}, Dev acc: {}".format(
                epoch, train_loss, hits / total, dev_loss, dev_acc))
            if dev_acc < prev_best:
                patience += 1
                if patience == 3:
                    # Reduce the learning rate by a factor of 2 if dev acc doesn't increase for 3 epochs
                    # Learning rate annealing
                    optim_state = optimizer.state_dict()
                    optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2
                    optimizer.load_state_dict(optim_state)
                    tqdm.write('Dev accuracy did not increase, reducing the learning rate by 2 !!!')
                    patience = 0
            else:
                prev_best = dev_acc
                # Save the model
                torch.save(model.state_dict(), f"models/model_{save_plots_as}_partition_{train_partition}_epoch{epoch}.t7")

        # Acc vs time plot
        fig = plt.figure()
        plt.plot(range(1, self.params.max_epochs + 1), train_accs, color='b', label='train')
        plt.plot(range(1, self.params.max_epochs + 1), dev_accs, color='r', label='dev')
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.legend()
        plt.xticks(np.arange(1, self.params.max_epochs + 1, step=4))
        fig.savefig('accuracy/' + f'{save_plots_as}_partition_{train_partition}_accuracy.png')

        return timer() - s_t

    # def get_pre_trained_embeddings(self):
    #     print("Reading pre-trained embeddings...")
    #     embeddings = np.random.uniform(-0.25, 0.25, (len(self.data_loader.w2i), self.params.emb_dim))
    #     count = 0
    #     with open(self.params.pte, 'r', encoding='utf-8') as f:
    #         ignore_first_row = True
    #         for row in f.readlines():
    #             if ignore_first_row:
    #                 ignore_first_row = False
    #                 continue
    #             split_row = row.split(" ")
    #             vec = np.array(split_row[1:-1]).astype(np.float)
    #             if split_row[0] in self.data_loader.w2i and len(vec) == self.params.emb_dim:
    #                 embeddings[self.data_loader.w2i[split_row[0]]] = vec
    #                 count += 1
    #     print("Successfully loaded {} embeddings out of {}".format(count, len(self.data_loader.w2i)))
    #     return np.array(embeddings)
