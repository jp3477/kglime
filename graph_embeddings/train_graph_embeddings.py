# Python imports
import os

os.environ['DGLBACKEND'] = 'pytorch'
import argparse
from pathlib import Path
import logging
from contextlib import nullcontext
import json
from tqdm import tqdm
import configparser

# Third-party imports
import torch
import numpy as np
from torchmetrics.functional import retrieval_reciprocal_rank
from torchmetrics import RetrievalMRR
import dgl
import networkx as nx
from sklearn.model_selection import train_test_split

# Package imports
from .gcn import RGCNModel

RANDOM_SEED = 1

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def print_first_proc(*values, proc_id=0):
    if proc_id == 0:
        print(*values, )


def train(model,
          proc_id,
          train_dataloader,
          val_dataloader,
          loss_fn,
          optimizer,
          negative_samples,
          epochs=5,
          early_stopping=False,
          patience=5):
    def _get_triples(positive_graph, negative_graph):
        pos_srcs = positive_graph.edges()[0]
        pos_dests = positive_graph.edges()[1]

        pos_triples = torch.stack(
            [pos_srcs, positive_graph.edata['rel_type'], pos_dests], dim=-1)

        neg_srcs = negative_graph.edges()[0]
        neg_dests = negative_graph.edges()[1]
        neg_triples = torch.reshape(
            torch.stack([
                neg_srcs,
                torch.repeat_interleave(positive_graph.edata['rel_type'],
                                        negative_samples), neg_dests
            ],
                        dim=-1),
            (pos_triples.shape[0], -1, pos_triples.shape[1]))

        x = torch.cat([torch.unsqueeze(pos_triples, 1), neg_triples], dim=1)
        y = torch.cat([
            torch.ones((x.shape[0], 1)),
            torch.zeros((x.shape[0], neg_triples.shape[1]))
        ],
                      dim=-1)

        return x, y

    history = {
        "loss": [],
        "mrr": [],
    }
    best_model_state = model.state_dict()
    best_mrr = 0
    epochs_without_improvement = 0
    for epoch in range(epochs):
        print_first_proc(f'Epoch {epoch+1}/{epochs}', proc_id=proc_id)
        model.train()

        batch_losses = []
        batch_mrrs = []

        with tqdm(train_dataloader) if proc_id == 0 else nullcontext() as td:
            td = td if proc_id == 0 else train_dataloader
            for step, (input_nodes, positive_graph, negative_graph,
                       blocks) in enumerate(td):

                x, y = _get_triples(positive_graph, negative_graph)

                scores = model((x, blocks))

                loss = loss_fn(scores)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 5 == 0:
                    sub_mrrs = torch.zeros(scores.shape[0])
                    for i in range(scores.shape[0]):
                        sub_mrr = retrieval_reciprocal_rank(scores[i], y[i])
                        sub_mrrs[i] = sub_mrr

                    mrr = torch.mean(sub_mrrs)

                    batch_losses.append(loss.item())
                    batch_mrrs.append(mrr.item())

                    if proc_id == 0:
                        td.set_postfix(
                            {
                                "loss": "%.03f" % np.mean(batch_losses),
                                "mrr": "%.03f" % np.mean(batch_mrrs),
                            },
                            refresh=False)

        model.eval()

        val_mrr_metric = RetrievalMRR('error')
        val_batch_losses = []
        val_batch_mrrs = []
        val_batch_imrrs = []

        with tqdm(val_dataloader) if proc_id == 0 else nullcontext(
        ) as vd, torch.no_grad():
            vd = vd if proc_id == 0 else val_dataloader
            for input_nodes, positive_graph, negative_graph, blocks in vd:
                x, y = _get_triples(positive_graph, negative_graph)
                scores = model((x, blocks))

                loss = loss_fn(scores)

                sub_mrrs = torch.zeros(scores.shape[0])
                for i in range(scores.shape[0]):
                    sub_mrr = retrieval_reciprocal_rank(scores[i], y[i])
                    sub_mrrs[i] = sub_mrr

                mrr = torch.mean(sub_mrrs)

                val_batch_losses.append(loss.item())
                val_batch_mrrs.append(mrr.item())

            val_loss = np.mean(val_batch_losses)
            val_mrr = torch.tensor(np.mean(val_batch_mrrs),
                                   device=model.device)

            print_first_proc(
                {
                    "val_loss": "%.03f" % val_loss,
                    "val_mrr": "%.03f" % val_mrr,
                },
                proc_id=proc_id)

            history["loss"].append(val_loss)
            history["mrr"].append(val_mrr)

            torch.distributed.all_reduce(val_mrr)
            val_mrr /= torch.distributed.get_world_size()

            if val_mrr > best_mrr:
                best_mrr = val_mrr
                best_model_state = model.state_dict()
                epochs_without_improvement = 0

                if (epoch + 1) % 2 == 0 and proc_id == 0:
                    print("Checkpointing...")
                    save_weights(model.module, model.module.G,
                                 model.module.n_rels, Path('./pt_output/'))
            else:
                epochs_without_improvement += 1

            if early_stopping and epochs_without_improvement == patience:
                print(f"Early stopping proc_id {proc_id} at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)

    return history, best_mrr


def create_model(device,
                 G,
                 embedding_size=300,
                 basis=8,
                 n_layers=3,
                 dropout=0.0,
                 regularizer='basis'):
    n_rels = len(torch.unique(G.edata['rel_type']))
    model = RGCNModel(device,
                      G,
                      embedding_size,
                      embedding_size,
                      n_rels,
                      basis=basis,
                      n_layers=n_layers,
                      dropout=dropout,
                      regularizer=regularizer)

    return model, n_rels


def save_weights(model, G, n_rels, model_path):
    device = model.device
    G = G.to(device)
    node_embeddings = model.rgcn_block(G).detach().cpu().numpy()

    relation_embeddings = model.hake.rel_embedding(
        torch.arange(n_rels).to(device)).cpu().detach().numpy()

    np.save(model_path / 'node_embeddings.npy', node_embeddings)
    np.save(model_path / 'rel_embeddings.npy', relation_embeddings)

    lam = model.hake.lam.cpu().detach().numpy()
    lam2 = model.hake.lam2.cpu().detach().numpy()
    with open(model_path / 'lambda.json', 'w') as f:
        json.dump({'lambda': float(lam), 'lambda2': float(lam2)}, f)

    torch.save(model.state_dict(), model_path / 'saved_model_weights.pt')

    print(f"Node embeddings saved to {model_path / 'node_embeddings.npy'}")
    print(f"Relational matrices saved to {model_path / 'rel_embeddings.npy'}")
    print(
        f"Full model weights saved to {model_path / 'saved_model_weights.pb'}")


def train_graph_embeddings(proc_id,
                           devices,
                           nx_g_path,
                           model_output_path,
                           learning_rate=0.01,
                           batch_size=256,
                           epochs=10,
                           embedding_size=300,
                           n_layers=2,
                           negative_samples=128,
                           patience=5,
                           regularizer='basis',
                           basis=6,
                           dropout=0.0):
    # Initialize distributed training context.
    dev_id = devices[proc_id]
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12346')
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(backend='gloo',
                                             init_method=dist_init_method,
                                             world_size=len(devices),
                                             rank=proc_id)
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device('cuda:' + str(dev_id))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=dist_init_method,
                                             world_size=len(devices),
                                             rank=proc_id)

    # Load graph

    model_path = Path(model_output_path)

    if not model_path.exists():
        model_path.mkdir(exist_ok=True)

    print_first_proc("Converting networkx to dgl", proc_id=proc_id)
    nx_G = nx.read_gpickle(nx_g_path)

    G = dgl.from_networkx(nx_G,
                          node_attrs=['concept_id'],
                          edge_attrs=['id', 'rel_type']).to(device)

    print_first_proc("Calculating norms", proc_id=proc_id)

    edge_rel_counts = {}
    dests = G.edges()[1]
    rel_types = G.edata['rel_type']

    edges_range = tqdm(range(len(dests))) if proc_id == 0 else range(
        len(dests))
    for i in edges_range:
        dest = int(dests[i])
        rel_type = int(rel_types[i])
        if dest not in edge_rel_counts:
            edge_rel_counts[dest] = {}
            edge_rel_counts[dest][rel_type] = 1
        else:
            if rel_type in edge_rel_counts[dest]:
                edge_rel_counts[dest][rel_type] += 1
            else:
                edge_rel_counts[dest][rel_type] = 1

    norms = []
    for i in range(len(dests)):
        dest = int(dests[i])
        rel_type = int(rel_types[i])
        norms.append(1.0 / max(edge_rel_counts[dest][rel_type], 1))

    norms = torch.Tensor(norms).to(device)
    G.edata['norm'] = torch.unsqueeze(norms, 1)

    print_first_proc("Generating samples", proc_id=proc_id)
    eids = G.edata['id']
    train_eids, val_eids = train_test_split(eids,
                                            test_size=0.2,
                                            random_state=RANDOM_SEED)

    val_eids, test_eids = train_test_split(val_eids,
                                           test_size=0.5,
                                           random_state=RANDOM_SEED)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
    # negative_sampler = dgl.dataloading.negative_sampler.GlobalUniform(
    #     negative_samples, replace=True)
    negative_sampler = RandomCorruptionNegativeSampler(negative_samples)

    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=negative_sampler)
    train_dataloader = dgl.dataloading.DataLoader(G,
                                                  train_eids,
                                                  sampler,
                                                  device=device,
                                                  use_ddp=True,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=False,
                                                  num_workers=0)

    val_dataloader = dgl.dataloading.DataLoader(G,
                                                val_eids,
                                                sampler,
                                                device=device,
                                                use_ddp=False,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=0)

    print_first_proc("Defining model", proc_id=proc_id)
    model_args = (device, G)
    model_kwargs = {
        'embedding_size': embedding_size,
        'n_layers': n_layers,
        'basis': basis,
        'dropout': dropout,
        'regularizer': regularizer
    }
    model, n_rels = create_model(*model_args, **model_kwargs)
    # model.load_state_dict(torch.load('./pt_output/saved_model_weights.pt'))

    model = model.to(device)

    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=None,
                                                          output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[device],
                                                          output_device=device)

    loss_fn = AdverserialLoss(margin=5.0, temp=2.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print_first_proc("Training model", proc_id=proc_id)
    history, this_best_mrr = train(model,
                                   proc_id,
                                   train_dataloader,
                                   val_dataloader,
                                   loss_fn,
                                   optimizer,
                                   negative_samples,
                                   epochs=epochs,
                                   early_stopping=True,
                                   patience=patience)

    outputdata = {
        'best_mrr': this_best_mrr.item(),
        # 'model_weights': model.state_dict(),
        # 'model_args': model_args,
        # 'model_kwargs': model_kwargs
    }

    print(f'Proc {proc_id} finished')

    outputs = [None for _ in range(len(devices))]

    # Waits for all gpus to be finished
    torch.distributed.all_gather_object(outputs, outputdata)

    all_best_mrrs = [output['best_mrr'] for output in outputs]
    best_mrr = np.max(all_best_mrrs)
    best_mrr_index = np.argmax(all_best_mrrs)

    if proc_id == best_mrr_index:

        print("Restoring best model weights")
        print(f'Best MRR: {best_mrr}')

        best_model = model
        save_weights(best_model.module, G, n_rels, model_path)

    cleanup()


def cleanup():
    torch.distributed.destroy_process_group()


def train_graph_embeddings_mp(nx_G,
                              model_output_path,
                              learning_rate=0.01,
                              batch_size=256,
                              epochs=10,
                              embedding_size=300,
                              n_layers=2,
                              negative_samples=128,
                              patience=5,
                              num_gpus=2,
                              regularizer='basis',
                              basis=6,
                              dropout=0.0):
    import torch.multiprocessing as mp
    print(f"Using {num_gpus} gpus")
    try:
        mp.spawn(train_graph_embeddings,
                 args=(list(range(num_gpus)), nx_G, model_output_path,
                       learning_rate, batch_size, epochs, embedding_size,
                       n_layers, negative_samples, patience, regularizer,
                       basis, dropout),
                 nprocs=num_gpus)
    except KeyboardInterrupt:
        cleanup()


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()

    parser = argparse.ArgumentParser(
        description="Create embeddings for nodes in a networkx graph")
    parser.add_argument('nx_g_path', help='Pickled NetworkX graph')
    parser.add_argument('model_output_path',
                        help='Path for model output and weights')

    args = parser.parse_args()

    train_graph_embeddings_mp(args.nx_g_path,
                              args.model_output_path,
                              num_gpus=num_gpus,
                              embedding_size=300,
                              batch_size=1024,
                              learning_rate=0.01,
                              negative_samples=128,
                              n_layers=2,
                              epochs=300,
                              patience=10,
                              regularizer='basis')
