import os
import dgl
import time
import torch
import torch.nn.functional as F
import logging

from myModel import MTHG
from arguments import args
from datetime import datetime
from utils.pytorchtools import EarlyStopping
from utils.dataset import LLMDataset_node, LLMDataset_eval, LLMDataset_t, LLMDataset_edge
from utils.util import *
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


if __name__ == '__main__':
    # Assumes a saved base model as input and model name to get the right directory.
    output_dir = 'result/{}/model_output/'.format(args.dataset)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Set paths of sub-directories.
    LOG_DIR = output_dir + args.log_dir
    SAVE_DIR = output_dir + args.save_dir
    CSV_DIR = output_dir + args.csv_dir
    MODEL_DIR = output_dir + args.model_dir

    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if not os.path.isdir(CSV_DIR):
        os.mkdir(CSV_DIR)

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    today = datetime.today()

    # Setup logging
    log_file = LOG_DIR + '/%s_%s_%s_%s.log' % (args.dataset, str(today.year),
                                                  str(today.month), str(today.day))

    log_level = logging.INFO
    logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    logging.info(args)

    # Create file name for result log csv from certain flag parameters.
    output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (args.dataset, str(today.year),
                                                  str(today.month), str(today.day))

    # model_dir is not used in this code for saving.

    # utils folder: utils.py, random_walk.py, minibatch.py
    # models folder: layers.py, models.py
    # main folder: train.py
    # eval folder: link_prediction.py

    """
    #1: Train logging format: Create a new log directory for each run (if log_dir is provided as input). 
    Inside it,  a file named <>.log will be created for each time step. The default name of the directory is "log" and the 
    contents of the <>.log will get appended per day => one log file per day.

    #2: Model save format: The model is saved inside model_dir. 

    #3: Output save format: Create a new output directory for each run (if save_dir name is provided) with embeddings at 
    each 
    time step. By default, a directory named000 "output" is created.

    #4: Result logging format: A csv file will be created at csv_dir and the contents of the file will get over-written 
    as per each day => new log file for each day.
    """

    # Load graphs and features.
    device = args.device
    graph = dgl.load_graphs(args.graph_path)[0][0]
    graphs = dgl.load_graphs(args.graphs_path)[0]
    feature_dict = graph.ndata['f'] # keys(['ipc', 'paper', 'patent'])
    if args.feature_masks[0] == 0:
        feature_dict['ipc'] = torch.randn_like(feature_dict['ipc'], dtype=torch.float32)
    if args.feature_masks[1] == 0:
        feature_dict['patent'] = torch.randn_like(feature_dict['patent'], dtype=torch.float32)
    if args.feature_masks[2] == 0:
        feature_dict['paper'] = torch.randn_like(feature_dict['paper'], dtype=torch.float32)
    del graph

    train_graph = get_train_graph() #2010-2022
    test_graph = get_test_graph() #2023
    train_adj = train_graph.adjacency_matrix(scipy_fmt='csr')
    test_adj = test_graph.adjacency_matrix(scipy_fmt='csr')
    del test_graph
    val_edges, val_edges_false, test_edges, test_edges_false = get_evaluation_data(train_adj, test_adj)

    dataset = LLMDataset_node(graph_path=args.graph_path, args=args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    valid_dataset = LLMDataset_eval(val_edges, val_edges_false)
    val_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    test_dataset = LLMDataset_eval(test_edges, test_edges_false)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # model
    model = MTHG(feature_dict, args).to(device)
    opt = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    early_stop = EarlyStopping(patience=args.patience, verbose=True, save_path=MODEL_DIR + '/checkpoint.pt')
    epochs_val_result = []
    state_dicts = []

    nodes = train_graph.nodes(ntype='ipc')
    # metapaths = get_metapaths_till_t(nodes, timestamp=args.max_timestamp)
    # metapath_graphs = get_metapath_graphs_till_t(metapaths, timestamp=args.max_timestamp)

    for epoch in range(args.num_epochs):
        print('Epoch: %04d' % (epoch + 1))
        # training loop
        model.train()
        mode = 'train'
        epoch_time = 0.0
        epoch_loss = 0.0
        it = 0
        for batch in dataloader:
            t_train = time.time()
            batch_loss = []
            all_nodes, edges_dict = get_node_pairs(graphs, batch)
            nodes2id = dict(zip(all_nodes.cpu().numpy().tolist(), list(range(all_nodes.shape[0]))))
            metapaths = get_metapaths_till_t(all_nodes, timestamp=args.max_timestamp)
            metapath_graphs = get_metapath_graphs_till_t(metapaths, timestamp=args.max_timestamp)
            embd = model([all_nodes, metapaths, metapath_graphs, args.max_timestamp, mode])

            for tidx, t in enumerate(args.time_range[1:],start=1):
            #for tidx, t in enumerate(args.time_range):
                pu, pv = edges_dict[t]['pos']
                nu, nv = edges_dict[t]['neg']

                pu = list(map(lambda x: nodes2id[x], pu.cpu().numpy().tolist()))
                pv = list(map(lambda x: nodes2id[x], pv.cpu().numpy().tolist()))
                nu = list(map(lambda x: nodes2id[x], nu.cpu().numpy().tolist()))
                nv = list(map(lambda x: nodes2id[x], nv.cpu().numpy().tolist()))

                embd_pu = embd[pu]
                embd_pv = embd[pv]
                embd_nu = embd[nu]
                embd_nv = embd[nv]

                embd_pu = embd_pu[:, tidx-1, :].view(-1, 1, embd_pu.shape[-1])
                embd_pv = embd_pv[:, tidx-1, :].view(-1, embd_pv.shape[-1], 1)
                embd_nu = embd_nu[:, tidx-1, :].view(-1, 1, embd_nu.shape[-1])
                embd_nv = embd_nv[:, tidx-1, :].view(-1, embd_nv.shape[-1], 1)

                pos_out = torch.bmm(embd_pu, embd_pv).flatten()
                neg_out = -torch.bmm(embd_nu, embd_nv).flatten()

                loss_t = -torch.sum(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
                batch_loss.append(torch.unsqueeze(loss_t, dim=0))

            batch_loss = torch.cat(batch_loss, dim=0)
            batch_loss = torch.sum(batch_loss, dim=0)
            epoch_time += time.time() - t_train


            print("Mini batch Iter: {} train_loss= {:.5f}".format(it, batch_loss))
            logging.info("Mini batch Iter: {} train_loss= {:.5f}".format(it, batch_loss))

            # autograd
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            epoch_loss += batch_loss
            it += 1

        epoch_loss /= it
        print("Time for epoch ", epoch_time)
        logging.info("Time for epoch : {}".format(epoch_time))
        print("Mean Loss at epoch {} : {}".format(epoch, epoch_loss))

        # validation
        model.eval()
        mode = 'eval'
        val_pred_pos = []
        val_pred_neg = []
        with torch.no_grad():
            for batch in val_dataloader:
                pu, pv, nu, nv = batch

                metapaths_till_t_pu = get_metapaths_till_t(pu,
                                                           timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
                metapath_graphs_till_t_pu = get_metapath_graphs_till_t(metapaths_till_t_pu,
                                                                       timestamp=args.test_timestamp)  # [t, num_mtypes]

                metapaths_till_t_pv = get_metapaths_till_t(pv,
                                                           timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
                metapath_graphs_till_t_pv = get_metapath_graphs_till_t(metapaths_till_t_pv,
                                                                       timestamp=args.test_timestamp)  # [t, num_mtypes]

                metapaths_till_t_nu = get_metapaths_till_t(nu,
                                                           timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
                metapath_graphs_till_t_nu = get_metapath_graphs_till_t(metapaths_till_t_nu,
                                                                       timestamp=args.test_timestamp)  # [t, num_mtypes]

                metapaths_till_t_nv = get_metapaths_till_t(nv,
                                                           timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
                metapath_graphs_till_t_nv = get_metapath_graphs_till_t(metapaths_till_t_nv,
                                                                       timestamp=args.test_timestamp)  # [t, num_mtypes]

                embd_pu = model([pu, metapaths_till_t_pu, metapath_graphs_till_t_pu, args.test_timestamp, mode])
                embd_pv = model([pv, metapaths_till_t_pv, metapath_graphs_till_t_pv, args.test_timestamp, mode])
                embd_nu = model([nu, metapaths_till_t_nu, metapath_graphs_till_t_nu, args.test_timestamp, mode])
                embd_nv = model([nv, metapaths_till_t_nv, metapath_graphs_till_t_nv, args.test_timestamp, mode])

                embd_pu = embd_pu.view(-1, 1, embd_pu.shape[-1])
                embd_pv = embd_pv.view(-1, embd_pv.shape[-1], 1)
                embd_nu = embd_nu.view(-1, 1, embd_nu.shape[-1])
                embd_nv = embd_nv.view(-1, embd_nv.shape[-1], 1)

                pos_out = torch.bmm(embd_pu, embd_pv).flatten()
                neg_out = torch.bmm(embd_nu, embd_nv).flatten()

                val_pred_pos.append(torch.sigmoid(pos_out))
                val_pred_neg.append(torch.sigmoid(neg_out))

        val_pred = torch.cat(val_pred_pos + val_pred_neg).cpu().numpy()
        val_true = np.array([1] * (val_pred.shape[0] // 2) + [0] * (val_pred.shape[0] // 2))
        auc_val = roc_auc_score(val_true, val_pred)
        epochs_val_result.append(auc_val)
        print("Epoch {}, Val AUC {}".format(epoch, auc_val))
        logging.info("Val results at epoch {}: AUC: {}".format(epoch, auc_val))
        early_stop(-auc_val, model)
        if early_stop.early_stop:
            print('Early stopping!')
            # Choose best model by validation set performance.
            best_epoch = epochs_val_result.index(max(epochs_val_result))
            print("Best epoch ", best_epoch)
            logging.info("Best epoch {}".format(best_epoch))
            break

    # test
    model.load_state_dict(torch.load(MODEL_DIR + '/checkpoint.pt'))
    model.eval()
    mode = 'eval'
    test_pred_pos = []
    test_pred_neg = []
    with torch.no_grad():
        for batch in test_dataloader:
            pu, pv, nu, nv = batch

            metapaths_till_t_pu = get_metapaths_till_t(pu,
                                                       timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
            metapath_graphs_till_t_pu = get_metapath_graphs_till_t(metapaths_till_t_pu,
                                                                   timestamp=args.test_timestamp)  # [t, num_mtypes]

            metapaths_till_t_pv = get_metapaths_till_t(pv,
                                                       timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
            metapath_graphs_till_t_pv = get_metapath_graphs_till_t(metapaths_till_t_pv,
                                                                   timestamp=args.test_timestamp)  # [t, num_mtypes]

            metapaths_till_t_nu = get_metapaths_till_t(nu,
                                                       timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
            metapath_graphs_till_t_nu = get_metapath_graphs_till_t(metapaths_till_t_nu,
                                                                   timestamp=args.test_timestamp)  # [t, num_mtypes]

            metapaths_till_t_nv = get_metapaths_till_t(nv,
                                                       timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
            metapath_graphs_till_t_nv = get_metapath_graphs_till_t(metapaths_till_t_nv,
                                                                   timestamp=args.test_timestamp)  # [t, num_mtypes]

            embd_pu = model([pu, metapaths_till_t_pu, metapath_graphs_till_t_pu, args.test_timestamp, mode])
            embd_pv = model([pv, metapaths_till_t_pv, metapath_graphs_till_t_pv, args.test_timestamp, mode])
            embd_nu = model([nu, metapaths_till_t_nu, metapath_graphs_till_t_nu, args.test_timestamp, mode])
            embd_nv = model([nv, metapaths_till_t_nv, metapath_graphs_till_t_nv, args.test_timestamp, mode])

            embd_pu = embd_pu.view(-1, 1, embd_pu.shape[-1])
            embd_pv = embd_pv.view(-1, embd_pv.shape[-1], 1)
            embd_nu = embd_nu.view(-1, 1, embd_nu.shape[-1])
            embd_nv = embd_nv.view(-1, embd_nv.shape[-1], 1)

            pos_out = torch.bmm(embd_pu, embd_pv).flatten()
            neg_out = torch.bmm(embd_nu, embd_nv).flatten()

            test_pred_pos.append(torch.sigmoid(pos_out))
            test_pred_neg.append(torch.sigmoid(neg_out))

    test_pred = torch.cat(test_pred_pos + test_pred_neg).cpu().numpy()
    test_true = np.array([1] * (test_pred.shape[0] // 2) + [0] * (test_pred.shape[0] // 2))
    auc_test = roc_auc_score(test_true, test_pred)
    ap_test = average_precision_score(test_true, test_pred)
    f1_test = f1_score(test_true, [1 if i >= 0.5 else 0 for i in test_pred])

    print("Best epoch test result auc={} ap={} f1={}\n".format(auc_test, ap_test, f1_test))
    logging.info("Best epoch test result auc={} ap={} f1={}\n".format(auc_test, ap_test, f1_test))
    write_to_csv(test_result=[auc_test, ap_test, f1_test], output_name=output_file, dataset=args.dataset)