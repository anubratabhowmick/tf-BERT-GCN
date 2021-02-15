from GCN.code.gcn_train import paper_pretrain, coauthor_pretrain
import GCN.code.config as config
import os


if __name__ == '__main__':
    args = config.get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.paper_author == 'paper':
        paper_pretrain(args.dataset, args.gcn_model, args.gcn_epochs, args.gcn_lr, args.gcn_hidden1, args.gcn_hidden2,
                   args.save_dir)
    else:
        coauthor_pretrain(args.dataset, args.gcn_model, args.gcn_epochs, args.gcn_lr, args.gcn_hidden1, args.gcn_hidden2,
                   args.save_dir)
