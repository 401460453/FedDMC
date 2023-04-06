#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverFedProx import FedProx
from FLAlgorithms.servers.serverFedDistill import FedDistill
from FLAlgorithms.servers.serverpFedGen import FedGen
from FLAlgorithms.servers.serverFedDMC import FedDMC
from FLAlgorithms.servers.serverpFedEnsemble import FedEnsemble
from utils.model_utils import create_model
from utils.plot_utils import *
import torch
import torch.nn.functional as F
import pickle
from multiprocessing import Pool


def create_server_n_user(args, env):
    model = create_model(args.model, args.dataset, args.algorithm)
    if ('FedAvg' in args.algorithm):
        server=FedAvg(args, model, env)
    elif 'FedGen' in args.algorithm:
        server=FedGen(args, model, env)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, env)
    elif ('FedDistill' in args.algorithm):
        server = FedDistill(args, model, env)
    elif ('FedEnsemble' in args.algorithm):
        server = FedEnsemble(args, model, env)
    elif ('FedDMC' in args.algorithm):
        server = FedDMC(args, model, env)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server


def es_initial(args, env):
    torch.manual_seed(env)
    print("\n\n         [ Start initializing server {} ]           \n\n".format(env))
    # Generate model
    server = create_server_n_user(args, env)
    server.pre_train(args, args.pre_glob_iters)
    server.test()
    return server

# def feature_distill(old_model, new_model):
#     old_model.model.eval()
#     new_model.model.eval()
#     # W_L = torch.rand(32)
#     w_l = WFD()
#     optimizer = torch.optim.Adam(w_l.parameters(), lr=0.035)
#     criterion = torch.nn.MSELoss()
#     max = 0
#     index = 0
#     cs = 0
#     for epoch in range(1500):
#         optimizer.zero_grad()
#         samples = new_model.selected_users[0].get_next_train_batch(count_labels=False)
#         X, y = samples['X'], samples['y']
#         fea_by_old = old_model.model.get_feature(X)
#         fea_by_new = new_model.model.get_feature(X)
#         output = w_l(fea_by_new)
#         # output = torch.matmul(W_L, fea_by_new)
#         # dW_L = 0
#         # loss = 0
#         # for i in range(X.shape[0]):
#         #     dW_L = 2 * (W_L * fea_by_new[i] - fea_by_old[i]) * fea_by_new[i]
#         #     loss = (W_L * fea_by_new[i] - fea_by_old[i]).sum()
#         #     W_L -= lr * dW_L
#         # dW_L = dW_L / X.shape[0]
#         # loss = (loss / X.shape[0]).sum()
#         loss = criterion(fea_by_old, output)
#         loss.backward()
#         optimizer.step()
#         cos_sim_bef = F.cosine_similarity(fea_by_old[0], fea_by_new[0], dim=0)
#         cos_sim_af = F.cosine_similarity(fea_by_old[0], output[0], dim=0)
#         cs += cos_sim_af
#         print("before:{}, after:{}".format(cos_sim_bef.data, cos_sim_af.data))
#         if cos_sim_af > max:
#             max = cos_sim_af
#             index = epoch
#         # print(loss)
#         # print("round{}, loss = {}".format(epoch, loss))
#         # W_L -= lr * dW_L
#     print(max, index, cs/1500)
#     print("Done")


def main(args):
    # # initial 3 edge server
    es_list = []
    for env in range(args.num_envs):
        es = es_initial(args, env)
        es_list.append(es)
    print("Finished initializing.")
    # dynamic client from environment 0 transfer to environment 1
    # feature_distill(es_list[0], es_list[1])
    es_list[1].dynamic_users.append(es_list[0].selected_users[0])
    es_list[1].train(args)
    print("Finished training.")
    # evaluate model
    print("----------performance in new environment----------")
    es_list[1].send_parameters(mode=es_list[1].mode)
    es_list[1].evaluate(ke=es_list[1].W_1to0)
    print("----------performance in old environment----------")
    correct_samples, loss, total_samples = es_list[1].dynamic_users[0].test(ke=es_list[1].W_1to0)
    print("Accuracy = {}, Loss = {}".format(correct_samples / total_samples, loss))
    es_list[1].dynamic_users[0].set_parameters(es_list[1].model)
    correct_samples, loss, total_samples = es_list[1].dynamic_users[0].test(ke=es_list[1].W_1to0)
    print("Accuracy = {}, Loss = {}".format(correct_samples / total_samples, loss))
    print("Finished testing.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="pFedMe")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=180)
    parser.add_argument("--pre_glob_iters", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--KE_epochs", type=int, default=20, help="Epochs of Knowledge Distillation")
    parser.add_argument("--num_users", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--num_envs", type=int, default=2, help="number of edge server")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")
    parser.add_argument("--ewc_lamb", type=int, default=0.5, help="lamb in EWC algorithm")

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("=" * 80)
    main(args)
