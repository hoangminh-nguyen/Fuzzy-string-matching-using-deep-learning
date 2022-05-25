# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-

# """
# References:
# - Main: https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
# - https://github.com/hpanwar08/sentence-classification-pytorch/blob/master/Sentiment%20analysis%20pytorch.ipynb
# New version of the above implementation (not used here)
# - https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
# - https://github.com/hpanwar08/sentence-classification-pytorch/blob/master/Sentiment%20analysis%20pytorch%201.0.ipynb
# Others:
# - https://blog.floydhub.com/gru-with-pytorch/
# - https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
# - https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_recurrent_neuralnetwork/
# - https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch
# - https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
# """

# from audioop import bias
# from calendar import c
# import copy
# from datetime import datetime
# import glob
# from operator import attrgetter
# from turtle import forward, position
# from matplotlib.pyplot import ylabel
# import numpy as np
# import os
# import pickle
# import shutil
# import sys
# import time
# #from tqdm import tqdm, tnrange
# from tqdm.autonotebook import tqdm
# from tqdm.notebook import tnrange

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.autograd import Variable

# from .data_processing import test_tokenize
# from .utils import cprint, bc, log_message
# from .utils import print_stats
# from .utils import torch_summarize
# from .utils import create_parent_dir
# from .utils import eval_map
# # --- set seed for reproducibility
# from .utils import set_seed_everywhere
# set_seed_everywhere(1364)

# # skip future warnings for now XXX
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# # ------------------- transformer_network --------------------
# def transformer_network(dl_inputs, model_name, train_dc, valid_dc=False, test_dc=False):
#     """
#     Main function for training and evaluation of GRU/LSTM network for matching
#     """
#     start_time = time.time()

#     print("\n\n")
#     cprint('[INFO]', bc.magenta,
#            '******************************'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))
#     cprint('[INFO]', bc.magenta,
#            '**** (Bi-directional) {} ****'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))
#     cprint('[INFO]', bc.magenta,
#            '******************************'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))

#     # --- read inputs
#     cprint('[INFO]', bc.dgreen, 'read inputs')
#     main_architecture = dl_inputs['gru_lstm']['main_architecture']
#     vocab_size = len(train_dc.vocab)
#     embedding_dim = dl_inputs['gru_lstm']['embedding_dim']
#     rnn_hidden_dim = dl_inputs['gru_lstm']['rnn_hidden_dim']
#     output_dim = dl_inputs['gru_lstm']['output_dim']
#     batch_size = dl_inputs['gru_lstm']['batch_size']
#     epochs = dl_inputs['gru_lstm']['epochs']
#     learning_rate = dl_inputs['gru_lstm']['learning_rate']
#     rnn_n_layers = dl_inputs['gru_lstm']['num_layers']
#     bidirectional = dl_inputs['gru_lstm']['bidirectional']
#     try:
#         rnn_drop_prob = dl_inputs['gru_lstm']['rnn_dropout']
#     except:
#         cprint('[WARNING]', bc.dred, 'DEPRECATED (gru_dropout): use rnn_dropout in the input file instead.')
#         rnn_drop_prob = dl_inputs['gru_lstm']['gru_dropout']
#     rnn_bias = dl_inputs['gru_lstm']['bias']
#     fc_dropout = dl_inputs['gru_lstm']['fc_dropout']
#     att_dropout = dl_inputs['gru_lstm']['att_dropout']
#     fc1_out_features = dl_inputs['gru_lstm']['fc1_out_dim']
#     pooling_mode = dl_inputs['gru_lstm']['pooling_mode']
#     dl_shuffle = dl_inputs['gru_lstm']['dl_shuffle']
#     map_flag = dl_inputs['inference']['eval_map_metric']
#     do_validation = dl_inputs["gru_lstm"]["validation"]
#     if do_validation in [-1]:
#         do_validation = 1
#     else:
#         do_validation = int(do_validation)
    
#     # --- create the model
#     model_transformer = two_parallel_transformer(main_architecture, vocab_size, embedding_dim, rnn_hidden_dim, output_dim,
#                                   rnn_n_layers, bidirectional, pooling_mode, rnn_drop_prob, rnn_bias,
#                                   fc1_out_features, fc_dropout, att_dropout)
#     model_transformer.to(dl_inputs['general']['device'])

#     # --- optimisation
#     #if dl_inputs['gru_lstm']['optimizer'].lower() in ['adam']:
#     opt = optim.Adam(model_transformer.parameters(), lr=learning_rate,  weight_decay=1e-5)
    

#     cprint('[INFO]', bc.lgreen, 'start fitting parameters')
#     train_dl = DataLoader(dataset=train_dc, batch_size=batch_size, shuffle=dl_shuffle)
#     valid_dl = DataLoader(dataset=valid_dc, batch_size=batch_size, shuffle=dl_shuffle)

#     if dl_inputs['gru_lstm']['create_tensor_board']:
#         tboard_path = os.path.join(dl_inputs["general"]["models_dir"], 
#                                    model_name, 
#                                    dl_inputs['gru_lstm']['create_tensor_board'])
#     else:
#         tboard_path = None

#     fit(model=model_transformer,
#         train_dl=train_dl, 
#         valid_dl=valid_dl,
#         loss_fn=nn.CrossEntropyLoss(reduction="mean"),  # The negative log likelihood loss
#         opt=opt,
#         epochs=epochs,
#         pooling_mode=pooling_mode,
#         device=dl_inputs['general']['device'], 
#         tboard_path=tboard_path,
#         model_path=os.path.join(dl_inputs["general"]["models_dir"], model_name),
#         csv_sep=dl_inputs['preprocessing']["csv_sep"],
#         map_flag=map_flag,
#         do_validation=do_validation,
#         early_stopping_patience=dl_inputs["gru_lstm"]["early_stopping_patience"],
#         model_name=model_name
#         )

#     # --- print some simple stats on the run
#     print_stats(start_time)

# # ------------------- fine_tuning --------------------
# def fine_tuning(pretrained_model_path, dl_inputs, model_name, 
#                 train_dc, valid_dc=False, test_dc=False):
#     """
#     Fine tuning function for further training a model on new data
#     """
#     batch_size = dl_inputs['gru_lstm']['batch_size']
#     dl_shuffle = dl_inputs['gru_lstm']['dl_shuffle']
#     device=dl_inputs['general']['device']
#     learning_rate = dl_inputs['gru_lstm']['learning_rate']
#     epochs = dl_inputs['gru_lstm']['epochs']
#     pooling_mode = dl_inputs['gru_lstm']['pooling_mode']
#     map_flag = dl_inputs['inference']['eval_map_metric']
#     do_validation = dl_inputs["gru_lstm"]["validation"]
#     if do_validation in [-1]:
#         do_validation = 1
#     else:
#         do_validation = int(do_validation)
    
#     pretrained_model = torch.load(pretrained_model_path, map_location=torch.device(device))
    
#     layers_to_freeze = dl_inputs['gru_lstm']['layers_to_freeze']
#     for one_layer in layers_to_freeze:
#         for name, param in pretrained_model.named_parameters():
#             if one_layer in name:
#                 param.requires_grad = False

#     print("\n")
#     print(20*"===")
#     print(f"List all parameters in the model")
#     print(20*"===")
#     for name, param in pretrained_model.named_parameters():
#         n = name.split(".")[0].split("_")[0]
#         print(name, param.requires_grad)
#     print(20*"===")
    
#     if dl_inputs['gru_lstm']['optimizer'].lower() in ['adam']:
#         opt = optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()), learning_rate)

#     start_time = time.time()

#     print("\n\n")
#     cprint('[INFO]', bc.magenta,
#            '******************************'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))
#     cprint('[INFO]', bc.magenta,
#            '**** (Bi-directional) {} ****'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))
#     cprint('[INFO]', bc.magenta,
#            '******************************'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))
    
#     train_dl = DataLoader(dataset=train_dc, batch_size=batch_size, shuffle=dl_shuffle)
#     valid_dl = DataLoader(dataset=valid_dc, batch_size=batch_size, shuffle=dl_shuffle)

#     if dl_inputs['gru_lstm']['create_tensor_board']:
#         tboard_path = os.path.join(dl_inputs["general"]["models_dir"], 
#                                    model_name, 
#                                    dl_inputs['gru_lstm']['create_tensor_board'])
#     else:
#         tboard_path = None

#     fit(model=pretrained_model,
#         train_dl=train_dl, 
#         valid_dl=valid_dl,
#         loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1, 1], 
#                                                         dtype=torch.float32, 
#                                                         device=dl_inputs['general']['device']), 
#                                     reduction="mean"),  # The negative log likelihood loss
#         opt=opt,
#         epochs=epochs,
#         pooling_mode=pooling_mode,
#         device=dl_inputs['general']['device'], 
#         tboard_path=tboard_path,
#         model_path=os.path.join(dl_inputs["general"]["models_dir"], model_name),
#         csv_sep=dl_inputs['preprocessing']["csv_sep"],
#         map_flag=map_flag,
#         do_validation=do_validation,
#         early_stopping_patience=dl_inputs["gru_lstm"]["early_stopping_patience"],
#         model_name=model_name
#         )

#     # --- print some simple stats on the run
#     print_stats(start_time)
   
# # ------------------- fit  --------------------
# def fit(model, train_dl, valid_dl, loss_fn, opt, epochs=3, 
#         pooling_mode='attention', device='cpu', 
#         tboard_path=False, model_path=False, csv_sep="\t", map_flag=False, do_validation=1,
#         early_stopping_patience=False, model_name="default"):

#     num_batch_train = len(train_dl)
#     num_batch_valid = len(valid_dl)
#     cprint('[INFO]', bc.dgreen, 'Number of batches: {}'.format(num_batch_train))
#     cprint('[INFO]', bc.dgreen, 'Number of epochs: {}'.format(epochs))
    
#     if tboard_path:
#         try:
#             from torch.utils.tensorboard import SummaryWriter       
#             tboard_writer = SummaryWriter(tboard_path) 
#         except ImportError:
#             cprint('[WARNING]', bc.dred, 'SummaryWriter could not be imported! Continue without creating a tensorboard.')
#             tboard_writer = False
#     else:
#         tboard_writer = False

#     # if do_validation is not -1, 
#     # perform validation at least once
#     if do_validation in [0]:
#         do_validation = epochs + 2

#     print_summary = True
#     wtrain_counter = 0
#     wvalid_counter = 0

#     # initialize early stopping parameters
#     es_loss = False
#     es_stop = False
#     cprint('[INFO]', bc.orange, str(list(model.parameters())))
#     for epoch in tnrange(epochs):
#         if train_dl:
#             model.train()
#             y_true_train = list()
#             y_pred_train = list()
#             total_loss_train = 0
    

#             t_train = tqdm(iter(train_dl), leave=False, total=num_batch_train)
#             t_train.set_description('Epoch {}/{}'.format(epoch+1, epochs))
#             for x1, len1, x2, len2, y, train_indxs in t_train:
#                 # transpose x1 and x2
#                 x1 = x1.transpose(0, 1)
#                 x2 = x2.transpose(0, 1)

#                 x1 = Variable(x1.to(device))
#                 x2 = Variable(x2.to(device))
#                 y = Variable(y.to(device))
#                 len1 = len1.numpy()
#                 len2 = len2.numpy()

#                 # step 1. zero the gradients
#                 opt.zero_grad()
#                 # step 2. compute the output
#                 pred = model(x1, len1, x2, len2, pooling_mode=pooling_mode, device=device)

#                 if print_summary:
#                     # print info about the model only in the first epoch
#                     torch_summarize(model)
#                     print_summary = False
#                 # step 3. compute the loss
#                 loss = loss_fn(pred, y)
#                 # step 4. use loss to produce gradients
#                 loss.backward()
#                 # step 5. use optimizer to take gradient step
#                 opt.step()

#                 pred_softmax = F.softmax(pred, dim=-1)
#                 t_train.set_postfix(loss=loss.data)
#                 pred_idx = torch.max(pred_softmax, dim=1)[1]

#                 y_true_train += list(y.cpu().data.numpy())
#                 y_pred_train += list(pred_idx.cpu().data.numpy())
#                 total_loss_train += loss.data

#                 wtrain_counter += 1

#             train_acc = accuracy_score(y_true_train, y_pred_train)
#             train_pre = precision_score(y_true_train, y_pred_train)
#             train_rec = recall_score(y_true_train, y_pred_train)
#             train_macrof1 = f1_score(y_true_train, y_pred_train, average='macro')
#             train_weightedf1 = f1_score(y_true_train, y_pred_train, average='weighted')

#             train_loss = total_loss_train / len(train_dl)
#             epoch_log = '{} -- Epoch: {}/{}; Train; loss: {:.3f}; acc: {:.3f}; precision: {:.3f}, recall: {:.3f}, macrof1: {:.3f}, weightedf1: {:.3f}'.format(
#                     datetime.now().strftime("%m/%d/%Y_%H:%M:%S"), epoch+1, epochs, train_loss, train_acc, train_pre, train_rec, train_macrof1,train_weightedf1)
#             cprint('[INFO]', bc.orange, epoch_log)
#             if model_path:
#                 log_message(epoch_log + "\n", mode="a+", filename=os.path.join(model_path, "log.txt"))
#             else:
#                 log_message(epoch_log + "\n", mode="a+")

#             if tboard_writer:    
#                 # Record loss
#                 tboard_writer.add_scalar('Train/Loss', loss.item(), epoch)
#                 # Record accuracy
#                 tboard_writer.add_scalar('Train/Accuracy', train_acc, epoch)
#                 tboard_writer.flush()

#         if valid_dl and (((epoch+1) % do_validation) == 0):
#             valid_desc = 'Epoch: {}/{}; Valid'.format(epoch+1, epochs)
#             valid_loss = test_model(model, 
#                                     valid_dl, 
#                                     eval_mode="valid", 
#                                     valid_desc=valid_desc,
#                                     pooling_mode=pooling_mode, 
#                                     device=device,
#                                     model_path=model_path, 
#                                     tboard_writer=tboard_writer,
#                                     csv_sep=csv_sep,
#                                     epoch=epoch+1,
#                                     map_flag=map_flag,
#                                     output_loss=True)

#             if (not es_loss) or (valid_loss <= es_loss):
#                 es_loss = valid_loss
#                 es_model = copy.deepcopy(model)
#                 es_checkpoint = epoch + 1
#                 es_counter = 0
#             else:
#                 es_counter += 1
            
#             if early_stopping_patience:
#                 if es_counter >= early_stopping_patience:
#                     # --- save the model
#                     checkpoint_path = os.path.join(model_path, 
#                                                    model_name + '.model')
#                     if not os.path.isdir(os.path.dirname(checkpoint_path)):
#                         os.makedirs(os.path.dirname(checkpoint_path))
#                     cprint('[INFO]', bc.lgreen, 
#                            f'saving the model (early stopped) with least valid loss (checkpoint: {es_checkpoint}) at {checkpoint_path}')
#                     torch.save(es_model, checkpoint_path)
#                     torch.save(es_model.state_dict(), checkpoint_path + "_state_dict")
#                     es_stop = True

#         if model_path:
#             # --- save the model
#             cprint('[INFO]', bc.lgreen, 'saving the model')
#             checkpoint_path = os.path.join(model_path, f'checkpoint{epoch+1:05d}.model')
#             if not os.path.isdir(os.path.dirname(checkpoint_path)):
#                 os.makedirs(os.path.dirname(checkpoint_path))
#             torch.save(model, checkpoint_path)
#             torch.save(model.state_dict(), checkpoint_path + "_state_dict")
        
#         if es_stop:
#             cprint('[INFO]', bc.dgreen, 'Early stopping at epoch: {}, selected epoch: {}'.format(epoch+1, es_checkpoint))
#             return 
    
#     cprint('[INFO]', bc.orange, str(list(model.parameters())))
#     if model_path and epoch > 0:
#         # --- save the model with least validation loss
#         model_path_save = os.path.join(model_path,
#                                        model_name + '.model')
#         if not os.path.isdir(os.path.dirname(model_path_save)):
#             os.makedirs(os.path.dirname(model_path_save))
#         cprint(f'[INFO]', bc.lgreen, 
#                f'saving the model with least valid loss (checkpoint: {es_checkpoint}) at {model_path_save}')
#         torch.save(es_model, model_path_save)
#         torch.save(es_model.state_dict(), model_path_save + "_state_dict")
    


# # ------------------- test_model --------------------
# def test_model(model, test_dl, eval_mode='test', valid_desc=None,
#                pooling_mode='attention', device='cpu', evaluation=True,
#                output_state_vectors=False, output_preds=False, 
#                output_preds_file=False, model_path=False, tboard_writer=False,
#                csv_sep="\t", epoch=1, map_flag=False, print_epoch=True,
#                output_loss=False):

#     model.eval()

#     # print info about the model only in the first epoch
#     #torch_summarize(model)

#     y_true_test = list()
#     y_pred_test = list()
#     y_score_test = list()
#     map_queries = {}
#     test_line_id = 0
#     total_loss_test = 0

#     # XXX HARD CODED! Also in rnn_networks
#     loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1, 1], 
#                                                      dtype=torch.float32, 
#                                                      device=device), 
#                                 reduction="mean")
#     # In first dump of the results, we add a header to the output file
#     first_dump = True

#     wtest_counter = 0
#     t_test = tqdm(iter(test_dl), leave=False, total=len(test_dl))
#     if eval_mode == 'valid':
#         eval_desc = valid_desc
#     elif eval_mode == 'test':
#         eval_desc = 'Epoch: 0/0; Test'
    
#     t_test.set_description(eval_mode)
    
#     for x1, len1, x2, len2, y, indxs in t_test:
#         if output_state_vectors:
#             output_par_dir = os.path.abspath(os.path.join(output_state_vectors, os.pardir))
#             if not os.path.isdir(output_par_dir):
#                 os.makedirs(output_par_dir)
#             torch.save(indxs, f'{output_state_vectors}_indxs_{wtest_counter}')
#         wtest_counter += 1

#         # transpose x1 and x2
#         x1 = x1.transpose(0, 1)
#         x2 = x2.transpose(0, 1)

#         x1 = Variable(x1.to(device))
#         x2 = Variable(x2.to(device))
#         y = Variable(y.to(device))
#         len1 = len1.numpy()
#         len2 = len2.numpy()

#         with torch.no_grad():
#             pred = model(x1, len1, x2, len2, pooling_mode=pooling_mode, 
#                          device=device, output_state_vectors=output_state_vectors, 
#                          evaluation=evaluation)
#             if output_state_vectors:
#                 all_preds = []
#                 continue

#             loss = loss_fn(pred, y)

#             if eval_mode == 'valid':
#                 t_test.set_postfix(loss=loss.data)
                
#             pred_softmax = F.softmax(pred, dim=-1)
#             pred_idx = torch.max(pred_softmax, dim=1)[1]

#             if wtest_counter == 1:
#                 # Confidence for label 1
#                 all_preds = pred_softmax[:, 1]
#             else:
#                 all_preds = torch.cat([all_preds, pred_softmax[:, 1]])

#             y_true_test += list(y.cpu().data.numpy())
#             y_pred_test += list(pred_idx.cpu().data.numpy())
            
#             if map_flag:

#                 # pulling out the scores for the prediction of 1
#                 y_score_test += pred_softmax.cpu().data.numpy()[:, 1].tolist()

#                 for q in test_dl.dataset.df.loc[indxs]["s1"].to_numpy():

#                     if q in map_queries:
#                         map_queries[q].append(test_line_id)              

#                     else:
#                         map_queries[q] = [test_line_id]

#                     test_line_id +=1

#             if output_preds:
#                 pred_results = np.vstack([test_dl.dataset.df.loc[indxs]["s1"].to_numpy(), 
#                                           test_dl.dataset.df.loc[indxs]["s2"].to_numpy(), 
#                                           pred_idx.cpu().data.numpy().T, 
#                                           pred_softmax.T.cpu().data.numpy(), 
#                                           y.cpu().data.numpy().T])
#                 if output_preds_file:
#                     with open(output_preds_file, "a+") as pred_f:
#                         if first_dump:
#                             np.savetxt(pred_f, pred_results.T, 
#                                     fmt=('%s', '%s', '%d', '%.4f', '%.4f', '%d'), delimiter=csv_sep, 
#                                     header=f"s1{csv_sep}s2{csv_sep}prediction{csv_sep}p0{csv_sep}p1{csv_sep}label")
#                             first_dump = False
#                         else:
#                             np.savetxt(pred_f, pred_results.T, 
#                                     fmt=('%s', '%s', '%d', '%.4f', '%.4f', '%d'), delimiter=csv_sep)

#             total_loss_test += loss.data

#     # here, we can exit the code
#     if output_state_vectors:
#         return None

#     if print_epoch or map_flag:
#         test_acc = accuracy_score(y_true_test, y_pred_test)
#         test_pre = precision_score(y_true_test, y_pred_test)
#         test_rec = recall_score(y_true_test, y_pred_test)
#         test_macrof1 = f1_score(y_true_test, y_pred_test, average='macro')
#         test_weightedf1 = f1_score(y_true_test, y_pred_test, average='weighted')
#         test_loss = total_loss_test / len(test_dl)

#         if not map_flag:
#             test_map = False
#             epoch_log = '{} -- {}; loss: {:.3f}; acc: {:.3f}; precision: {:.3f}, recall: {:.3f}, macrof1: {:.3f}, weightedf1: {:.3f}'.format(
#                    datetime.now().strftime("%m/%d/%Y_%H:%M:%S"), eval_desc, test_loss, test_acc, test_pre, test_rec, test_macrof1, test_weightedf1)
#         else:
#             # computing MAP
#             list_of_list_of_trues = []
#             list_of_list_of_preds = []

#             for q,pred_ids in map_queries.items():
#                 q_preds = [y_score_test[x] for x in pred_ids]
#                 q_trues = [y_true_test[x] for x in pred_ids]
#                 list_of_list_of_preds.append(q_preds)
#                 list_of_list_of_trues.append(q_trues)

#             test_map = eval_map(list_of_list_of_trues, list_of_list_of_preds)

#             epoch_log = '{} -- {}; loss: {:.3f}; acc: {:.3f}; precision: {:.3f}, recall: {:.3f}, macrof1: {:.3f}, weightedf1: {:.3f}, map: {:.3f}'.format(
#                datetime.now().strftime("%m/%d/%Y_%H:%M:%S"), eval_desc, test_loss, test_acc, test_pre, test_rec, test_macrof1,test_weightedf1, test_map)

#         cprint('[INFO]', bc.lred, epoch_log)
#         if model_path:
#             log_message(epoch_log + "\n", mode="a+", filename=os.path.join(model_path, "log.txt"))
#         else:
#             log_message(epoch_log + "\n", mode="a+")
        
#         if tboard_writer:
#             # Record loss
#             tboard_writer.add_scalar('Test/Loss', loss.item(), epoch)
#             # Record Accuracy, precision, recall, F1, MAP on validation set 
#             tboard_writer.add_scalar('Test/Accuracy', test_acc, epoch)
#             tboard_writer.add_scalar('Test/Precision', test_pre, epoch)
#             tboard_writer.add_scalar('Test/Recall', test_rec, epoch)
#             tboard_writer.add_scalar('Test/MacroF1', test_macrof1, epoch)
#             tboard_writer.add_scalar('Test/WeightedF1', test_weightedf1, epoch)
#             if test_map:
#                tboard_writer.add_scalar('Test/Map', test_map, epoch)
#             tboard_writer.flush()
    
#     if output_loss:
#         return test_loss

#     if output_preds or map_flag:
#         return all_preds
#     #elif map_flag:
#     #    return (test_acc, test_pre, test_rec, test_macrof1, test_weightedf1, test_map)


# # ------------------- Self_Attention  --------------------
# class SelfAttention(nn.Module):
#     def __init__(self, emb, heads):
#         super().__init__()
#         # self.embedding_size = embedding_size
#         # self.heads = heads
#         # self.head_dim = embedding_size // heads

#         # # --- methods
#         # self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         # self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         # self.fc_out = nn.Linear(self.heads*self.head_dim, self.embedding_size)

#         self.emb = emb
#         self.heads = heads

#         self.tokeys = nn.Linear(emb, emb * heads, bias=False)
#         self.toqueries = nn.Linear(emb, emb * heads, bias=False)
#         self.tovalues = nn.Linear(emb, emb * heads, bias=False)

#         self.unifyheads = nn.Linear(heads * emb, emb)

#     def forward(self, x, mask):
#         b, t, e = x.size()
#         h = self.heads
        
#         keys    = self.tokeys(x)   .view(b, t, h, e)
#         queries = self.toqueries(x).view(b, t, h, e)
#         values  = self.tovalues(x) .view(b, t, h, e)

#         # compute scaled dot-product self-attention

#         # - fold heads into the batch dimension
#         keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
#         queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
#         values = values.transpose(1, 2).contiguous().view(b * h, t, e)

#         # - get dot product of queries and keys, and scale
#         dot = torch.bmm(queries, keys.transpose(1, 2))
#         dot = dot /(e**(1/2)) # dot contains b*h  t-by-t matrices with raw self-attention logits

#         dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities

#         # apply the self attention to the values
#         out = torch.bmm(dot, values).view(b, h, t, e)

#         # swap h, t back, unify heads
#         out = out.transpose(1, 2).contiguous().view(b, t, h * e)
#         out = self.unifyheads(out)
#         return out


# # ------------------- Encoder_Block  --------------------
# class EncoderBlock(nn.Module):
#     def __init__(self, embedding_size, heads, dropout, forward_expansion):
#         super().__init__()
#         self.attention = SelfAttention(embedding_size, heads)
#         self.norm_1 = nn.LayerNorm(embedding_size)
#         self.norm_2 = nn.LayerNorm(embedding_size)

#         self.feed_forward = nn.Sequential(
#             nn.Linear(embedding_size, forward_expansion*embedding_size),
#             nn.ReLU(),
#             nn.Linear(forward_expansion*embedding_size, embedding_size)
#         )
#         self.dropout = dropout
    
#     def forward(self, x, mask):
#         attention = self.attention(x, mask)
#         norm_1_out = F.dropout((self.norm_1(attention + x)), self.dropout)
#         feed_forward_out = self.feed_forward(norm_1_out)
#         norm_2_out = F.dropout(self.norm_2(feed_forward_out + norm_1_out), self.dropout)
#         return norm_2_out


# # ------------------- Transformer_Encoder  --------------------
# class TransformerEncoder(nn.Module):
#     def __init__(self, vocab_size, embedding_size, num_layers, 
#                 heads, device, dropout, forward_expansion, max_length):
#         super().__init__()
#         self.embedding_size = embedding_size
#         self.device = device
#         self.word_embedding = nn.Embedding(vocab_size, embedding_size)
#         self.positional_embedding = nn.Embedding(max_length, embedding_size)

#         self.layers = nn.ModuleList(
#             [
#                 EncoderBlock(
#                     embedding_size=embedding_size, 
#                     heads=heads, 
#                     dropout=dropout, 
#                     forward_expansion=forward_expansion)
#                 for _ in range(num_layers)
#             ]
#         )
#         self.dropout = dropout

#     def forward(self, x, mask):
#         seq_length, N = x.shape
#         positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
#         word_embed = self.word_embedding(x)
#         position_embed = self.positional_embedding(positions.transpose(0,1))

#         out = F.dropout((word_embed + position_embed), self.dropout)
#         #cprint('[WARNING]', bc.dred, "transformer embedd" + str(embedded.shape))
#         for layer in self.layers:
#             out = layer(out, mask)
#         return out

# # ------------------- Transformer_Encoder  --------------------
# class two_parallel_transformer(nn.Module):
#     def __init__(self, main_architecture, vocab_size, embedding_dim, rnn_hidden_dim, output_dim,
#                  rnn_n_layers, bidirectional, pooling_mode, rnn_drop_prob, rnn_bias,
#                  fc1_out_features, fc_dropout=[0.5, 0.5], att_dropout=[0.5, 0.5], 
#                  maxpool_kernel_size=2):
#         super().__init__()
#         self.main_architecture = main_architecture
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.rnn_hidden_dim = rnn_hidden_dim
#         self.output_dim = output_dim
#         self.gru_output_dim = embedding_dim
#         self.rnn_n_layers = rnn_n_layers
#         self.bidirectional = bidirectional
#         self.pooling_mode = pooling_mode
#         self.rnn_drop_prob = rnn_drop_prob
#         self.rnn_bias = rnn_bias
#         self.fc1_out_features = fc1_out_features
#         self.fc1_dropout = fc_dropout[0]
#         self.fc2_dropout = fc_dropout[1]
#         self.att1_dropout = att_dropout[0]
#         self.att2_dropout = att_dropout[1]
#         self.file_id = None

#         self.maxpool_kernel_size = maxpool_kernel_size

#         # --- methods
#         self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
#         self.transformer = TransformerEncoder(vocab_size=vocab_size, 
#                                             embedding_size=embedding_dim, 
#                                             num_layers=6, 
#                                             heads=8, 
#                                             device="cuda",
#                                             dropout=0.01, 
#                                             forward_expansion=2,
#                                             max_length=120)

#         # encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
#         # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

#         self.fc1 = nn.Linear(240, self.fc1_out_features)
#         self.fc2 = nn.Linear(self.fc1_out_features, self.output_dim)

#     # ------------------- forward 
#     def forward(self, x1_seq, len1, x2_seq, len2, pooling_mode='hstates', device="cpu", output_state_vectors=False, evaluation=False):

#         if evaluation:
#             # XXX Set dropouts to zero manually
#             self.att1_dropout = 0
#             self.att2_dropout = 0
#             self.fc1_dropout = 0
#             self.fc2_dropout = 0

#         rnn_out_1 = self.transformer(x1_seq, None)
#         #cprint('[WARNING]', bc.dred, "rnn_out_1 " + str(rnn_out_1.shape))
#         #rnn_out_1, len1 = pad_packed_sequence(rnn_out_1)

#         pool_1 = F.adaptive_max_pool1d(rnn_out_1.permute(1, 2, 0), 1).view(x1_seq.size(1), -1)
#         #cprint('[WARNING]', bc.dred, "pool_1 " + str(pool_1.shape))

#         rnn_out_2 = self.transformer(x2_seq, None)
#         #rnn_out_2, len2 = pad_packed_sequence(rnn_out_2)
#         #cprint('[WARNING]', bc.dred, "rnn_out_2 " + str(rnn_out_2.shape))
        
#         pool_2 = F.adaptive_max_pool1d(rnn_out_2.permute(1, 2, 0), 1).view(x2_seq.size(1), -1)

#         # Combine outputs from GRU1 and GRU2
#         pool_rnn_cat = torch.cat((pool_1, pool_2), dim=1)
#         #cprint('[WARNING]', bc.dred, "pool_rnn_cat " + str(pool_rnn_cat.shape))
#         pool_rnn_mul = pool_1 * pool_2
#         pool_rnn_dif = pool_1 - pool_2
#         output_combined = torch.cat((pool_rnn_cat,
#                                         pool_rnn_mul,
#                                         pool_rnn_dif), dim=1)

#         #cprint('[WARNING]', bc.dred, "outputcombine " + str(output_combined.shape))

#         #y_out = F.relu(self.fc1(F.dropout(pool_rnn_cat, self.fc1_dropout)))
#         y_out = F.relu(self.fc1(F.dropout(output_combined, self.fc1_dropout)))
#         y_out = self.fc2(F.dropout(y_out, self.fc2_dropout))
#         #cprint('[WARNING]', bc.dred, "y_out " + str(y_out.shape))
#         return y_out

#     def init_hidden(self, batch_size, device):
#         first_dim = self.rnn_n_layers
#         if self.bidirectional:
#             first_dim *= 2
#         return (Variable(torch.zeros((first_dim, batch_size, self.rnn_hidden_dim)).to(device)), 
#                 Variable(torch.zeros((first_dim, batch_size, self.rnn_hidden_dim)).to(device)))
              
    


