from config import configs
from dgl import DGLGraph

import dgl.function as fn
import pickle as pkl
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

class Classifer(nn.Module):
    def __init__(self,g,input_dim,num_classes,conv):
        super().__init__()
        self.GCN = conv
        self.gcn1 = self.GCN(g,input_dim, 200, F.relu)
        self.gcn2 = self.GCN(g, 200, num_classes, F.relu)
    
    def forward(self, features):
        x = self.gcn1(features)
        self.embedding = x
        x = self.gcn2(x)        
        return x


class SimpleConv(nn.Module):
    def __init__(self,g,in_feats,out_feats,activation,feat_drop=True):
        super().__init__()
        self.graph = g
        self.activation = activation
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats,out_feats)))
        self.feat_drop = feat_drop
      
    def forward(self, feat):
        g = self.graph.local_var()
        g.ndata['h'] = feat.mm(getattr(self, 'W'))
        g.update_all(fn.src_mul_edge(src='h', edge='w', out='m'), fn.sum(msg='m',out='h'))
        rst = g.ndata['h']
        rst = self.activation(rst)
        return rst

class GCNNModel:
    def train_model(self,dataset_type, train_val_percentage):
        """
            This function trains the GCNN Model
            dataset_type :
                The type of the dataset to train
            train_val_percentage
                The train_validation percentage split
        """
        (adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size) = self.load_data(dataset_type,train_val_percentage)

        #Pre process features
        features = self.preprocess_features(features)

        #Construct graph
        g = self.construct_graph(adj)

        #Define placeholders
        t_features = torch.from_numpy(features.astype(np.float32))
        t_y_train = torch.from_numpy(y_train)
        t_y_val = torch.from_numpy(y_val)
        t_y_test = torch.from_numpy(y_test)
        t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
        tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])

        #Model creation
        model = Classifer(g,input_dim=features.shape[0], num_classes=y_train.shape[1],conv=SimpleConv)

        if torch.cuda.is_available():
            t_features = t_features.cuda()
            t_y_train = t_y_train.cuda()
            t_train_mask = t_train_mask.cuda()
            tm_train_mask = tm_train_mask.cuda()
            model = model.cuda()

        #Criterion and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = configs.GCNN_LEARNING_RATE)

        # Train model
        val_losses = []        
        for epoch in range(configs.GCNN_EPHOCS):            
            # Forward pass
            logits = model(t_features)
            loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])    
            acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()
                
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            val_loss, val_acc, pred, labels = self.evaluate(model,criterion,t_features, t_y_val, val_mask)
            val_losses.append(val_loss)

            print("Epoch: {}, train_loss= {}, train_acc= {}, val_loss= {}, val_acc= {}".format(epoch + 1, loss, acc, val_loss, val_acc))

            if (epoch > configs.GCNN_EARLY_STOPPING) and val_losses[-1] > np.mean(val_losses[-(configs.GCNN_EARLY_STOPPING + 1): -1]):
                break
        
        test_loss, test_acc, pred, labels = self.evaluate(model,criterion,t_features, t_y_test, test_mask)
        print("Test set results:")
        print("loss= {}, accuracy= {}".format(test_loss, test_acc))

        #Compute predictions
        test_pred = []
        test_labels = []
        for i in range(len(test_mask)):
            if test_mask[i]:
                test_pred.append(pred[i])
                test_labels.append(np.argmax(labels[i]))

        #Compute precision recall, f1 score, support
        print("Test Precision, Recall and F1-Score...")
        print(metrics.classification_report(test_labels, test_pred))

    def evaluate(self, model, criterion, features, labels, mask):
        """
            Evaluation function for a single iteration during training and for testing.
            model :
                The model to use for evaluation
            criterion :
                Criterion to use for evaluation
            features :
                Features to use for evaluation
            labels :
                Labels to use for evaluation
            mask :
                Mask to use for evaluation
            return :
                A 4-tuple represeting : (loss, accuracy, predictions, labels_predicted)
        """
        model.eval()
        with torch.no_grad():
            logits = model(features).cpu()
            t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
            tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
            loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
            pred = torch.max(logits, 1)[1]
            acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
            
        return loss.numpy(), acc, pred.numpy(), labels.numpy()

    def preprocess_adj(self, adj):
        """
            This function symmetrically normalize adjacency matrix provided as parameter
            adj :
                The adj matrix to be normalized
            return :
                The adj matrix normalized
        """
        adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def construct_graph(self, adjacency):
        """
            This function create the graph to be used for building the model
            adjacency :
                The adj matrix loaded
            return :
                The graph to be used for building the model
        """
        g = DGLGraph()
        adj = self.preprocess_adj(adjacency)
        g.add_nodes(adj.shape[0])
        g.add_edges(adj.row,adj.col)
        adjdense = adj.A
        adjd = np.ones((adj.shape[0]))
        for i in range(adj.shape[0]):
            adjd[i] = adjd[i] * np.sum(adjdense[i,:])
        weight = torch.from_numpy(adj.data.astype(np.float32))
        g.ndata['d'] = torch.from_numpy(adjd.astype(np.float32))
        g.edata['w'] = weight       
        g = g.to(torch.device('cuda:0')) 
        
        return g

    def preprocess_features(self, features):
        """ 
            This function row-normalizes features matrix and convert to tuple representation
            features :
                The features to preprocess
            return :
                The processed features
        """
        features = sp.identity(features.shape[0])
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features.A

    def load_data(self,dataset_type, train_val_percentage):
        """
            This function return a 10-tuple that represent : 
            (adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size)
            dataset_type :
                The type of the dataset
            train_val_percentage :
                Train-validation percentage
            return :
                The 10-tuple (adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size)
        """
        #Paths of dumped objects        
        # dataset/corpus/<dataset_type>_<train_val_percentage>_x_train.x
        path_x_train = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_x_train.x"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_x_test.x
        path_x_test = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_x_test.x"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_x_all.x
        path_x_all = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_x_all.x"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_y_train.y
        path_y_train = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_y_train.y"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_y_test.y
        path_y_test = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_y_test.y"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_y_all.y
        path_y_all = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_y_all.y"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_adj.adj
        path_adj = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_adj.adj"

        objects = []

        #Load dumped objects
        objects.append(self.load_dumped_object(path_x_train))
        objects.append(self.load_dumped_object(path_x_test))
        objects.append(self.load_dumped_object(path_x_all))
        objects.append(self.load_dumped_object(path_y_train))
        objects.append(self.load_dumped_object(path_y_test))
        objects.append(self.load_dumped_object(path_y_all))
        objects.append(self.load_dumped_object(path_adj))

        #Get the loaded objects
        x_train, x_test, x_all, y_train, y_test, y_all, adj = tuple(objects)

        features = sp.vstack((x_all, x_test)).tolil()
        labels = np.vstack((y_all, y_test))        

        #Compute train_size val_size test_size
        train_size = x_train.shape[0]
        val_size = self.get_train_and_validation_size(dataset_type, train_val_percentage) - train_size
        test_size = x_test.shape[0]

        #Compute ids
        ids_train = range(len(y_train))
        ids_val = range(len(y_train), len(y_train) + val_size)
        ids_test = range(x_all.shape[0], x_all.shape[0] + test_size)

        #Compute masks
        train_mask = self.sample_mask(ids_train, labels.shape[0])
        val_mask = self.sample_mask(ids_val, labels.shape[0])
        test_mask = self.sample_mask(ids_test, labels.shape[0])

        #Create one-hot vectors to return
        y_train_ret = np.zeros(labels.shape)
        y_val_ret = np.zeros(labels.shape)
        y_test_ret = np.zeros(labels.shape)
        
        #Populate one-hot vectors
        y_train_ret[train_mask, :] = labels[train_mask, :]
        y_val_ret[val_mask, :] = labels[val_mask, :]
        y_test_ret[test_mask, :] = labels[test_mask, :]

        #Compute adj matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        return adj, features, y_train_ret, y_val_ret, y_test_ret, train_mask, val_mask, test_mask, train_size, test_size

    def sample_mask(self, ids, length):
        """
            Create a zero mask of size length with a 1 in positions ids
            ids :
                Position where to place a 1
            length :
                Size of the mask
            return :
                The mask
        """
        mask = np.zeros(length)
        mask[ids] = 1
        return np.array(mask, dtype=np.bool)
    
    def get_train_and_validation_size(self,dataset_type, train_val_percentage):
        """
            This function returns the length of the train+validation set
            dataset_type :
                The type of the dataset
            train_val_percentage :
                The train-validation percentage
            return :
                The length of the train+validation set
        """
        path = configs.GCNN_PATH_DATASET_SPLIT + dataset_type + "_" + str(train_val_percentage) + "_" + "ids_train_test" + configs.GCNN_DATASET_FILE_FORMAT

        with open(path) as file:
            return len(file.readlines())

    def load_dumped_object(self, path_object):
        """
            This function loads a dumped object
            path_object :
                Path of the dumped object
            return :
                The dumped object
        """
        with open(path_object, 'rb') as file:
            return pkl.load(file)