import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data
import numpy as np
import lightning as L

'''
Fully self contained MeshGraphNet implementation. 
https://github.com/echowve/meshGraphNets_pytorch
'''

######################
# Module
######################

class GNNModule(L.LightningModule):
    def __init__(self,
                 modelconfig,
                 trainconfig,
                 normalizer=None,
                 batch_size=1,
                 accumulation_steps=1,
                 ckpt_path=None
                 ):
        super().__init__()

        self.model = EncoderProcesserDecoder(**modelconfig)
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.trainconfig = trainconfig

        self.dist = trainconfig['dist']

        self.save_hyperparameters()

        print("Training with batch size", self.batch_size)
        print("Training with accumulation steps", self.accumulation_steps)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, graph):
        velocity = graph.x[:, :2] 
        out = self.model(graph)
        out[:, :2] = out[:, :2] + velocity # forward euler step
        return out
    
    def get_graph_data_labels(self, x, y, pos, edge_idx) -> Data:
        # assume x, y in shape m c
        # pos in shape m 2
        # edge_idx in shape 2 e

        input_distances = torch.norm(x[edge_idx[0]] - x[edge_idx[1]], dim=1, p=1).unsqueeze(1)
        edge_distances = torch.norm(pos[edge_idx[0]] - pos[edge_idx[1]], dim=1, p=1).unsqueeze(1)

        edge_attr = torch.cat((input_distances, edge_distances), dim=1)

        graph = Data(x=x, edge_index=edge_idx, pos=pos, y=y, edge_attr=edge_attr)
        return graph
    
    def triangles_to_edges(self, cells):
        """Computes mesh edges from triangles."""
        # collect edges from triangles
        t_cell = torch.tensor(cells)
        edge_index = torch.cat((t_cell[:, :2], t_cell[:, 1:3], torch.cat((t_cell[:, 2].unsqueeze(1), t_cell[:, 0].unsqueeze(1)), -1)), 0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single torch long tensor
        r, _ = torch.min(edge_index, 1, keepdim=True)
        s, _ = torch.max(edge_index, 1, keepdim=True)
        packed_edges = torch.cat((s, r), 1).type(torch.long)
        # remove duplicates and unpack
        unique_edges = torch.unique(packed_edges, dim=0)
        s, r = unique_edges[:, 0], unique_edges[:, 1]
        # create two-way connectivity
        return torch.stack((torch.cat((s, r), 0), torch.cat((r, s), 0)))

    def training_step(self, batch, batch_idx):
        # assume batch size of 1 

        inputs = batch["x"].squeeze()  # t m c 
        pos = batch["pos"]
        pos = pos[:, 0, :, :2].squeeze() # m 2
        cells = batch["cells"].squeeze() # e 3
        edge_idx = self.triangles_to_edges(cells) # e 2
        inputs = self.normalizer.normalize(inputs)  # normalize inputs to [-1, 1]

        t = torch.randint(0, inputs.shape[0]-1, (1,)).item()
        x = inputs[t] # m c
        y = inputs[t+1] # m c

        graph = self.get_graph_data_labels(x, y, pos, edge_idx)

        pred = self(graph) # m c
        loss = F.mse_loss(pred, graph.y) # m c, m c

        self.log("train/mse_loss", loss, prog_bar=False,
                      logger=True, on_step=True, on_epoch=True,
                      sync_dist=self.dist,)
        
        labels_denorm = self.normalizer.denormalize(graph.y) # m c
        pred_denorm = self.normalizer.denormalize(pred) # m c 

        train_loss_denorm = F.l1_loss(labels_denorm, pred_denorm) # single step error 
        self.log("train/l1_loss", train_loss_denorm, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,
                        sync_dist=self.dist,)

        return loss 

    def validation_step(self, batch, batch_idx, eval=False ):
        # assume batch size of 1 
        inputs = batch["x"].squeeze()  # t m c 
        pos = batch["pos"]
        pos = pos[:, 0, :, :2].squeeze() # m 2
        cells = batch["cells"].squeeze() # e 3
        edge_idx = self.triangles_to_edges(cells) # e 2
        inputs = self.normalizer.normalize(inputs)  # normalize inputs to [-1, 1]

        t = torch.randint(0, inputs.shape[0]-1, (1,)).item()
        x = inputs[t] # m c
        y = inputs[t+1] # m c

        graph = self.get_graph_data_labels(x, y, pos, edge_idx)

        pred = self(graph) # m c
        loss = F.mse_loss(pred, graph.y) # m c, m c

        labels_denorm = self.normalizer.denormalize(graph.y) # m c
        pred_denorm = self.normalizer.denormalize(pred) # m c 

        loss_denorm = F.l1_loss(labels_denorm, pred_denorm) # single step error 
        
        rollout_error = 0 

        all_errors = []
        all_preds = torch.zeros_like(inputs) # t m c
        pred_step = inputs[0]
        all_preds[0] = pred_step
        t = inputs.shape[0]

        for i in range(t-1):
            graph_rollout = self.get_graph_data_labels(pred_step, inputs[i+1], pos, edge_idx)
            pred_step = self(graph_rollout) # predicts at i+1
            single_step_error = F.l1_loss(pred_step, graph_rollout.y)
            all_errors.append(single_step_error.item())
            all_preds[i+1] = pred_step

        if eval:
            return all_errors, all_preds
        
        rollout_error = F.l1_loss(self.normalizer.denormalize(all_preds), self.normalizer.denormalize(inputs))

        self.log("val/mse_loss", loss, prog_bar=False,
                      logger=True, on_step=False, on_epoch=True,
                      sync_dist=self.dist,)
        self.log("val/l1_loss", loss_denorm, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,
                        sync_dist=self.dist,)
        self.log("val/rollout_loss", rollout_error, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,
                        sync_dist=self.dist,)
        
        return loss

    def configure_optimizers(self):
        lr = self.trainconfig["learning_rate"]
        opt_ae = torch.optim.Adam(self.model.parameters(),
                                    lr=lr,)
            
        effective_batch_size = self.batch_size * self.accumulation_steps
        if self.trainconfig["scheduler"] == "OneCycle":
            scheduler_ae = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_ae,
                                                            max_lr=lr,
                                                            total_steps=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),
                                                            pct_start=self.trainconfig["pct_start"],)

        elif self.trainconfig["scheduler"] == "Cosine":
            scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt_ae,
                                                                      T_max=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),)
        else:
            scheduler_ae = None
        return [opt_ae], [scheduler_ae]

######################
# Model
######################

# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def decompose_graph(graph):
    # graph: torch_geometric.data.data.Data
    # TODO: make it more robust
    x, edge_index, edge_attr = None, None, None
    for key in graph.keys():
        if key=="x":
            x = graph.x
        elif key=="edge_index":
            edge_index = graph.edge_index
        elif key=="edge_attr":
            edge_attr = graph.edge_attr
        else:
            pass
    return (x, edge_index, edge_attr)

# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    node_attr, edge_index, edge_attr = decompose_graph(graph)
    
    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    
    return ret

class EdgeBlock(nn.Module):

    def __init__(self, custom_func=None):
        
        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):

        node_attr, edge_index, edge_attr, = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        edges_to_collect = []

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)

        collected_edges = torch.cat(edges_to_collect, dim=1)
        
        edge_attr_ = self.net(collected_edges)   # Update

        return Data(x=node_attr, edge_attr=edge_attr_, edge_index=edge_index)

class NodeBlock(nn.Module):

    def __init__(self, custom_func=None):

        super(NodeBlock, self).__init__()

        self.net = custom_func

    def forward(self, graph):
        # Decompose graph
        edge_attr = graph.edge_attr
        nodes_to_collect = []
        
        _, receivers_idx = graph.edge_index
        num_nodes = graph.num_nodes
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)

        nodes_to_collect.append(graph.x)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)
        x = self.net(collected_nodes)
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = nn.Sequential(nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_size))
    if lay_norm: return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    return module

class Encoder(nn.Module):

    def __init__(self,
                edge_input_size=128,
                node_input_size=128,
                hidden_size=128):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
    
    def forward(self, graph):

        node_attr, _, edge_attr = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)
        
        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)

class GnBlock(nn.Module):

    def __init__(self, hidden_size=128):

        super(GnBlock, self).__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
    
        graph_last = copy_geometric_data(graph)
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)
        edge_attr = graph_last.edge_attr + graph.edge_attr
        x = graph_last.x + graph.x
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)

class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, hidden_size=128):

        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size, hidden_size=hidden_size)
        
        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)
        
        self.decoder = Decoder(hidden_size=hidden_size, output_size=node_input_size)

    def forward(self, graph):

        graph= self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded
    


