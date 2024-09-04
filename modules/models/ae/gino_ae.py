import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

from modules.modules.integral_transform import IntegralTransform, MLPLinear
from modules.modules.neighbor_search import NeighborSearch
from modules.modules.embedding import FourierEmb
from modules.models.ae.cnn_ae import CNN_Encoder, CNN_Decoder

class GINO_Encoder(nn.Module):
    """GINO: Geometry-informed Neural Operator

        Parameters
        ----------
        in_channels : int
            feature dimension of input points
        out_channels : int
            feature dimension of output points
        projection_channels : int, optional
            number of channels in FNO pointwise projection
        gno_coord_dim : int, optional
            geometric dimension of input/output queries, by default 3
        gno_coord_embed_dim : int, optional
            dimension of positional embedding for gno coordinates, by default None
        gno_embed_max_positions : int, optional
            max positions for use in gno positional embedding, by default None
        gno_radius : float, optional
            radius in input/output space for GNO neighbor search, by default 0.033
        gno_mlp_hidden_layers : list, optional
            widths of hidden layers in input GNO, by default [80, 80, 80]
        gno_mlp_non_linearity : nn.Module, optional
            nonlinearity to use in gno MLP, by default F.gelu
        gno_transform_type : str, optional
            transform type parameter for output GNO, by default 'linear'
            see neuralop.layers.IntegralTransform
        gno_use_torch_scatter : bool, optional
            whether to use torch_scatter's neighborhood reduction function
            or the native PyTorch implementation in IntegralTransform layers.
            If False, uses the fallback PyTorch version.
        """
    def __init__(
            self,
            in_channels,
            projection_channels=256,
            gno_coord_dim=3,
            gno_coord_embed_dim=None,
            gno_radius=0.05,
            gno_mlp_hidden_layers=[80, 80, 80],
            gno_mlp_non_linearity=F.gelu, 
            gno_transform_type='linear',
            gno_use_torch_scatter=True,
            use_open3d=True,
        ):
        
        super().__init__()
        self.in_channels = in_channels
        self.projection_channels = projection_channels
        self.gno_coord_dim = gno_coord_dim
        
        self.nb_search_out = NeighborSearch(use_open3d=use_open3d)
        self.gno_radius = gno_radius

        self.x_projection = MLPLinear(layers=[in_channels, projection_channels])

        if gno_coord_embed_dim is not None:
            self.pos_embed = FourierEmb(hidden_dim=gno_coord_embed_dim, in_dim=gno_coord_dim)
            self.gno_coord_dim = gno_coord_embed_dim 
        else:
            self.pos_embed = None
        
        ### input GNO
        # input to the first GNO MLP: x pos encoding, y (integrand) pos encoding
        in_kernel_in_dim = self.gno_coord_dim * 2
        # add f_y features if input GNO uses a nonlinear kernel
        if gno_transform_type == "nonlinear" or gno_transform_type == "nonlinear_kernelonly":
            in_kernel_in_dim += self.projection_channels
            
        gno_mlp_hidden_layers.insert(0, in_kernel_in_dim)
        gno_mlp_hidden_layers.append(projection_channels) 
        self.gno_in = IntegralTransform(
                    mlp_layers=gno_mlp_hidden_layers,
                    mlp_non_linearity=gno_mlp_non_linearity,
                    transform_type=gno_transform_type,
                    use_torch_scatter=gno_use_torch_scatter
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, input_geom, latent_queries):
        """forward pass of GNO --> latent embedding w/FNO --> GNO out

        Parameters
        ----------
        x : torch.Tensor
            input function a defined on the input domain `input_geom`
            shape (1, n_in, in_channels) 
        input_geom : torch.Tensor
            input domain coordinate mesh
            shape (1, n_in, gno_coord_dim)
        latent_queries : torch.Tensor
            latent geometry on which to compute FNO latent embeddings
            a grid on [0,1] x [0,1] x ....
            shape (1, n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        """
        batch_size = x.shape[0]
        input_geom = input_geom.squeeze(0) 
        latent_queries = latent_queries.squeeze(0)
    
        spatial_nbrs = self.nb_search_out(input_geom, 
                                          latent_queries.view((-1, latent_queries.shape[-1])), 
                                          radius=self.gno_radius)
        x = self.x_projection(x) # b, n_in, in_channels -> b, n_in, projection_channels
        x = x.squeeze(0) # n_in, projection_channels

        if self.pos_embed is not None:
            input_geom = self.pos_embed(input_geom) # n_in, coord_dim -> n_in, gno_coord_dim_embed
            latent_queries = self.pos_embed(latent_queries) # n_gridpts_1, .... n_gridpts_n, coord_dim -> n_gridpts_1, .... n_gridpts_n, gno_coord_dim_embed

        in_p = self.gno_in(y=input_geom, # n_in, gno_coord_dim
                           x=latent_queries.view((-1, latent_queries.shape[-1])), # (n_gridpts_1, .... n_gridpts_n), gno_coord_dim
                           f_y=x, # b, n_in, projection_channels
                           neighbors=spatial_nbrs)
        
        grid_shape = latent_queries.shape[:-1] 
        in_p = in_p.view((batch_size, *grid_shape, self.projection_channels))
        
        return in_p # shape [1, n_gridpts_1, ..., n_gridpts_n, f_dim]

class GINO_Decoder(nn.Module):
    """GINO: Geometry-informed Neural Operator

        Parameters
        ----------
        out_channels : int
            feature dimension of output points
        projection_channels : int, optional
            number of channels in FNO pointwise projection
        gno_coord_dim : int, optional
            geometric dimension of input/output queries, by default 3
        gno_coord_embed_dim : int, optional
            dimension of positional embedding for gno coordinates, by default None
        gno_embed_max_positions : int, optional
            max positions for use in gno positional embedding, by default None
        gno_radius : float, optional
            radius in input/output space for GNO neighbor search, by default 0.033
        gno_mlp_hidden_layers : list, optional
            widths of hidden layers in output GNO, by default [512, 256]
        gno_mlp_non_linearity : nn.Module, optional
            nonlinearity to use in gno MLP, by default F.gelu
        gno_transform_type : str, optional
            transform type parameter for output GNO, by default 'linear'
            see neuralop.layers.IntegralTransform
        gno_use_torch_scatter : bool, optional
            whether to use torch_scatter's neighborhood reduction function
            or the native PyTorch implementation in IntegralTransform layers.
            If False, uses the fallback PyTorch version.
        out_gno_tanh : bool, optional
            whether to use tanh to stabilize outputs of the output GNO, by default False
        """
    def __init__(
            self,
            out_channels,
            projection_channels=256,
            gno_coord_dim=3,
            gno_coord_embed_dim=None,
            gno_radius=0.033,
            gno_mlp_hidden_layers=[512, 256],
            gno_mlp_non_linearity=F.gelu, 
            gno_transform_type='linear',
            gno_use_torch_scatter=True,
            use_open3d=True,
            tanh_out = False,
        ):
        
        super().__init__()
        self.out_channels = out_channels
        self.gno_coord_dim = gno_coord_dim
        
        self.nb_search_out = NeighborSearch(use_open3d=use_open3d)
        self.gno_radius = gno_radius
        self.tanh_out = tanh_out

        if gno_coord_embed_dim is not None:
            self.pos_embed = FourierEmb(hidden_dim=gno_coord_embed_dim, in_dim=gno_coord_dim)
            self.gno_coord_dim = gno_coord_embed_dim 
        else:
            self.pos_embed = None

        ### output GNO
        out_kernel_in_dim = 2 * self.gno_coord_dim
        out_kernel_in_dim += projection_channels if gno_transform_type != 'linear' else 0
        gno_mlp_hidden_layers.insert(0, out_kernel_in_dim)
        gno_mlp_hidden_layers.append(projection_channels)
        self.gno_out = IntegralTransform(
                    mlp_layers=gno_mlp_hidden_layers,
                    mlp_non_linearity=gno_mlp_non_linearity,
                    transform_type=gno_transform_type,
                    use_torch_scatter=gno_use_torch_scatter
        )

        self.projection = MLPLinear(layers=[projection_channels, out_channels]) 


    # out_p : (n_out, gno_coord_dim)

    def integrate_latent(self, in_p, out_p, latent_embed):
        # in_p is in shape (n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        # out_p is in shape (n_out, gno_coord_dim)

        in_to_out_nb = self.nb_search_out(
            in_p.view(-1, in_p.shape[-1]), 
            out_p,
            self.gno_radius,
            )# for each output point, find the neighbors in the latent grid 
    
        #Embed input points
        n_in = in_p.view(-1, in_p.shape[-1]).shape[0]
        in_p_embed = in_p.reshape((n_in, -1)) # flatten to ((n_gridpts_1, .... n_gridpts_n), gno_coord_dim)
        if self.pos_embed is not None:
            in_p_embed = self.pos_embed(in_p_embed)
        
        #Embed output points
        out_p_embed = out_p
        if self.pos_embed is not None:
            out_p_embed = self.pos_embed(out_p_embed)
        
        latent_embed = rearrange(latent_embed, 'b n1 n2 n3 c -> b (n1 n2 n3) c')
        # rehape to batch, (n_1 * n_2 * ... * n_k), hidden_channels

        #(n_out, fno_hidden_channels)
        out = self.gno_out(y=in_p_embed, 
                    neighbors=in_to_out_nb,
                    x=out_p_embed,
                    f_y=latent_embed,) # apply kernel integration to latent embedding, sum on output points
        
        out = self.projection(out)

        if self.tanh_out:
            out = torch.tanh(out)

        return out
    
    def forward(self, latent_embed, latent_queries, output_queries):
        """forward pass of GNO --> latent embedding w/FNO --> GNO out

        Parameters
        ----------
        latent_embed : torch.Tensor
            latent_embedding to be decoded
            shape (batch, n_gridpts_1, .... n_gridpts_n, hidden_channels) 
        latent_queries : torch.Tensor
            latent geometry on which to compute FNO latent embeddings
            a grid on [0,1] x [0,1] x ....
            shape (1, n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        output_queries : torch.Tensor
            points at which to query the final GNO layer to get output
            shape (batch, n_out, gno_coord_dim)
        """
        latent_queries = latent_queries.squeeze(0)
        output_queries = output_queries.squeeze(0)

        out = self.integrate_latent(latent_queries, output_queries, latent_embed) 

        return out # shape (batch, n_latent_queries, f_dim)

class Encoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            gno_coord_dim=3,
            gno_coord_embed_dim=None,
            gno_radius=0.033,
            gno_mlp_hidden_layers=[80, 80, 80],
            gno_mlp_non_linearity=F.gelu, 
            gno_transform_type='linear',
            gno_use_torch_scatter=True,
            hidden_channels=64,
            ch_mult = (1, 2, 2, 4),
            num_res_blocks = 1,
            attn_resolutions = (16, ),
            dropout = 0.0,
            resolution = 64,
            z_channels = 256,
            double_z = False,
            use_open3d=True,
            tanh_out=False
        ):

        super().__init__()

        self.gino_encoder = GINO_Encoder(
            in_channels=in_channels,
            projection_channels=hidden_channels,
            gno_coord_dim=gno_coord_dim,
            gno_coord_embed_dim=gno_coord_embed_dim,
            gno_radius=gno_radius,
            gno_mlp_hidden_layers=gno_mlp_hidden_layers,
            gno_mlp_non_linearity=gno_mlp_non_linearity,
            gno_transform_type=gno_transform_type,
            gno_use_torch_scatter=gno_use_torch_scatter,
            use_open3d=use_open3d
        )

        self.cnn_encoder = CNN_Encoder(in_channels=hidden_channels,
                                     hidden_channels=hidden_channels,
                                     out_channels=out_channels,
                                     ch_mult=ch_mult,
                                     num_res_blocks=num_res_blocks,
                                     attn_resolutions=attn_resolutions,
                                     dropout=dropout,
                                     resolution=resolution,
                                     z_channels=z_channels,
                                     double_z=double_z,
                                     tanh_out=tanh_out,
                                     dim=gno_coord_dim)
        
    def forward(self, x, input_geom, latent_queries, pad_mask=None):
        """forward pass of GNO --> latent embedding w/FNO --> GNO out

        Parameters
        ----------
        x : torch.Tensor
            input function a defined on the input domain `input_geom`
            shape (batch, nt, n, in_channels) 
        input_geom : torch.Tensor
            input domain coordinate mesh
            shape (batch, nt, n, gno_coord_dim)
        latent_queries : torch.Tensor
            latent geometry on which to compute FNO latent embeddings
            a grid on [0,1] x [0,1] x ....
            shape (1, n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        """

        b = x.shape[0]

        if b > 1:
            latent = []
            for i in range(b):
                x_batch = x[i].unsqueeze(0) # shape [1, nt, n, in_channels]
                input_geom_batch = input_geom[i].unsqueeze(0) # shape [1, nt, n, gno_coord_dim]
                length_batch = torch.sum(pad_mask[i], dtype=torch.long) # pad_mask in shape [b, 1, n, 1]
 
                x_batch = x_batch[:, :, :length_batch]
                input_geom_batch = input_geom_batch[:, :, :length_batch]

                x_batch = rearrange(x_batch, 'b nt n c -> b (nt n) c')
                input_geom_batch = rearrange(input_geom_batch, 'b nt n c -> b (nt n) c')

                latent_batch = self.gino_encoder(x_batch, input_geom_batch, latent_queries) # 1 n_gridpts_1, .... n_gridpts_n, hidden_channels
                latent.append(latent_batch)
            latent = torch.cat(latent, dim=0) # shape [batch, n_gridpts_1, .... n_gridpts_n, hidden_channels]

        else:
            x = rearrange(x, 'b nt n c -> b (nt n) c')
            input_geom = rearrange(input_geom, 'b nt n c -> b (nt n) c')
            latent = self.gino_encoder(x, input_geom, latent_queries) # shape [batch, n_gridpts_1, ..., n_gridpts_n, f_dim]

        if len(latent.shape) == 4:
            latent = rearrange(latent, 'b n1 n2 c -> b c n1 n2')
        else:
            latent = rearrange(latent, 'b n1 n2 n3 c -> b c n1 n2 n3')
        z = self.cnn_encoder(latent)

        return z
    
class Decoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            gno_coord_dim=3,
            gno_coord_embed_dim=None,
            gno_radius=0.033,
            gno_mlp_hidden_layers=[80, 80, 80],
            gno_mlp_non_linearity=F.gelu, 
            gno_transform_type='linear',
            gno_use_torch_scatter=True,
            hidden_channels=64,
            ch_mult = (1, 2, 2, 4),
            num_res_blocks = 1,
            attn_resolutions = (16, ),
            dropout = 0.0,
            resolution = 64,
            z_channels = 256,
            double_z = False,
            use_open3d=True,
            tanh_out=False
        ):

        super().__init__()

        self.gino_decoder = GINO_Decoder(out_channels=out_channels,
                                         projection_channels=hidden_channels,
                                         gno_coord_dim=gno_coord_dim,
                                         gno_coord_embed_dim=gno_coord_embed_dim,
                                         gno_radius=gno_radius,
                                         gno_mlp_hidden_layers=gno_mlp_hidden_layers,
                                         gno_mlp_non_linearity=gno_mlp_non_linearity,
                                         gno_transform_type=gno_transform_type,
                                         gno_use_torch_scatter=gno_use_torch_scatter,
                                         use_open3d=use_open3d,
                                         tanh_out=tanh_out)
        
        self.cnn_decoder = CNN_Decoder(in_channels=in_channels,
                                     hidden_channels=hidden_channels,
                                     out_channels=hidden_channels,
                                     ch_mult=ch_mult,
                                     num_res_blocks=num_res_blocks,
                                     attn_resolutions=attn_resolutions,
                                     dropout=dropout,
                                     resolution=resolution,
                                     z_channels=z_channels,
                                     double_z=double_z,
                                     dim=gno_coord_dim)
        
        
    def forward(self, z, latent_queries, output_queries, pad_mask=None):
        """forward pass of GNO --> latent embedding w/FNO --> GNO out

        Parameters
        ----------
        z : torch.Tensor
            latent_embedding to be decoded
            shape (batch, embed_dim, n_1, .... n_n) 
        latent_queries : torch.Tensor
            latent geometry on which to compute FNO latent embeddings
            a grid on [0,1] x [0,1] x ....
            shape (1, n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        output_queries : torch.Tensor
            points at which to query the final GNO layer to get output
            shape (batch, nt, n, gno_coord_dim)
        """

        latent_embed = self.cnn_decoder(z)
        if len(latent_embed.shape) == 4:
            latent_embed = rearrange(latent_embed, 'b c n1 n2 -> b n1 n2 c')
        else:
            latent_embed = rearrange(latent_embed, 'b c n1 n2 n3 -> b n1 n2 n3 c')

        b = z.shape[0]

        if b > 1:
            out = []
            for i in range(b):
                length_batch = torch.sum(pad_mask[i], dtype=torch.long)  # pad_mask in shape [b, 1, n, 1]
                max_length = pad_mask.shape[2]

                output_queries_batch = output_queries[i].unsqueeze(0) # shape [1, nt, n, gno_coord_dim]
                output_queries_batch = output_queries_batch[:, :, :length_batch] # truncate to unpadded length
                _, nt, n, _ = output_queries_batch.shape
                output_queries_batch = rearrange(output_queries_batch, 'b nt n c -> b (nt n) c') # reshape for gino

                latent_embed_batch = latent_embed[i].unsqueeze(0) # shape [1, n_1, .... n_n, hidden_channels]
                out_batch = self.gino_decoder(latent_embed_batch, latent_queries, output_queries_batch) # shape [1, n_out, f_dim]
                out_batch = rearrange(out_batch, 'b (nt n) c -> b nt n c', nt=nt, n=n)

                # pad output back to max length, (1, nt, l, c) + (1, nt, max_length - l, c) -> (1, nt, max_length, c)
                out_batch = torch.cat([out_batch, torch.zeros(1, nt, max_length - length_batch, out_batch.shape[-1], device=z.device)], dim=2)
                out.append(out_batch)

            out = torch.cat(out, dim=0) # shape [batch, nt, n, f_dim]
        else:
            _, nt, n, c = output_queries.shape
            output_queries = rearrange(output_queries, 'b nt n c -> b (nt n) c')
            out = self.gino_decoder(latent_embed, latent_queries, output_queries)
            out = rearrange(out, "b (nt n) c -> b nt n c", nt=nt, n=n)

        return out

class ConditionalEncoder(nn.Module):
    def __init__(
            self,
            config,
        ):

        super().__init__()
        if "ablate" in config.keys() and config["ablate"]:
            self.ablate = True 
            self.encoder = nn.Sequential(nn.Linear(1, config["out_dim"]),
                                         nn.GELU(),
                                        nn.Linear(config["out_dim"], config["out_dim"]))
            latent_queries = self.get_latent_grid(1) # dummy latent grid
            self.register_buffer('latent_grid', latent_queries)
            print("Ablating encoder")
        
        else:
            self.ablate = False
            self.encoder = Encoder(**config["encoder"])
            self.flatten = Rearrange('b c n1 n2 -> b (n1 n2) c')
            self.head = nn.Linear(config["encoder"]['z_channels'], config['out_dim'])
            self.latent_dim = config["encoder"]["resolution"]
            latent_queries = self.get_latent_grid(self.latent_dim)

            self.register_buffer('latent_grid', latent_queries)

    def get_latent_grid(self, N):
        xx = torch.linspace(0, 1, N)
        yy = torch.linspace(0, 1, N)

        xx, yy = torch.meshgrid(xx, yy, indexing='ij')
        latent_queries = torch.stack([xx, yy], dim=-1)
        
        return latent_queries.unsqueeze(0)
        
    def forward(self, x, input_geom=None, latent_queries=None, pad_mask=None):
        """

        Parameters
        ----------
        x : torch.Tensor
            input function a defined on the input domain `input_geom`
            shape (batch, n_in, in_channels) 
        input_geom : torch.Tensor
            input domain coordinate mesh
            shape (1, n_in, gno_coord_dim)
        latent_queries : torch.Tensor
            latent geometry on which to compute FNO latent embeddings
            a grid on [0,1] x [0,1] x ....
            shape (1, n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        """
        if self.ablate:
            x = x.unsqueeze(-1) # B N -> B N 1
            z = self.encoder(x) # B N D

        else:
            z = self.encoder(x, input_geom, latent_queries, pad_mask) # b c n1 n2 n3
            z = self.flatten(z) # b (n1 n2) c
            z = self.head(z) # b (n1 n2) out_dim
        return z # expected in b n d for cross attention
    