import math
from logging import raiseExceptions

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def init_gru(rnn):
    """Initialize a GRU layer. """

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

# ---------------------------------------------------------------------------------
# from https://github.com/ap229997/Neural-Toolbox-PyTorch/blob/master/film_layer.py
# FiLM layer with a linear transformation from context to FiLM parameters
# ---------------------------------------------------------------------------------

class FilmLayer(nn.Module):
    # def __init__(self): for default FilmLayer used in original paper of Olga.
    def __init__(self, in_planes, out_planes):  # init first not to miss state_duct keys.
        super(FilmLayer, self).__init__()

        self.batch_size = None
        self.channels = None
        self.height = None
        self.width = None
        self.extra5d = None
        self.feature_size = None
        self.in_planes = in_planes
        self.out_planes = out_planes

        if config.condition_net == "DynamicFCInitFirst":
            self.condition_gen_fc = DynamicFCInitFirst(
                self.in_planes, self.out_planes * 2
            )  # 2 is for beta and gamma
        elif config.condition_net == "DynamicFC":
            self.condition_gen_fc = DynamicFC()
        else:
            raiseExceptions("No such condition_net exists.")

    def film3d(self, feature_maps, film_params):
        self.batch_size, self.channels, self.height = feature_maps.data.shape
        # stack the FiLM parameters across the temporal dimension
        film_params = torch.stack([film_params] * self.height, dim=2)

        # slice the film_params to get betas and gammas
        gammas = film_params[:, : self.feature_size, :]
        betas = film_params[:, self.feature_size :, :]

        return gammas, betas

    def film4d(self, feature_maps, film_params):
        self.batch_size, self.channels, self.height, self.width = feature_maps.data.shape

        # stack the FiLM parameters across the spatial dimension
        film_params = torch.stack([film_params] * self.height, dim=2)
        film_params = torch.stack([film_params] * self.width, dim=3)

        # slice the film_params to get betas and gammas
        gammas = film_params[:, : self.feature_size, :, :]
        betas = film_params[:, self.feature_size :, :, :]

        return gammas, betas

    def forward(self, feature_maps, context):
        """
        Arguments:
            feature_maps : input feature maps (feature vectors from the main DNN to be affine trasformed) (N, C, H, W) or (N, C, W)
            context : context (i.e. condition to be considered; score info)
            embedding (N, L)
        Return:
            output : feature maps modulated with betas and gammas (FiLM parameters)
        """

        # FiLM parameters needed for each channel in the feature map
        # hence, feature_size defined to be same as no. of channels
        self.feature_size = feature_maps.data.shape[1]

        # linear transformation of context to FiLM parameters
        if config.condition_net == "DynamicFC":
            context = torch.flatten(context, start_dim=1)
            # context.shape supposed to be (Batch_size, number_of_frames, num_classes) e.g. torch.Size([1, 201, 88])
            film_params = self.condition_gen_fc(
                context, out_planes=2 * self.feature_size, activation=None
            )
        elif config.condition_net == "DynamicFCInitFirst":
            context = torch.flatten(context, start_dim=1)
            film_params = self.condition_gen_fc(context)
        else:
            raise Exception("No such conditional model exists.")

        gammas, betas = (
            self.film4d(feature_maps, film_params)
            if len(feature_maps.data.shape) == 4
            else self.film3d(feature_maps, film_params)
        )
        # modulate the feature map with FiLM parameters
        output = (1 + gammas) * feature_maps + betas

        return output


class DynamicFC(nn.Module):
    def __init__(self):
        super(DynamicFC, self).__init__()

        self.in_planes = None
        self.out_planes = None
        self.activation = None
        self.use_bias = None

        self.activation = None
        self.linear = None
        self.initialized = False

    def forward(self, embedding, out_planes, activation="relu", use_bias=True):
        """
        Arguments:
            embedding : input to the MLP (N,*,C)
            out_planes : total channels in the output
            activation : 'relu' or 'tanh'
            use_bias : True / False
        Returns:
            out : output of the MLP (N,*,out_planes)
        """
        self.in_planes = embedding.data.shape[-1]
        self.out_planes = out_planes
        self.use_bias = use_bias

        if not self.initialized:
            self.linear = nn.Linear(self.in_planes, self.out_planes, bias=use_bias).cuda()
            if activation == "relu":
                self.activation = nn.ReLU(inplace=True).cuda()
            elif activation == "tanh":
                self.activation = nn.Tanh().cuda()

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    print("initialize conditioning")
                    if self.use_bias:
                        nn.init.constant_(m.bias, 0.1)
            self.initialized = True

        out = self.linear(embedding)
        if self.activation is not None:
            out = self.activation(out)

        return out


class DynamicFCInitFirst(nn.Module):
    def __init__(self, in_planes, out_planes, use_bias=True):
        super(DynamicFCInitFirst, self).__init__()

        self.in_planes = (
            in_planes  # must be same shape as embedding.data.shape[-1]. i.e. to be fixed as it is.
        )
        self.out_planes = out_planes
        self.use_bias = use_bias
        self.activation = None  # TODO: pick proper choice
        self.linear = nn.Linear(self.in_planes, self.out_planes, bias=use_bias)

        self.init_weight()

    def init_weight(self):
        init_layer(self.linear)

    def forward(self, embedding, activation="relu"):
        """
        Arguments:
            embedding : input to the MLP (N,*,C)
            out_planes : total channels in the output
            activation : 'relu' or 'tanh'
            use_bias : True / False
        Returns:
            out : output of the MLP (N,*,out_planes)
        """
        # check why embedding (20 batch, 201 frames) structure is flattened.... this should be (20, 201, 88)
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (4020x88 and 8800x96)
        out = self.linear(embedding)

        if self.activation is not None:
            # normally set to None. TODO: why? Is this since condition generators are for affine transfomation before activation fxn?
            if activation == "relu":
                self.activation = nn.ReLU(inplace=True).cuda()
            elif activation == "tanh":
                self.activation = nn.Tanh().cuda()

            out = self.activation(out)

        return out

# ---------------------------------------------------------------------------------
# End of FiLM layer, start of UNet and submodules
# ---------------------------------------------------------------------------------
    
#TODO : refactor them to include them  
batchNorm_momentum = 0.1
num_instruments = 1

class Unet(nn.Module):
    def __init__(self, ds_ksize, ds_stride):
        super(Unet, self).__init__()
        if config.condition_check:
            self.encoder = ConditionedEncoder(ds_ksize, ds_stride)
        else: 
            self.encoder = Encoder(ds_ksize, ds_stride)
        self.decoder = Decoder(ds_ksize, ds_stride)

    def forward(self, x, score):
        if config.condition_check:
            x,s,c = self.encoder(x, score)
        else: 
            x,s,c = self.encoder(x)

        x = self.decoder(x,s,c)

        return x
    

class ConditionedEncoder(nn.Module):
    def __init__(self,ds_ksize, ds_stride):
        super(ConditionedEncoder, self).__init__()

        self.block1 = block(1,16,(3,3),(1,1),ds_ksize, ds_stride)
        self.block2 = block(16,32,(3,3),(1,1),ds_ksize, ds_stride)
        self.block3 = block(32,64,(3,3),(1,1),ds_ksize, ds_stride)
        self.block4 = block(64,128,(3,3),(1,1),ds_ksize, ds_stride)

        self.conv1 = nn.Conv2d(64,64, kernel_size=(3,3), padding=(1,1)) 
        self.conv2 = nn.Conv2d(32,32, kernel_size=(3,3), padding=(1,1)) 
        self.conv3 = nn.Conv2d(16,16, kernel_size=(3,3), padding=(1,1)) 

        self.score_block1 = score_encoder(1,16,(3,3),(1,1),ds_ksize, ds_stride)
        self.score_block2 = score_encoder(16,32,(3,3),(1,1),ds_ksize, ds_stride)
        self.score_block3 = score_encoder(32,64,(3,3),(1,1),ds_ksize, ds_stride)

        self.bg_gen1 = nn.Linear(44,2)
        self.bg_gen2 = nn.Linear(22,2)
        self.bg_gen3 = nn.Linear(11,2)

    def compress2param(self, x, fc_layer):

        """_summary_
        This function generates film parameters, gamma and beta,
        from encoded score information [B, C, encoded scoreH, encoded scoreW]  

        Returns:
            tensor:  
        """

        original_size = x.size()
        x = x.view(-1, original_size[-1])  # Flatten the last dimension
        x = fc_layer(x)
        beta_gamma_mat = x.view(*original_size[:-1], 2) 

        return beta_gamma_mat
    
    def film_operation(self, x, param):

        gamma = param[..., 0].unsqueeze(-1)
        beta = param[..., 1].unsqueeze(-1)
        # Broadcasting will automatically expand these tensors to shape [30, 16, 201, 88]
        x = gamma * x + beta

        return x


    def forward(self, x, score):

        score = torch.unsqueeze(score, 1)

        x1,idx1,s1 = self.block1(x)
        fp1, fp1_id, fp1_s = self.score_block1(score)
        beta_gamma_mat1 = self.compress2param(fp1, self.bg_gen1)
        x1 = self.film_operation(x1, beta_gamma_mat1)
        c3=self.conv3(x1) 

        x2,idx2,s2 = self.block2(x1)
        fp2, fp2_id, fp2_s = self.score_block2(fp1)
        beta_gamma_mat2 = self.compress2param(fp2, self.bg_gen2)
        x2 = self.film_operation(x2, beta_gamma_mat2)
        c2=self.conv2(x2) 

        x3,idx3,s3 = self.block3(x2)
        fp3, fp3_id, fp3_s = self.score_block3(fp2)
        beta_gamma_mat3 = self.compress2param(fp3, self.bg_gen3)
        x3 = self.film_operation(x3, beta_gamma_mat3)
        c1=self.conv1(x3)
        
        x4,idx4,s4 = self.block4(x3)
        
        return x4,[s1,s2,s3,s4],[c1,c2,c3,x1]

class Encoder(nn.Module):
    def __init__(self,ds_ksize, ds_stride):
        
        super(Encoder, self).__init__()

        self.block1 = block(1,16,(3,3),(1,1),ds_ksize, ds_stride)
        self.block2 = block(16,32,(3,3),(1,1),ds_ksize, ds_stride)
        self.block3 = block(32,64,(3,3),(1,1),ds_ksize, ds_stride)
        self.block4 = block(64,128,(3,3),(1,1),ds_ksize, ds_stride)

        self.conv1 = nn.Conv2d(64,64, kernel_size=(3,3), padding=(1,1)) 
        self.conv2 = nn.Conv2d(32,32, kernel_size=(3,3), padding=(1,1)) 
        self.conv3 = nn.Conv2d(16,16, kernel_size=(3,3), padding=(1,1)) 

    def forward(self, x):

        x1,idx1,s1 = self.block1(x)
        x2,idx2,s2 = self.block2(x1)
        x3,idx3,s3 = self.block3(x2)
        x4,idx4,s4 = self.block4(x3)
        
        c1=self.conv1(x3)
        c2=self.conv2(x2) 
        c3=self.conv3(x1) 
        return x4,[s1,s2,s3,s4],[c1,c2,c3,x1]


class score_encoder(nn.Module):
    def __init__(self, inp, out, ksize, pad, ds_ksize, ds_stride):
        super(score_encoder, self).__init__()
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, padding=pad)
        self.bn1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=ksize, padding=pad)
        self.bn2 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.skip = nn.Conv2d(inp, out, kernel_size=1, padding=0)
        self.ds = nn.Conv2d(out, out, kernel_size=ds_ksize, stride=ds_stride, padding=0)

    def forward(self, score):

        x11 = F.leaky_relu(self.bn1(self.conv1(score)))
        x12 = F.leaky_relu(self.bn2(self.conv2(x11)))
        x12 += self.skip(score)
        xp = self.ds(x12)
        # return [fp_block1], [fp_block2], [fp_block3],  
        return xp, xp, x12.size()



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))

        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        return x

class Decoder(nn.Module):

    def __init__(self,ds_ksize, ds_stride):
        super(Decoder, self).__init__()
        self.d_block1 = d_block(192,64,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block2 = d_block(96,32,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block3 = d_block(48,16,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block4 = d_block(16,num_instruments,True,(3,3),(1,1),ds_ksize, ds_stride)
            
    def forward(self, x, s, c=[None,None,None,None]):
        x = F.scaled_dot_product_attention(query=x, key=x, value=x, attn_mask=None)
        x = self.d_block1(x,s[3],False,c[0])
        x = F.scaled_dot_product_attention(query=x, key=x, value=x, attn_mask=None)
        x = self.d_block2(x,s[2],False,c[1])
        x = F.scaled_dot_product_attention(query=x, key=x, value=x, attn_mask=None)
        x = self.d_block3(x,s[1],False,c[2])
        x = F.scaled_dot_product_attention(query=x, key=x, value=x, attn_mask=None)
        x = self.d_block4(x,s[0],True,c[3])
        
        return torch.sigmoid(x)

class block(nn.Module):
    def __init__(self, inp, out, ksize, pad, ds_ksize, ds_stride):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, padding=pad)
        self.bn1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=ksize, padding=pad)
        self.bn2 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.skip = nn.Conv2d(inp, out, kernel_size=1, padding=0)
        self.ds = nn.Conv2d(out, out, kernel_size=ds_ksize, stride=ds_stride, padding=0)

    def forward(self, x):
        x11 = F.leaky_relu(self.bn1(self.conv1(x)))
        x12 = F.leaky_relu(self.bn2(self.conv2(x11)))
        x12 += self.skip(x)
        xp = self.ds(x12)
        return xp, xp, x12.size()

class d_block(nn.Module):
    def __init__(self, inp, out, isLast, ksize, pad, ds_ksize, ds_stride):
        super(d_block, self).__init__()
        self.conv2d = nn.ConvTranspose2d(inp, int(inp/2), kernel_size=ksize, padding=pad)
        self.bn2d = nn.BatchNorm2d(int(inp/2), momentum= batchNorm_momentum)
        self.conv1d = nn.ConvTranspose2d(int(inp/2), out, kernel_size=ksize, padding=pad)
        
        if not isLast: 
            self.bn1d = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
            self.us = nn.ConvTranspose2d(inp-out, inp-out, kernel_size=ds_ksize, stride=ds_stride) 
        else: 
            self.us = nn.ConvTranspose2d(inp, inp, kernel_size=ds_ksize, stride=ds_stride) 
            

    def forward(self, x, size=None, isLast=None, skip=None):
        x = self.us(x,output_size=size)
        
        if not isLast: 
            x = torch.cat((x, skip), 1) 
        x = F.leaky_relu(self.bn2d(self.conv2d(x)))

        if isLast: x = self.conv1d(x)
        else:  x = F.leaky_relu(self.bn1d(self.conv1d(x)))

        return x

class FinalDecoder(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(FinalDecoder, self).__init__()
        if config.spec_feat == "power":
            self.dim_adjust = nn.Linear(8192, midfeat, bias=False)
            self.bn_dim_adjust = nn.BatchNorm1d(midfeat, momentum=momentum)
        elif config.spec_feat == "mel":
            self.dim_adjust = nn.Linear(1792, midfeat, bias=False)
            self.bn_dim_adjust = nn.BatchNorm1d(midfeat, momentum=momentum)
        elif config.spec_feat in ("bark", "sone"):
            # Bark bands are small; after 4x (1,2) pooling, freq bins reduce to 1,
            # so input feature size becomes 128 * 1 = 128.
            # If you want to auto-infer the input dim for experiments:
            # self.dim_adjust = nn.LazyLinear(midfeat, bias=False)
            self.dim_adjust = nn.Linear(128, midfeat, bias=False)
            self.bn_dim_adjust = nn.BatchNorm1d(midfeat, momentum=momentum)

        self.fc = nn.Linear(midfeat, 768, bias=False)
        self.bn = nn.BatchNorm1d(768, momentum=momentum)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.finalfc = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.dim_adjust)
        init_bn(self.bn_dim_adjust)
        init_layer(self.fc)
        init_bn(self.bn)
        init_gru(self.gru)
        init_layer(self.finalfc)

    def forward(self, input):

        x = input.transpose(1, 2).flatten(2)

        x = F.relu(self.bn_dim_adjust(self.dim_adjust(x).transpose(1, 2)).transpose(1, 2))
        x=x.clone()
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)
        x = F.relu(self.bn(self.fc(x).transpose(1, 2)).transpose(1, 2))
        x=x.clone()
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)
        
        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.finalfc(x))

        return output

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class SpecEncoder(nn.Module):
    def __init__(self, momentum):
        super(SpecEncoder, self).__init__()

        self.conv_block1 = Conv(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = Conv(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = Conv(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = Conv(in_channels=96, out_channels=128, momentum=momentum)


    def forward(self, input):
        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        return x
    

class attn_d_block(nn.Module):
    
    def __init__(self, inp, out, isLast, ksize, pad, ds_ksize, ds_stride):
        super(attn_d_block, self).__init__()
        self.conv2d = nn.ConvTranspose2d(inp, int(inp/2), kernel_size=ksize, padding=pad)
        self.bn2d = nn.BatchNorm2d(int(inp/2), momentum= batchNorm_momentum)
        self.conv1d = nn.ConvTranspose2d(int(inp/2), out, kernel_size=ksize, padding=pad)
        
        if not isLast: 
            self.bn1d = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
            self.us = nn.ConvTranspose2d(inp-out, inp-out, kernel_size=ds_ksize, stride=ds_stride) 
        else: 
            self.us = nn.ConvTranspose2d(inp, inp, kernel_size=ds_ksize, stride=ds_stride) 


    def forward(self, x, size=None, isLast=None, skip=None):
        # print(f'x.shape={x.shape}')
        # print(f'target shape = {size}')
        x = self.us(x,output_size=size)
        
        if not isLast: 
            x = torch.cat((x, skip), 1) 
        x = F.leaky_relu(self.bn2d(self.conv2d(x)))

        if isLast: x = self.conv1d(x)
        else:  x = F.leaky_relu(self.bn1d(self.conv1d(x)))

        return x
