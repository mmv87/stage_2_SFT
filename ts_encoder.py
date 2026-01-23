### TS_encoder class that can be used to instantiate in main llm_wrapper
## ts_encoder with the position embeddings for local channel
import torch
import torch.nn as nn  
class ts_encoder_mlp(nn.Module):
    def __init__(self,max_patches,max_channel,patch_length,d_model,device=None):
        super(ts_encoder_mlp,self).__init__()
        self.max_patches=max_patches
        self.max_channel=max_channel
        self.d_model=d_model
        self.d_ff=2*d_model
        self.patch_length=patch_length
        ##self.shared_embeding=shared_embeding
        self.pe_feature=patch_length
        self.meta_feature_ts=8
        self.meta_feature_ch=3
        self.device=device
        
        ## positional encoding for the local timesteps/patch_len and the channel's dimension
        self.ts_pos=nn.Embedding(self.pe_feature,self.meta_feature_ts)
        self.ch_pos = nn.Embedding(self.max_channel+1,self.meta_feature_ch,padding_idx=self.max_channel)

        self.W_p=nn.Sequential(
            nn.Linear((self.patch_length+self.patch_length*self.meta_feature_ts+self.patch_length*self.meta_feature_ch),self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff,self.d_model)
        )
    
    def forward(self,x,ch_mask):
        bs,N,max_ch,p = x.shape
        x_reshaped=x.unsqueeze(-1)  ## (bs,N,c_in,p,1)
        ts_pos_embeds=self.ts_pos(torch.arange(p).to(x.device))
        ts_pos_embeds=ts_pos_embeds.expand(bs,N,max_ch,p,self.meta_feature_ts)

        ##filtered_ch_idx=self.filter_ch_idx(bs,c_in)
        idx = torch.arange(self.max_channel).expand(bs,-1).view(bs,-1)
        filtered_idx = torch.where(ch_mask,idx.to(self.device), self.max_channel)
        ch_pos_embeds=self.ch_pos(filtered_idx)
        ch_pos_embeds=ch_pos_embeds.unsqueeze(1).unsqueeze(-2)  ## [1,1,c_in,1,ts_embed]
        ch_pos_embeds=ch_pos_embeds.expand(-1,N,max_ch,p,self.meta_feature_ch)
        ##print(ch_pos_embeds.shape)
        
        ts_plus_embed = torch.cat([x_reshaped, ts_pos_embeds, ch_pos_embeds], dim=-1)
        ##print(ts_plus_embed.shape)
        x_reshaped = ts_plus_embed.view(bs,N,self.max_channel,-1)
        
        z = self.W_p(x_reshaped)
        ##print(f'z.shape before return: {z.shape}')
        return z.view(bs,max_ch,N,-1)  ## (bs,N,c_in,d_model)

