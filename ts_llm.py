##main ts_llm.py file that implements the 
### 1. Wrapper class , instantiation of classes (like dataset, dataloader, ts_encoder ,)
### 2. training routine , optimizer , calling the .pth file pretrained 
### 3. Monitoring/ saving

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model 
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
from TS_encoder import PatchTSTEncoder
import sys
import os
import json
from torch.utils.data import Dataset,DataLoader
from ts_dataloader import ts_multimodal_text,collate_func 
from ts_encoder import ts_encoder_mlp
import matplotlib.pyplot as plt
import matplotlib

device ='cuda' if torch.cuda.is_available() else 'cpu'
##print(device)
###print("Is CUDA available? ", torch.cuda.is_available())

"""model_name="/home/mmk/projects/def-zonata/mmk/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/7a8c4e2e81eae20a606d811f475d7dc316dd916a"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"""

model_name='microsoft/Phi-4-mini-reasoning'
model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True)
tokenizer =AutoTokenizer.from_pretrained(model_name,local_files_only=True)
print('model_loaded')
tokenizer_path =os.path.join(os.environ["SLURM_TMPDIR"],'llm_tokenizer')
tokenizer_modified = AutoTokenizer.from_pretrained(tokenizer_path)
print('tokenizer_loaded')

device ='cuda' if torch.cuda.is_available() else 'cpu'
model_dtype=next(model.parameters()).dtype

###load the dataset from .jsonl file
_json_path=os.path.join(os.environ["SLURM_TMPDIR"],"processed_dataset.jsonl")

def dataset_align(file):
    list_data=[]
    with open(file,'r',encoding='utf-8') as file:
        for idx,line in enumerate(file):
            if idx<1000:
                obj= json.loads(line)
                list_data.append(obj)
            else:
                break

    return list_data

sft_dataset = dataset_align(_json_path)

dataset=ts_multimodal_text(256,256,sft_dataset,tokenizer_modified,device=device,model_dtype=None)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer_modified,device=device))

##Lora_config defintion based on best practices
peft_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "v_proj","k_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
            modules_to_save=["embed_tokens",'projection'],lora_dropout=0.1, # Very important for Stage-2
            task_type="CAUSAL_LM",ensure_weight_tying=True)

#####################################

class LLM_wrapper(nn.Module):
    def __init__(self,tokenizer,max_patches,max_channel,patch_len,llm_model,device=device,ts_checkpoint=None,embed_path=None,peft_config=None):
        super().__init__()
        self.tokenizer=tokenizer
        self.llm_model=llm_model
        self.embed_size=llm_model.config.hidden_size
        self.max_patches=max_patches
        self.max_channel=max_channel
        self.P=patch_len
        self.device=device
        ##resize the llm_embedding layer based on modified tokenizer
        self.llm_model.resize_token_embeddings(len(self.tokenizer))
        
        if embed_path:
            self.llm_model.get_input_embeddings().load_state_dict(torch.load(embed_path))
            
        ###creating peft model for stage-2 training
        if peft_config:
            self.peft_model=get_peft_model(self.llm_model,peft_config)
        
        ##loading the input_embedding from stage-1
        
        ##initialise the ts_encoder
        self.ts_encoder=ts_encoder_mlp(self.max_patches,self.max_channel,self.P,self.embed_size,device=self.device)
        ts_enc_state_dict = torch.load(ts_checkpoint, map_location=self.device)
        self.ts_encoder.load_state_dict(ts_enc_state_dict,strict=False)
            
        ##self.ts_encoder=PatchTSTEncoder(c_in=self.max_channel,num_patch=self.max_patches,patch_len=self.P,n_layers=2,d_model=llm_model.config.hidden_size,n_heads=4,shared_embedding=False,d_ff=2*256,
        ##  norm='BatchNorm',attn_dropout=0,dropout=0.1,activation='gelu',store_attn=True,res_attention=False,pre_norm=False,pe='zeros',learn_pe=True,verbose=True)
        
        self.ts_encoder.to(self.device)
        

    def assemble_input_embeds(self,input_ids,ts_embeddings,ts_pairs:torch.tensor,labels:torch.tensor):
        assemb_embed_tensor=[]
        attention_mask_list=[]
        labels_list=[]
        assemb_labels=[]
        channels=ts_pairs.shape[1]
        ##print(f'sliced_ts_embedding:{ts_embeddings.shape}')
        ts_embeddings=ts_embeddings.view(-1,channels,10,self.embed_size)
        ts_embeddings.to(self.device)
        bs=ts_embeddings.shape[0]
        c_in=ts_embeddings.shape[1]
        num_ts_tokens=ts_embeddings.shape[2]
        ts_emb_dim=ts_embeddings.shape[3]
    
        input_embeds=self.llm_model.get_input_embeddings()(input_ids) ##seq_len,d_emb
        input_embeds.requires_grad_(requires_grad=True)
        text_emb_dim= input_embeds.shape[2]
        assert (ts_emb_dim==text_emb_dim)
    ##inplace operation to adjust ts_pairs 
        tokens_per_ts_inst = c_in * num_ts_tokens
    
        ts_pairs=ts_pairs.squeeze(0)
        instance_indices = torch.arange(c_in, dtype=ts_pairs.dtype, device=self.device)
        cumulative_offsets = instance_indices * num_ts_tokens
        offset_expanded = cumulative_offsets.unsqueeze(1).expand(-1,2)

        ts_pairs_displaced=ts_pairs.to(device) + offset_expanded
        ts_pairs_displaced[:, 1] += num_ts_tokens
        flat_ts_embeddings=ts_embeddings.view(-1,c_in*num_ts_tokens,ts_emb_dim)
    
        if bs==1:
            flat_ts_embeddings=flat_ts_embeddings.squeeze(0) ##[c_in*ts_tokens,emb_dim]
            input_embeds=input_embeds.squeeze(0)

        ##total_seq_len = text_tokens + ts_tokens
        T_new = input_embeds.shape[0]+ c_in*num_ts_tokens
        local_indices= torch.arange(num_ts_tokens,device=device).repeat(c_in, 1)
    
        new_starts = ts_pairs_displaced[:,0] + 1
        new_starts.to(device)
        final_ts_indices = ((new_starts.unsqueeze(1).to(self.device)) + local_indices.to(self.device)).view(-1)
        is_ts_new=torch.zeros(T_new, dtype=torch.bool, device=self.device)
        is_ts_new[final_ts_indices]=True
        
        new_text_indices = torch.nonzero(~is_ts_new).squeeze()

        text_container_new=torch.zeros((T_new,text_emb_dim),device=device)
        ts_container_new=torch.zeros((T_new,ts_emb_dim),device=device)
        ts_scatter_idx=final_ts_indices.unsqueeze(1).expand(-1,ts_emb_dim)
        text_scatter_idx = new_text_indices.unsqueeze(1).expand(-1,text_emb_dim)

        text_container_new=text_container_new.scatter(dim=0,index=text_scatter_idx.to(device),src=input_embeds)
        ts_container_new=ts_container_new.scatter(dim=0,index=ts_scatter_idx.to(device),src=flat_ts_embeddings)
       
        final_embeds = text_container_new + ts_container_new
        
        assemb_embed_tensor.append(final_embeds)
        seq_len=final_embeds.shape[0]
        ##attention_mask=torch.ones((seq_len,),dtype=torch.long,device=self.device)
        ##attention_mask_list.append(attention_mask)

        ##return labels.
        loss_tokens=labels.squeeze(0).to(self.device)
        no_losstokens=torch.full((seq_len-labels.shape[1],),-100.0,dtype=torch.long,device=self.device)
        assemb_labels.append(torch.cat([no_losstokens,loss_tokens]))
        
        return torch.stack(assemb_embed_tensor),torch.stack(assemb_labels,dim=0) ### two variable the input_embed, labels
        

    def forward(self,input_ids=None,ts_input=None,ts_pairs=None,ts_masks=None,ch_mask=None,attention_mask=None,labels=None):
        ##convert the ts_patches into ts_embeddings
        ts_tensor = ts_input.view(-1,self.max_patches,self.max_channel,self.P).to(self.device)  ## (bs,N,c_in,P)
        ts_embedding = self.ts_encoder(ts_tensor.to(self.device),ch_mask) ## (bs,n_vars,num_patch,d_model)
        ts_embedding_sliced =ts_embedding[ts_masks] ##flattened ts_embeddings
        ## two variables returned by assemb_embeds
        input_embeddings,lable_batch = self.assemble_input_embeds(input_ids,ts_embedding_sliced,ts_pairs,labels)
        
      ###input_embeddings=input_embeddings.squeeze(0)
        ##attention_mask = attentionmask_batch.to(self.device)
        ##print(f'attn_mask_shape:{attention_mask}')
        labels = lable_batch.to(self.device)
        ##print(f'labels)
        output=self.peft_model(inputs_embeds=input_embeddings,attention_mask=attention_mask,labels=lable_batch)
        
        return output,input_embeddings,ts_embedding_sliced

### loading the pretrained .pth for transfer learning
## load the pre-trained weights
ts_encoder_weights=os.path.join(os.environ["SLURM_TMPDIR"],'ts_encoder_warmup_stage1_multivar.pth')
embed_path=os.path.join(os.environ["SLURM_TMPDIR"],'aligned_embeddings.pt')

from tqdm import tqdm

model_wrapper=LLM_wrapper(tokenizer_modified,10,20,256,model,device=device,ts_checkpoint=ts_encoder_weights,embed_path=embed_path,peft_config=peft_config)
model_wrapper.train()
model_wrapper.to(device)

for name, param in model_wrapper.ts_encoder.named_parameters():
    param.requires_grad = True
    
"""for p in model_wrapper.llm_model.get_input_embeddings().parameters():
    p.requires_grad = True
   for p in model_wrapper.peft_model.parameters():
   p.requires_grad=True"""

encoder_params = list(model_wrapper.ts_encoder.parameters())
llm_trainable_params = [p for n,p in model_wrapper.peft_model.named_parameters() if p.requires_grad]

optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': 1e-4},      # Physics: Fast learning
    {'params': llm_trainable_params, 'lr': 5e-5}])

epoch_losses=[]

for epoch in range(1):  ##1 epochs
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    num_batches = 0
    running_loss=0
    epoch_loss=0
    for batch in pbar:
        input_ids=batch['input_ids'].to(device) ## input and output
        ts_input=batch['time_seried_padded'].to(device) ### batch of patchified padded ts_inputs (bs,c_in,N,p)
        ##print(ts_input.shape)
        ts_pairs=batch['ts_pairs']
        labels_batch=batch['labels'].to(device)
        ts_mask = batch['ts_mask'].to(device)
        ch_mask=batch['ch_mask'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        
      ##model_wrapper=LLM_wrapper(tokenizer,ts_input,model,device=device)
        outputs,inputs,ts_embeds =model_wrapper(input_ids=input_ids,ts_input=ts_input,ts_pairs=ts_pairs,ts_masks=ts_mask,ch_mask=ch_mask,attention_mask=attention_mask,labels=labels_batch)
        loss=outputs.loss
        
        loss.backward()  ##gradient calculation
        ###check_ts_gradients(model_wrapper.ts_encoder)
        ##track_gradients(ts_encoder)
        running_loss+=loss.item()
        num_batches+=1
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix(loss=loss.item())
        epoch_loss=running_loss/num_batches
        epoch_losses.append(epoch_loss)
        ##print(num_batches)
        
        """
        print(name, param.grad is not None)"""
      ##to track the gradients of the TS-Encoder
    

saved_file=os.path.join(os.environ["SLURM_TMPDIR"],'ts_encoder_stage3.pth')
torch.save(model_wrapper.ts_encoder.state_dict(),saved_file)
model_wrapper.peft_model.config.save_embedding_layers = True
model_wrapper.peft_model.save_pretrained(os.path.join(os.environ["SLURM_TMPDIR"],'phi4-ts-adapter_3'),save_embedding_layers=True)

##plot and save the fig
matplotlib.use('Agg')
plt.figure(figsize=(8, 10))
x=range(len(epoch_losses))
y=epoch_losses
plt.plot(x,epoch_losses)
out_path = os.path.join(os.environ["SLURM_TMPDIR"], "training_loss_stage2.png")
plt.savefig(out_path)
