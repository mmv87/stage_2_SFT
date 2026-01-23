###Datapipeline for SFT.jsonl suitable for multi-variate and univariate timeseries
## Updated with the attention_mask (accounting for ts_token padding)
## dataloader to set the pipeline
### for the subset of the dataset
from torch.utils.data import Dataset,DataLoader
import torch
import json
device ='cuda' if torch.cuda.is_available() else 'cpu'

## Dataset class to get the pipeline for a sample
class ts_multimodal_text(Dataset):
    def __init__(self,patch_len,stride,file,tokenizer,device=device,model_dtype=None):
        ##self.data = data
        self.tokenizer = tokenizer
        self.mode_dtype=model_dtype
        self.device=device
        self.p=patch_len 
        self.s=stride
        self.file_path=file
        self.offsets=[]
        
        with open(self.file_path,'rb') as f:
            offset=0
            for line in f:
                self.offsets.append(offset)
                offset+=len(line)
    
    def __len__(self):
        return len(self.offsets)
        
    ## to patchify/sliding window operation of the ts_input
    ## the continuous stream of number is converted in to N,P
    def padding_stride(self,ts_list:list,max_size=0,p=256,s=256):
        x=torch.tensor(ts_list,dtype=torch.float32).to(self.device)
        x=x.view(1,-1)
        ##print(x.shape)
        l=x.shape[1]
        pattern = torch.tensor([0.0,1.0],device=self.device)
        ##print(l)
        ##special case when the length is smaller than the patch_len selected
        if (l<p):
            ##pad to the self.p or self.s
            local_padding=p-l
            pad_local=local_padding//2
            num_pad=pattern.repeat(pad_local).view(1,-1)
            x_pad=torch.cat([x,num_pad],axis=1)
        else:
            x_pad=x.clone()
            
        ##print(x.shape)  
        
        padding_0=max_size-x_pad.shape[1]
        ##print(f'global_padding:{padding_0}')
        r =(l-p)%s
        
        ###patchify the signal
        if (r==0):
            num_windows=(l-p)//s+1
        #print(f'num_windows: {num_windows}')
            pad_width=padding_0
            num_repeats = pad_width // 2
            pad = pattern.repeat(num_repeats).view(1,-1)  ## (1,pad_width)
            x_padded = torch.cat([x_pad,pad],axis=1)
            x_unfolded = x.unfold(1,self.p,self.s)
            ##print(f'patchify_ts:{x_unfolded.shape}')
            return x_unfolded  ## (bs,N,P)

        else:
            num_windows=(l-p)//s+2
            #print(f'num_windows: {num_windows}')
            pad_width=(s-r)+padding_0
            ##pattern = torch.tensor([0.0,1.0],device=self.device)
            num_repeats = pad_width // 2
            pad = pattern.repeat(num_repeats).view(1,-1)  ## (1,pad_width)
            x_padded = torch.cat([x_pad,pad],axis=1)
            ##x_padded = torch.nn.functional.pad(x,(0,pad_width),mode='constant')
            x_unfolded = x_padded.unfold(1,self.p,self.s)
            ##print(f'patchify_ts:{x_unfolded.shape}')
            return x_unfolded

    ##def derive_ts_mask():
        
    def parse_extract_ts_boundary(self,prompt):
        tokenized= self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        input_ids= tokenized['input_ids'][0]
        ts_start_token=self.tokenizer.convert_tokens_to_ids('<ts>')
        ts_end_token=self.tokenizer.convert_tokens_to_ids('<ts/>')
        ts_position=[]
    
        ##data structure to save the <ts>,<ts/> tokens ,list of tuples
        for i,token_id in enumerate(input_ids):
            if (token_id==ts_start_token):
                ts_position.append(('start',i))
            elif (token_id==ts_end_token):
                ts_position.append(('end',i))
                
        stack =[]
        ts_pairs=[]
        
        for j in range(len(ts_position)):
            pos,idx = ts_position[j]
            if pos=='start':
                stack.append(idx)
            elif stack and pos=='end':
                start=stack.pop(0)
                ts_pairs.append((start,idx))

        return ts_pairs,input_ids
        
    def __getitem__(self,idx):
        
        # 2. Fetch the specific line from disk
        with open(self.file_path, 'rb') as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            sample = json.loads(line)
            
        ###sample=self.data[idx]
        prompt=sample['input']
        output=sample['output']
        timeseries=sample['timeseries']
        ##min_len = min(len(c) for c in timeseries)
        # Truncate all channels to the shortest one
        ##timeseries_trunc = [c[:min_len] for c in timeseries]
        ##print(len(timeseries))
        ts_inputs=[]
        ch_padded=[]
        ts_mask=[]
        ts_mask_spl=[]
        output_ids=self.tokenizer(output,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
        ts_pairs,prompt_ids=self.parse_extract_ts_boundary(prompt)
        combined_input_ids=torch.cat([prompt_ids,output_ids],dim=0)  ## input + output tokens
        ts_metrics=len(ts_pairs)
        ##print(f'num_channel:{ts_metrics}')
        len_ts_data = [len(i) for i in timeseries]
        ##print(len_ts_data)
        assert ts_metrics==len(len_ts_data),f'the number of timeseries not same size as the ts_placeholders'
        ##print(f'channel_wise_ts:{len_ts_data}')
        max_len=max(len_ts_data)

        ## loop through the channels to patch and add to the list of tensors
        for i in range(len(len_ts_data)):
            ts_patched =self.padding_stride(timeseries[i],max_size=max_len,p=self.p,s=self.s)
            ts_patched=ts_patched.squeeze(dim=0)
            ts_inputs.append(ts_patched)
            ts_mask_channel=torch.cat([torch.ones(ts_patched.shape[0],dtype=torch.long),torch.zeros(10-ts_patched.shape[0],dtype=torch.long)])
            ts_mask.append(ts_mask_channel)
            
        assert len(ts_inputs)==len(ts_pairs),f'ts_input size not matching with the ts_pairs'
        ##normalize the patches to max(patches)
        patch_shapes=[ch.shape[0] for ch in ts_inputs]
        mean=sum(patch_shapes)/len(patch_shapes)
        max_N=max(patch_shapes) ##max of the patches
        ##print(max_N)
        if(max_N!=mean)and(ts_metrics>1):
            ##pad to max_patches (N) at the sample level
            patch_matching=torch.tensor([0.0,1.0],device=self.device)
            for i,ch_ts in enumerate(ts_inputs):
                ts_padding_len=ch_ts.shape[1]*(max_N-ch_ts.shape[0])
                ch_pad =patch_matching.repeat(ts_padding_len//2).view((max_N-ch_ts.shape[0]),self.p)
                new_ts=torch.cat([ch_ts,ch_pad],axis=0)
                ch_padded.append(new_ts)
                ts_mask_channel=torch.cat([torch.ones(max_N,dtype=torch.long),torch.zeros(10-max_N,dtype=torch.long)])
                ts_mask_spl.append(ts_mask_channel)
                
            assert len(ch_padded)==len(ts_pairs),f'ts_input size not matching with the ts_pairs'
                ##torch.ones(
                
            return{'input_ids':combined_input_ids,
                'output_ids':output_ids,
                'ts_inputs':torch.stack(ch_padded,dim=1),  ##stacking along the dim=1 (N,C_in,P)
                'ts_pairs':ts_pairs, ## in the multi-variate data
                 'patch_mismatch':True,
                  'ts_mask':torch.stack(ts_mask_spl,dim=0)}
        else:
            return{'input_ids':combined_input_ids,
                'output_ids':output_ids,
                'ts_inputs':torch.stack(ts_inputs,dim=1), ##stacking along the dim=1 (N,C_in,P)
                'ts_pairs':ts_pairs,      ##list of ts_data(tensors) of size (1,N_i,P)
                'patch_mismatch':False,
                'ts_mask':torch.stack(ts_mask,dim=0)}  

## applies transformation on the individual samples in a batch
##padding along the channel dimensions to max_channel length 10
##assembled token(ts_tokens+ ts_tokens)
##ts_tokens=c_in*max_N_patches

import torch.nn.functional as F
### masking along the channel axis , to slice the ts_embeddings after TS_encoder.
def mask(actual_in:torch.Tensor,device=device):
    ##c_in,N,P=ts_input.shape
    bs=1
    embed_dim=3072
    c_max=20
    N=10
    batched_c_in=actual_in.view(bs,1,1,1)
    channel_range = torch.arange(c_max).view(1,c_max,1,1) #bs,c_max,N,P
    mask = channel_range.to(device) < batched_c_in.to(device)
    token_mask=mask.expand(-1,-1,N,embed_dim)
    
    return token_mask

##ch_mask used to filter the actual_channels in TS_encoder input embedding layer
def filter_ch_idx(batch_size,c_in:torch.tensor):
    c_max=20
    actual_cin=c_in.view(-1,1)
    idx = torch.arange(c_max).expand(batch_size,-1).view(batch_size,-1)
    mask_attention_channel=idx < actual_cin

    return mask_attention_channel

## to assemble the attention mask [ts_token + textual token]
def assemble_attn_mask(ts_pairs:torch.tensor,ts_mask,c_in,total_tokens_count,max_N=10):
    ##displace the (start,end) based on inserted #tokens
    c_in_tensor=torch.arange(c_in).view(-1,1)
    ts_pair_tensor=ts_pairs
    new_ts_pair=ts_pair_tensor+(c_in_tensor*max_N)
    new_ts_pair[:,1] += max_N
    ## new indices based on the displaced indices
    local_indices= torch.arange(max_N).repeat(c_in, 1)
    new_starts = new_ts_pair[:,0] + 1
    final_ts_indices = ((new_starts.unsqueeze(1)) + local_indices).view(-1)
    assert torch.max(final_ts_indices)<=total_tokens_count,f'Max_of_ts_index:{torch.max(final_ts_indices)},token:{total_tokens_count}'
    ## to get the text_mask tokens
    ##input_tokens =torch.arange(total_tokens_count,dtype=torch.long)
    is_ts_new=torch.zeros(total_tokens_count, dtype=torch.bool)
    is_ts_new[final_ts_indices]=True
    new_text_indices = torch.nonzero(~is_ts_new).squeeze()
    ##ts_mask=torch.ones(new_text_indices.shape,dtype=torch.long) ### the shape textual tokens (inferred input+output_ids)
    attention_mask_container=torch.zeros(total_tokens_count,dtype=torch.long) ## final shape of input_ids+ output_ids + ts_tokens inferred (c_in*max_N+(input_ids+ outputids))
    attention_mask = attention_mask_container.scatter(0,final_ts_indices,ts_mask.flatten())
    textual_mask=torch.ones(new_text_indices.shape,dtype=torch.long)
    attention_mask=attention_mask.scatter(0,new_text_indices,textual_mask)
    
    return attention_mask

### to add padding for for ts_data
def collate_func(batch,tokenizer=None,device=device):
    input_ids = [x['input_ids'] for x in batch] ##(input+output_ids)
    output_ids=[x['output_ids'] for x in batch] ### output -answer for a question
    ts_data=[x['ts_inputs'] for x in batch]  ###(N,C_in,P) list of tensors of shape(patches,n_vars,patch_len) for batch on inputs
    channels=torch.tensor([x['ts_inputs'].shape[1] for x in batch]) 
    actual_cin=[x['ts_inputs'].shape[1] for x in batch][0]
    ts_patches_len=[x['ts_inputs'].shape[0] for x in batch] # Accessing shape from the actual tensor for univariate case
    ts_positions=[torch.tensor(x['ts_pairs']) for x in batch] ### list of tuples converted into torch.tensor
    ts_local_mask=[x['ts_mask'] for x in batch] ### list of tensors of shape (c_in,max_N) ## ts_localmask
    ts_input_item=ts_data[0]
    
   ## setting the max_patch_length for ts_tokens
    max_n_per_batch=10 ### maximum number of patches
    max_channel_dim=20  ### maximum channels 
    padded_ts_data=[]
    labels_batch=[]
    attention_mask_batch=[]
    ##padding the times series in the batch to max_N and max_channels
    ## padding has to happen along the dim=0 with the padding values [0.0,1.0]
    for i,ts_input_sample in enumerate(ts_data):
        padded_patch_len=max_n_per_batch-ts_input_sample.shape[0]
        patch_len=ts_input_sample.shape[2]
        c_in=ts_input_sample.shape[1]
        ts_padding_len= padded_patch_len*patch_len*c_in
        pattern = torch.tensor([0.0,1.0]).to(device)
        num_repeats = ts_padding_len // 2
        pad = pattern.repeat(num_repeats)
        ##converting the ts_input = <ts_tokens>+<padded_ts_token>
        padded_ts_token=pad.view(-1,c_in,patch_len) ###broadcast to (N,C,P)
        padded_ts_data.append(torch.cat([ts_input_sample.to(device),padded_ts_token.to(device)],dim=0)) ##to concatenate the padded block along dim=0

    ##pad along the channel dimension
    x=torch.stack(padded_ts_data) ### introduces the new axis along the batch.
    pad_channel=max_channel_dim-c_in
    pad=(0,0,0,pad_channel)
    ts_padded_channel = F.pad(x, pad, "constant", 0.0)
    
  ## N_i of batch of samples after padding
    ts_patch_padded_len=[x.shape[1] for x in padded_ts_data]
    max_text_len=max([x.size(0) for x in input_ids])
    max_ts_len=max(ts_patches_len)
    ts_seq_len = [seq.size(0) for seq in input_ids]
    tot_len=[(x+y) for x,y in zip(ts_patch_padded_len,ts_seq_len)]
    max_len_batch=max(tot_len)
    
##treat batch size of 1 as special case
    if len(batch)==1:
        input_ids_padded=input_ids[0].unsqueeze(0)    ##print(f'textual_shape {input_ids_padded.shape}')
        output_len=output_ids[0].shape[0]
        output_start_index = input_ids[0].shape[0] - output_len
        ##print(actual_cin)
        ##labels = torch.full((ts_seq_len[0],),-100,dtype=torch.long,device=device)
        ##labels[-output_len:] = output_ids[0]
        
        ##ts_mask for channel dimension
        ts_token=mask(channels,device=device)
        ch_mask=filter_ch_idx(1,channels)
        ## create a list of output_ids with no no_loss tokens
        labels_batch.append(output_ids)

        ##attention_mask for the sequence of inputs
        textual_tokens=input_ids[0].shape[0]
        ts_tokens=(max_n_per_batch*ts_data[0].shape[1])
        total_tokens=textual_tokens+ts_tokens
        attention_mask = assemble_attn_mask(ts_positions[0],ts_local_mask[0],actual_cin,total_tokens,max_N=10)
        
        ##attention_mask=torch.cat([torch.ones(channels[0]*max_n_per_batch,dtype=torch.long,device=device),torch.ones(ts_seq_len[0],dtype=torch.long,device=device)])
        attention_mask_batch.append(attention_mask)

        return {
            'input_ids':input_ids_padded,
            "labels":torch.stack(output_ids),
            'attention_mask':torch.stack(attention_mask_batch),
            "time_seried_padded":ts_padded_channel,
            'ts_mask':ts_token,
            'ch_mask':ch_mask.to(device),
            "time_series":torch.stack(padded_ts_data),
            'ts_pairs':torch.stack(ts_positions).to(dtype=torch.long)} ##list of tensor (N,C_in,P)}

    else:
        input_ids_padded= torch.stack([torch.cat([torch.full(((max_len_batch-seq.size(0)),),tokenizer.pad_token_id,dtype=seq.dtype),seq]) for seq in input_ids])

  ##max_len_batch=input_ids_padded.shape[1] # Correctepl=d to use shape[1] for sequence length
  ###max_N_per_batch=max(ts_data[])
  
    for i,sample in enumerate(batch):
        labels = torch.full((max_len_batch,),-100,dtype=torch.long,device=device)
        combined_len = sample['input_ids'].shape[0] + sample['ts_inputs'][0].shape[1] # Assuming one ts input per sample for simplicity
        pad_len = max_len_batch - combined_len
        seq_len=sample['input_ids'].shape[0]
        output_len=sample['output_ids'].shape[0]
        # Adjust label assignment based on padding at the beginning and TS embeddings
        # The labels correspond to the output_ids, which are at the end of the combined sequence
        # Calculate the starting index for output_ids in the padded label tensor
        output_start_index = max_len_batch - output_len
        labels[-output_len:] = sample['output_ids']
        labels_batch.append(labels)

        # Adjust attention mask based on padding and TS embeddings
        attention_mask=torch.cat([torch.zeros(pad_len,dtype=torch.long,device=device),torch.ones(sample['ts_inputs'][0].shape[1],dtype=torch.long,device=device),
                                torch.ones(seq_len,dtype=torch.long,device=device)
                                    ]) # Assuming one ts input
        attention_mask_batch.append(attention_mask)

  ##return the batch of input_ids , labels and timeseries
    return{
        'input_ids':input_ids_padded,
        "labels":torch.stack(labels_batch),
        'attention_mask':torch.stack(attention_mask_batch),
        "time_series":torch.cat(padded_ts_data,dim=0)} ##list of tensor (bs,max_N,Patch_len)