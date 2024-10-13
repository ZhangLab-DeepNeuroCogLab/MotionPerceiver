import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from Module_Repository.slot_attention import SlotAttentionEncoder
from Video_Transformer.Model_Configs.dino import dino_vitb16
from Video_Transformer.Layers import coords_grid, get_1d_sincos_pos_embed_from_grid
from timm.models.vision_transformer import Block

import Mainframe.set_logging as set_logging
logger = set_logging.get_logger(__name__)

class MP(nn.Module):
    def __init__(self,cfg,mode,backbone,n_pretrain_classes,n_finetune_classes):
        super(MP, self).__init__()
        self.cfg = cfg
        
        #################################################################
        # Load Feature Extraction Backbone 
        ################################################################# 
        func = globals()[backbone]
        
        dino = func()
        
        for p in dino.parameters():
            p.requires_grad = False
        
        dino.eval()
        
        self.vit_encoder = dino

        self.dim=768
        
        self.NUM_FRAMES_LIST = cfg.MYMODEL.NUM_FRAMES_LIST
        
        self.TEMPORAL_SLOT_NUM_SLOTS =cfg.MYMODEL.TEMPORAL_SLOT.NUM_SLOTS
        
        self.TEMPORAL_CORR_TEMP = cfg.MYMODEL.TEMPORAL_CORR_TEMP
        
        self.time_embed = nn.Parameter(torch.zeros(1, 32, self.dim), requires_grad=False)
        
        self.temporal_slot = nn.ModuleList([SlotAttentionEncoder(num_iterations=cfg.MYMODEL.TEMPORAL_SLOT.NUM_ITER,
                                            num_slots=cfg.MYMODEL.TEMPORAL_SLOT.NUM_SLOTS[i%len(cfg.MYMODEL.NUM_FRAMES_LIST)],
                                            input_channels=cfg.MYMODEL.TEMPORAL_SLOT.INPUT_CHANNEL[i%len(cfg.MYMODEL.NUM_FRAMES_LIST)],
                                            slot_size=cfg.MYMODEL.TEMPORAL_SLOT.SLOT_SIZE[i%len(cfg.MYMODEL.NUM_FRAMES_LIST)],
                                            mlp_hidden_size=cfg.MYMODEL.TEMPORAL_SLOT.MLP_HIDDEN_SIZE[i%len(cfg.MYMODEL.NUM_FRAMES_LIST)],
                                            num_heads=cfg.MYMODEL.TEMPORAL_SLOT.NUM_HEAD)
                                            for i in range(len(cfg.MYMODEL.NUM_FRAMES_LIST))])                                 

        self.flow_proj = nn.ModuleList([nn.Sequential(
           nn.Linear(14*14,self.dim, bias=True),
        )  
        for _ in range(len(cfg.MYMODEL.NUM_FRAMES_LIST))])
           
        self.flow_combination_x_slots = nn.Sequential(
            Block(dim = self.dim, num_heads=8),
            Block(dim = self.dim, num_heads=8),
        )
       
        self.flow_combination_x_time = nn.Sequential(
            Block(dim = self.dim, num_heads=8),
            Block(dim = self.dim, num_heads=8),
        )
        
        self.invar_proj = nn.Sequential(
           nn.Linear(14*14,self.dim, bias=True),
        )
        
        self.invar_combination_x_axes = nn.Sequential(
            Block(dim = self.dim, num_heads=8),
            Block(dim = self.dim, num_heads=8),
        )
           
        self.invar_combination_x_time = nn.Sequential(
            Block(dim = self.dim, num_heads=8),
            Block(dim = self.dim, num_heads=8),
        )
        
        self.flow_head = nn.Linear(self.dim, n_finetune_classes, bias=True)
        
        self.invar_head = nn.Linear(self.dim, n_finetune_classes, bias=True)
        
        self.all_head = nn.Linear(2*self.dim, n_finetune_classes, bias=True)
        
        self.CELoss = nn.CrossEntropyLoss()
    
        self.initialize_weights()
    
    def initialize_weights(self):
        grid_num_frames = np.arange(32, dtype=np.float32)
        
        time_embed = get_1d_sincos_pos_embed_from_grid(self.time_embed.shape[-1], grid_num_frames)
        
        self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))
        
    def temporal_walk_loss(self,features,slot_idx):
        B = features.size(0)

        normalized_features = F.normalize(features, p=2, dim=-1) 
        
        slots, _ = self.temporal_slot[slot_idx](features) 
            
        k = slots.size(1)
        normalized_slots = F.normalize(slots, p=2, dim=-1) 
        ss_corr = torch.eye(k).expand(B, k, k).cuda() 
        
        fs_corr = torch.matmul(normalized_features, normalized_slots.permute(0, 2, 1)) 
        sf_corr = fs_corr.permute(0, 2, 1) 
        fs_corr = (fs_corr / self.TEMPORAL_CORR_TEMP).softmax(dim=-1)
        sf_corr = (sf_corr / self.TEMPORAL_CORR_TEMP).softmax(dim=-1)
        
        transition1 = torch.matmul(sf_corr, fs_corr) 
        wpw_loss = (torch.log(transition1 + 1e-4).flatten(0, 1) * ss_corr.flatten(0, 1) * (-1)).mean()
        
        return 10.0*wpw_loss,slots
      
                
    def dense_OF(self,tokens,tau,fs):
        B,T,N,E =  tokens.shape 
        H_t = W_t = int(math.sqrt(N))
        normalized_tokens = F.normalize(tokens, p=2, dim=-1)
        
        pos_embed = coords_grid(B,H_t,W_t).permute(0,2,3,1).to(tokens.device)
        pos_embed = pos_embed.view(B,H_t*W_t,2)
        pos_embed_OF = pos_embed.unsqueeze(1).repeat(1,fs,1,1) 

        frame_interval = T//fs
        for t in range(T):
            idx_group=[]
            num_frame_backward = t//frame_interval
            for f_idx in range(1,num_frame_backward+1):
                idx_group.append(t-f_idx*frame_interval)
            idx_group.reverse()
            idx_group.append(t)
            num_frame_forward = (T-1-t)//frame_interval
            for f_idx in range(1,num_frame_forward+1):
                idx_group.append(t+f_idx*frame_interval)
            
            temporal_tt_corr= torch.matmul(normalized_tokens[:,t].unsqueeze(1).repeat(1,fs,1,1),normalized_tokens[:,idx_group].permute(0,1,3,2)) 
            temporal_tt_prob_mask = (temporal_tt_corr/tau).softmax(dim=-1) 
            pos_new_embed = torch.matmul(temporal_tt_prob_mask,pos_embed_OF) 
            OF_t = pos_new_embed[:,1:]-pos_new_embed[:,:-1] 
            OF_t = OF_t.permute(0,2,1,3).reshape(B,H_t*W_t,-1).unsqueeze(1)
            if t ==0:
                OF = OF_t
            else:
                OF = torch.cat((OF,OF_t),dim=1) 
                
        OF = torch.where(torch.abs(OF)<0.2,0.0,OF)
        return OF+1e-6
    
    
    def motion_invar(self,tokens,tau):
        B,T,N,E =  tokens.shape 
        H_t = W_t = int(math.sqrt(N))
        normalized_tokens = F.normalize(tokens, p=2, dim=-1) 
        
        pos_embed = coords_grid(B,H_t,W_t).permute(0,2,3,1).to(tokens.device)
        pos_embed = pos_embed.view(B,H_t*W_t,2)
        pos_embed_OF = pos_embed.unsqueeze(1).repeat(1,T,1,1) 

        for t in range(T):
            temporal_tt_corr= torch.matmul(normalized_tokens[:,t:(t+1)].repeat(1,T,1,1),normalized_tokens.permute(0,1,3,2)) 
            temporal_tt_prob_mask = (temporal_tt_corr/tau).softmax(dim=-1) 
            pos_new_embed = torch.matmul(temporal_tt_prob_mask,pos_embed_OF) 
            motion = pos_new_embed[:,t:(t+1)]-pos_new_embed 
            motion_positive = torch.where(motion>0.0,motion,0.0) 
            motion_negative = torch.where(motion<0.0,motion,0.0) 
            motion_positive_scalar = torch.abs(motion_positive).mean(dim=1) 
            motion_negative_scalar = torch.abs(motion_negative).mean(dim=1) 
            motion_invar = torch.cat((motion_positive_scalar,motion_negative_scalar),dim=-1) 
            motion_invar = motion_invar.permute(0,2,1) 
            if t ==0:
                total_motion_invar = motion_invar.unsqueeze(1)
            else:
                total_motion_invar = torch.cat((total_motion_invar,motion_invar.unsqueeze(1)),dim=1) 

        return total_motion_invar
    
    def forward(self, frames, labels):
        B,C,T,H,W = frames.shape 
        aux_loss = 0.0
        
        for frame_idx in range(T):
            tokens = self.vit_encoder.forward_feats(frames[:,:,frame_idx])[:,1:].detach()
            if frame_idx==0:
                attn_tokens=tokens.unsqueeze(1)
            else:
                attn_tokens= torch.cat((attn_tokens,tokens.unsqueeze(1)),dim=1)
        _,_,N,E =  attn_tokens.shape 
        H_t = W_t = int(math.sqrt(N))


        for i in range(len(self.NUM_FRAMES_LIST)):
            ##########################################################################
            # Patch-level Optical Flow
            ##########################################################################
            attn_tokens_OF= self.dense_OF(attn_tokens.view(B,T,H_t*W_t,-1),tau=0.001,fs = self.NUM_FRAMES_LIST[i]) 
            normalized_temporal_feature = F.normalize(attn_tokens_OF, p=2, dim=-1) 
               
            ##########################################################################
            # Temporal Slots
            ##########################################################################
            aux_temporal_loss, temporal_slots=self.temporal_walk_loss(attn_tokens_OF.view(B,T*H_t*W_t,-1),i) 
            aux_loss +=  1.0*aux_temporal_loss
            
            ##########################################################################
            # Multi-scale Features Concatenation
            ##########################################################################
            normalized_temporal_slots = F.normalize(temporal_slots, p=2, dim=-1) 
            corr_tokens_temporal_slots = torch.matmul(normalized_temporal_feature.view(B,T*H_t*W_t,-1),normalized_temporal_slots.permute(0,2,1)) 
            corr_tokens_temporal_slots = corr_tokens_temporal_slots.view(B,T,H_t*W_t,-1) 

            flow_features = corr_tokens_temporal_slots.permute(0,3,1,2) 
            flow_features = self.flow_proj[i](flow_features) 

            if i == 0:
                flow_features_total = flow_features.permute(0,2,1,3).reshape(B*T,self.TEMPORAL_SLOT_NUM_SLOTS[i],-1) 
            else:
                flow_features_i = flow_features.permute(0,2,1,3).reshape(B*T,self.TEMPORAL_SLOT_NUM_SLOTS[i],-1) 
                flow_features_total = torch.cat((flow_features_total,flow_features_i),dim=1) 
           
        ##########################################################################
        # Flow Snapshot Feature Fusion and classification
        ##########################################################################     
        fused_flow_features_x_slots = self.flow_combination_x_slots(flow_features_total).mean(dim=1)  
        fused_flow_features_x_slots = fused_flow_features_x_slots.view(B,T,-1) 
        
        fused_flow_features_x_slots = fused_flow_features_x_slots+self.time_embed
        fused_flow_features = self.flow_combination_x_time(fused_flow_features_x_slots).mean(dim=1) 
        
        flow_logit = self.flow_head(fused_flow_features)
        aux_loss += self.CELoss(flow_logit,labels)
       
        ##########################################################################
        # Motion Invariance
        #########################################################################
        tokens_motion_map = self.motion_invar(attn_tokens.view(B,T,H_t*W_t,-1),tau=0.001)
        
        ##########################################################################
        # Motion Ivariance Feature Fusion and classification
        ##########################################################################  
        motion_invar_features = self.invar_proj(tokens_motion_map) 
        fused_motion_invar_features_x_axes = self.invar_combination_x_axes(motion_invar_features.view(B*T,4,-1)).mean(dim=1) 
        fused_motion_invar_features = self.invar_combination_x_time(fused_motion_invar_features_x_axes.view(B,T,-1)).mean(dim=1) 
        
        invar_logit = self.invar_head(fused_motion_invar_features)
        aux_loss += self.CELoss(invar_logit,labels)

        ##########################################################################
        # Two Branches Fusion and classification
        ##########################################################################  
        all_features = torch.cat((fused_flow_features,fused_motion_invar_features),dim=-1) 
        logit = self.all_head(all_features)
        
        return aux_loss,logit
        
