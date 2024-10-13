import torch
import torch.nn as nn
import torch.nn.functional as F

import Mainframe.set_logging as set_logging
logger = set_logging.get_logger(__name__)

def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    m = nn.Linear(in_features, out_features, bias)
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)
    return m


def gru_cell(input_size, hidden_size, bias=True):
    m = nn.GRUCell(input_size, hidden_size, bias)
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    return m

class SlotAttention(nn.Module):

    def __init__(self, num_iterations, num_slots,
                 slot_size, mlp_hidden_size, heads,
                 epsilon=1e-8):
        super().__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.num_heads = heads

        self.norm_inputs = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(slot_size, slot_size, bias=False)
        self.project_v = linear(slot_size, slot_size, bias=False)
        
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))
    def forward(self, inputs, slots_init):

        B, N_kv, D_inp = inputs.size()
        B, N_q, D_slot = slots_init.size()

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)  
        v = self.project_v(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)  
        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k

        
        # Multiple rounds of attention.
        slots = slots_init
        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots).view(B, N_q, self.num_heads, -1).transpose(1, 2)  
            attn_logits = torch.matmul(k, q.transpose(-1, -2))  
            attn = F.softmax(attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q), dim=-1).view(B, N_kv, self.num_heads, N_q).transpose(1, 2)  
            attn_vis = attn.sum(1)  

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True) 
            updates = torch.matmul(attn.transpose(-1, -2),
                                   v)  
            updates = updates.transpose(1, 2).reshape(B, N_q, -1)  

            # Slot update.
            slots = self.gru(updates.reshape(-1, self.slot_size),
                             slots_prev.reshape(-1, self.slot_size))
            slots = slots.reshape(-1, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))
            
        return slots, attn.mean(1).permute(0, 2, 1)


class SlotAttentionEncoder(nn.Module):

    def __init__(self, num_iterations, num_slots, input_channels, slot_size, mlp_hidden_size, num_heads):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_channels = input_channels
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_heads = num_heads

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_channels),
            linear(input_channels, mlp_hidden_size),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))

        self.slot = nn.Parameter(torch.ones(1,num_slots, slot_size))
        nn.init.xavier_uniform_(self.slot)

        self.slot_attention = SlotAttention(num_iterations, num_slots, slot_size, mlp_hidden_size, num_heads)
        
    def forward(self, x, slots=None):
        B = x.size(0)
        x = self.mlp(x)

        # Slot Attention module.
        slots = self.slot.repeat_interleave(B,dim=0)
        slots, attn = self.slot_attention(x, slots)

        return slots, attn
    