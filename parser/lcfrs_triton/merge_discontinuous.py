import pdb
import statistics
import torch
import triton
import triton.language as tl

from torch.utils.checkpoint import checkpoint as ckp





def checkpoint(func):
    def wrapper(*args, **kwargs):
        return ckp(func, *args, **kwargs)

    return wrapper
@triton.jit
def logaddexp(a, b):
    tmp = a - b
    return tl.where(tmp > 0, tl.log(tl.exp(b - a) + 1) + a, tl.log(tl.exp(a-b) + 1) + b)




@triton.jit
def _kernel_inside_merge_discontinuous_v1(
            alpha_c, 
            tmp_merge, tmp_merge_normalizer, 
            w, batch, L, 
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            stride_tmp_merge1, stride_tmp_merge2, stride_tmp_merge3,
            stride_normalizer1, stride_normalizer2,
            r1, r2, r3, r4,
            BLOCK_R3: tl.constexpr,
            ):
    
    ## Find index. tl.program_id(1) is of size w-1, each indicates the length of the left continuous subspan given a discontinuous parent.
    # for each tl.program_id(1), the number of possible discontinuous spans is the same:  len(tl.program_id(2)) := (L-w) + (L-w-1) + (L-w-2) + ... + 1 = (L-w)*(L-w+1)/2.  
    ## (L-w-i) parts means that the start position is $i$, and each j \in [0, L-w-i] means the gap length (gap end - gap start)
    ## To avoid the waste of half amount of computation, I manually compute the index in the following way
    b_idx = tl.program_id(0)


    if b_idx >= batch:
        return 

    span_length_left = tl.program_id(1) + 1
    tid = tl.program_id(2)
    start = 0            
    # To find the group (L-w-start).  tid is the gap length then        

    while tid >= (L-w-start):
        tid -= (L-w-start)
        start += 1 
    
    gap_start = start + span_length_left
    gap_end = gap_start + (tid + 1)    
    end = gap_end + (w - span_length_left)
    
    l_ptr = alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + gap_start * stride_alpha_c3 + 2*r1 + r2 + tl.arange(0, BLOCK_R3)
    r_ptr = alpha_c + b_idx * stride_alpha_c1 + gap_end * stride_alpha_c2 + end * stride_alpha_c3 + 2*r1 + r2 + r3 + tl.arange(0, BLOCK_R3)
    mask = tl.arange(0, BLOCK_R3) < r3
    child_l = tl.load(l_ptr, mask=mask, other=-1e9)
    child_r = tl.load(r_ptr, mask=mask, other=-1e9)
    acc1 = child_l + child_r


    acc_max = tl.max(acc1, 0)
    tl.store(tmp_merge_normalizer + b_idx * stride_normalizer1 + tl.program_id(1) * stride_normalizer2 + tl.program_id(2), acc_max)

    acc = tl.exp(acc1 - acc_max)
    tl.store(tmp_merge +   b_idx * stride_tmp_merge1 + tl.program_id(1) * stride_tmp_merge2 + tl.program_id(2) * stride_tmp_merge3 + tl.arange(0, BLOCK_R3), acc, mask=mask)



@triton.jit
def _kernel_bwd_merge_discontinuous_v1(
            alpha_c,
            tmp_merge_normalized, tmp_merge_grad, 
            w, batch, L, 
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            stride_tmp_merge1, stride_tmp_merge2, stride_tmp_merge3,
            r1, r2, r3, r4,
            BLOCK_R3: tl.constexpr,
            BLOCK_R4: tl.constexpr,
            ):
    
    ## Find index. tl.program_id(1) is of size w-1, each indicates the length of the left continuous subspan given a discontinuous parent.
    # for each tl.program_id(1), the number of possible discontinuous spans is the same:  len(tl.program_id(2)) := (L-w) + (L-w-1) + (L-w-2) + ... + 1 = (L-w)*(L-w+1)/2.  
    ## (L-w-i) parts means that the start position is $i$, and each j \in [0, L-w-i] means the gap length (gap end - gap start)
    ## To avoid the waste of half amount of computation, I manually compute the index in the following way
    b_idx = tl.program_id(0)
    if b_idx >= batch:
        return 
    span_length_left = tl.program_id(1) + 1
    tid = tl.program_id(2)
    start = 0            
    # To find the group (L-w-start).  tid is the gap length then        

    while tid >= (L-w-start):
        tid -= (L-w-start)
        start += 1 
    
    gap_start = start + span_length_left
    gap_end = gap_start + (tid + 1)    
    end = gap_end + (w - span_length_left)
        
    # acc3 = tl.zeros((BLOCK_R4,), dtype=tl.float32) - 1e9
    # acc4 = tl.zeros((BLOCK_R4,),dtype=tl.float32) - 1e9
    # acc5 = tl.zeros((BLOCK_R4,),dtype=tl.float32) - 1e9
    ## discontinuous parent nodes with two continuous child nodes
    # [i, j], [m, n] -> [i, j, m, n] 

    l_bwd_ptr = alpha_c + b_idx * stride_alpha_c1 + gap_start * stride_alpha_c2 + start * stride_alpha_c3 + 2*r1 + r2 + tl.arange(0, BLOCK_R3)
    r_bwd_ptr = alpha_c + b_idx * stride_alpha_c1 + end * stride_alpha_c2 + gap_end * stride_alpha_c3 + 2*r1 + r2 + r3 + tl.arange(0, BLOCK_R3)
    
    mask =  tl.arange(0, BLOCK_R3) < r3

    do = tl.load(
        tmp_merge_normalized + b_idx * stride_tmp_merge1 + tl.program_id(1) * stride_tmp_merge2 + tl.program_id(2) * stride_tmp_merge3 + tl.arange(0, BLOCK_R3), mask=mask, other=0
    )
    
    do *= tl.load(
        tmp_merge_grad + b_idx * stride_tmp_merge1 + tl.program_id(1) * stride_tmp_merge2 + tl.program_id(2) * stride_tmp_merge3 + tl.arange(0, BLOCK_R3),
        mask = mask, other=0
    )
    
    tl.atomic_add(l_bwd_ptr, do, mask=mask)    
    tl.atomic_add(r_bwd_ptr, do, mask=mask)
    





@triton.jit
def _kernel_inside_merge_discontinuous_v2(
            alpha_c, alpha_d,
            tmp_merge,
            tmp_merge_normalized,
            tmp_normalizer, 
            w, batch, L, 
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            stride_alpha_d1, stride_alpha_d2, stride_alpha_d3, stride_alpha_d4, stride_alpha_d5,
            stride_tmp_merge1, stride_tmp_merge2, stride_tmp_merge3,
            stride_tmp_normalizer1, stride_tmp_normalizer2,
            r1, r2, r3, r4,
            BLOCK_R3: tl.constexpr,
            BLOCK_R4: tl.constexpr,
            ):
    
    ## Find index. tl.program_id(1) is of size w-1, each indicates the length of the left continuous subspan given a discontinuous parent.
    # for each tl.program_id(1), the number of possible discontinuous spans is the same:  len(tl.program_id(2)) := (L-w) + (L-w-1) + (L-w-2) + ... + 1 = (L-w)*(L-w+1)/2.  
    ## (L-w-i) parts means that the start position is $i$, and each j \in [0, L-w-i] means the gap length (gap end - gap start)
    ## To avoid the waste of half amount of computation, I manually compute the index in the following way
    b_idx = tl.program_id(0)
    if b_idx >= batch:
        return 
    span_length_left = tl.program_id(1) + 1
    tid = tl.program_id(2)
    start = 0            
    # To find the group (L-w-start).  tid is the gap length then        

    while tid >= (L-w-start):
        tid -= (L-w-start)
        start += 1 
    
    gap_start = start + span_length_left
    gap_end = gap_start + (tid + 1)    
    end = gap_end + (w - span_length_left)

    acc2 = tl.zeros((BLOCK_R4,), dtype=tl.float32) - 1e9
        
    ## discontinuous parent nodes with one continuous child node and another discontinuous child node
    alpha_c_ptr = alpha_c + b_idx * stride_alpha_c1 + 2*r1 + r2 + 2 * r3 + tl.arange(0, BLOCK_R4)
    alpha_d_ptr = alpha_d + b_idx * stride_alpha_d1 +  r2 + tl.arange(0, BLOCK_R4)

    mask = tl.arange(0, BLOCK_R4) < r4

    for split in range(start+1, gap_start):
        #### continuous [i, j], discontinuous [j, k, m, n] -> discontinuous [i, k, m, n]
        c_ptr = alpha_c_ptr + start * stride_alpha_c2 + split * stride_alpha_c3 
        d_ptr = alpha_d_ptr + split * stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5
        
        child_c = tl.load(c_ptr, mask, other=-1e9)        
        child_d = tl.load(d_ptr, mask, other=-1e9)
        acc2 = logaddexp(acc2, child_c + child_d)

        #### continuous [j, k], discontinuous [i, j, m, n] -> discontinuous [i, k, m, n]
        c_ptr = alpha_c_ptr +  split * stride_alpha_c2 + gap_start * stride_alpha_c3 + r4
        d_ptr = alpha_d_ptr + start * stride_alpha_d2 + split * stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5 

        child_c = tl.load(c_ptr, mask, other=-1e9)
        child_d = tl.load(d_ptr, mask, other=-1e9)
        acc2 = logaddexp(acc2, child_c + child_d)


    for split in range(gap_end+1, end):
        #### continuous [m, j], discontinuous [i, k, j, n] -> discontinuous [i, k, m, n]. 
        c_ptr = alpha_c_ptr + gap_end * stride_alpha_c2 + split * stride_alpha_c3  + 2* r4
        d_ptr = alpha_d_ptr + start * stride_alpha_d2 + gap_start * stride_alpha_d3 + split * stride_alpha_d4 + end * stride_alpha_d5 
        
        child_c = tl.load(c_ptr, mask, other=-1e9)        
        child_d = tl.load(d_ptr, mask, other=-1e9)
        acc2 = logaddexp(acc2, child_c + child_d)

        #### continuous [j, k], discontinuous [i, j, m, n] -> discontinuous [i, k, m, n]
        c_ptr = alpha_c_ptr + split * stride_alpha_c2 + end * stride_alpha_c3 + 3 * r4 
        d_ptr = alpha_d_ptr + start * stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + split * stride_alpha_d5  
        
        child_c = tl.load(c_ptr, mask, other=-1e9)
        child_d = tl.load(d_ptr, mask, other=-1e9)
        acc2 = logaddexp(acc2, child_c + child_d)

    # acc = tl.cat(acc1, acc2, acc3, acc4, acc5)       
    # acc_max = tl.max(acc1, 0)
    acc_max =  tl.max(acc2, 0)

    # acc_max = tl.maximum(acc_max,  tl.max(acc3, 0))
    # acc_max = tl.maximum(acc_max,  tl.max(acc4, 0))
    # acc_max = tl.maximum(acc_max,  tl.max(acc5, 0))
    
    tl.store(tmp_normalizer + b_idx * stride_tmp_normalizer1 + tl.program_id(1) * stride_tmp_normalizer2 + tl.program_id(2), acc_max)

    ptr =  b_idx * stride_tmp_merge1 + tl.program_id(1) * stride_tmp_merge2 + tl.program_id(2) * stride_tmp_merge3 + tl.arange(0, BLOCK_R4)

    out = tl.exp(acc2 - acc_max)
    tl.store(tmp_merge + ptr , acc2, mask=mask)    
    tl.store(tmp_merge_normalized + ptr, out, mask=mask)
    
    


@triton.jit
def _kernel_bwd_merge_discontinuous_v2(
            alpha_c, alpha_d,
            tmp_merge,
            tmp_merge_normalized,
            tmp_merge_grad, 
            w, batch, L, 
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            stride_alpha_d1, stride_alpha_d2, stride_alpha_d3, stride_alpha_d4, stride_alpha_d5,
            stride_tmp_merge1, stride_tmp_merge2, stride_tmp_merge3,
            r1, r2, r3, r4,
            BLOCK_R3: tl.constexpr,
            BLOCK_R4: tl.constexpr,
            ):
    

    ## Find index. tl.program_id(1) is of size w-1, each indicates the length of the left continuous subspan given a discontinuous parent.
    # for each tl.program_id(1), the number of possible discontinuous spans is the same:  len(tl.program_id(2)) := (L-w) + (L-w-1) + (L-w-2) + ... + 1 = (L-w)*(L-w+1)/2.  
    ## (L-w-i) parts means that the start position is $i$, and each j \in [0, L-w-i] means the gap length (gap end - gap start)
    ## To avoid the waste of half amount of computation, I manually compute the index in the following way
    b_idx = tl.program_id(0)
    if b_idx >= batch:
        return 
    span_length_left = tl.program_id(1) + 1
    tid = tl.program_id(2)
    start = 0            
    # To find the group (L-w-start).  tid is the gap length then        

    while tid >= (L-w-start):
        tid -= (L-w-start)
        start += 1 
    
    gap_start = start + span_length_left
    gap_end = gap_start + (tid + 1)    
    end = gap_end + (w - span_length_left)

    ptr =  b_idx * stride_tmp_merge1 + tl.program_id(1) * stride_tmp_merge2 + tl.program_id(2) * stride_tmp_merge3 + tl.arange(0, BLOCK_R4)

    ## discontinuous parent nodes with one continuous child node and another discontinuous child node
    alpha_c_ptr = alpha_c + b_idx * stride_alpha_c1 + 2*r1 + r2 + 2 * r3 + tl.arange(0, BLOCK_R4)
    alpha_d_ptr = alpha_d + b_idx * stride_alpha_d1 +  r2 + tl.arange(0, BLOCK_R4)

    mask = tl.arange(0, BLOCK_R4) < r4

    parent_score = tl.load(tmp_merge + ptr, mask=mask, other=0)
    do = tl.load(tmp_merge_normalized + ptr, mask=mask, other=0) * tl.load(tmp_merge_grad + ptr, mask=mask, other=0)

    for split in range(start+1, gap_start):
        #### continuous [i, j], discontinuous [j, k, m, n] -> discontinuous [i, k, m, n]
        c_ptr = alpha_c_ptr + start * stride_alpha_c2 + split * stride_alpha_c3 
        d_ptr = alpha_d_ptr + split * stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5        

        child_c = tl.load(c_ptr,mask=mask, other=0)        
        child_d = tl.load(d_ptr,mask=mask,other=0)
        new_grad = tl.exp(child_c + child_d - parent_score) * do

        c_bwd_ptr = alpha_c_ptr + split * stride_alpha_c2 + start * stride_alpha_c3 
        d_bwd_ptr = alpha_d_ptr + gap_start * stride_alpha_d2 + split * stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5        
        tl.atomic_add(c_bwd_ptr,   new_grad)
        tl.atomic_add(d_bwd_ptr,   new_grad)

        

        #### continuous [j, k], discontinuous [i, j, m, n] -> discontinuous [i, k, m, n]
        c_ptr = alpha_c_ptr +  split * stride_alpha_c2 + gap_start * stride_alpha_c3 + r4
        d_ptr = alpha_d_ptr + start * stride_alpha_d2 + split * stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5 

        child_c = tl.load(c_ptr,mask=mask,other=0)
        child_d = tl.load(d_ptr,mask=mask,other=0)
        new_grad = tl.exp(child_c + child_d - parent_score) * do
    
        c_bwd_ptr = alpha_c_ptr +  gap_start * stride_alpha_c2 + split * stride_alpha_c3 + r4
        d_bwd_ptr = alpha_d_ptr + split * stride_alpha_d2 + start * stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5 
        tl.atomic_add(c_bwd_ptr,   new_grad)
        tl.atomic_add(d_bwd_ptr,   new_grad)
                

    for split in range(gap_end+1, end):
        #### continuous [m, j], discontinuous [i, k, j, n] -> discontinuous [i, k, m, n]. 
        c_ptr = alpha_c_ptr + gap_end * stride_alpha_c2 + split * stride_alpha_c3  + 2* r4
        d_ptr = alpha_d_ptr + start * stride_alpha_d2 + gap_start * stride_alpha_d3 + split * stride_alpha_d4 + end * stride_alpha_d5 
        
        child_c = tl.load(c_ptr,mask=mask,other=0)        
        child_d = tl.load(d_ptr,mask=mask,other=0)
        new_grad = tl.exp(child_c + child_d - parent_score) * do

        c_bwd_ptr = alpha_c_ptr + split * stride_alpha_c2 + gap_end * stride_alpha_c3  + 2* r4
        d_bwd_ptr = alpha_d_ptr + gap_start * stride_alpha_d2 + start * stride_alpha_d3 + split * stride_alpha_d4 + end * stride_alpha_d5 
        tl.atomic_add(c_bwd_ptr, new_grad)
        tl.atomic_add(d_bwd_ptr, new_grad)

        #### continuous [j, k], discontinuous [i, j, m, n] -> discontinuous [i, k, m, n]
        c_ptr = alpha_c_ptr + split * stride_alpha_c2 + end * stride_alpha_c3 + 3 * r4 
        d_ptr = alpha_d_ptr + start * stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + split * stride_alpha_d5  
        
        child_c = tl.load(c_ptr,mask=mask,other=0)
        child_d = tl.load(d_ptr,mask=mask,other=0)
        new_grad = tl.exp(child_c + child_d - parent_score) * do

        c_bwd_ptr = alpha_c_ptr + end * stride_alpha_c2 + split * stride_alpha_c3 + 3 * r4 
        d_bwd_ptr = alpha_d_ptr + gap_start * stride_alpha_d2 + start * stride_alpha_d3 + gap_end * stride_alpha_d4 + split * stride_alpha_d5  

        tl.atomic_add(c_bwd_ptr, new_grad)
        tl.atomic_add(d_bwd_ptr, new_grad)

    

    
### The reason why not save tmp_merge is that it could be recomputed very easily w/o overhead
### while saving ``tmp_merge'' wastes lots of memory
class MERGE_D1(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  alpha_c,  dimension_info):        
        B = alpha_c.shape[0]
        N = alpha_c.shape[1] - 1
        w = int(dimension_info[0])
        n = N - w 
        r1 = int(dimension_info[1])
        r2 = int(dimension_info[2])
        r3 = int(dimension_info[3])
        r4 = int(dimension_info[4])


        tmp_merge_normalized = alpha_c.new_zeros(B, w-1, int((N-w)*(N-w+1)/2), r3).fill_(0)
        tmp_normalizer =  alpha_c.new_zeros(B, w-1, int((N-w)*(N-w+1)/2)).fill_(-1e9)
        grid = (triton.next_power_of_2(B), (w-1), int((N-w)*(N-w+1)/2))            

        
        _kernel_inside_merge_discontinuous_v1[grid](alpha_c, 
                                                tmp_merge_normalized,   tmp_normalizer,
                                                w, B, N, 
                                                alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
                                                tmp_merge_normalized.stride(0), tmp_merge_normalized.stride(1), tmp_merge_normalized.stride(2),
                                                tmp_normalizer.stride(0), tmp_normalizer.stride(1),
                                                r1, r2, r3, r4,
                                                BLOCK_R3=triton.next_power_of_2(r3)
                                                )

        ctx.save_for_backward(tmp_merge_normalized, alpha_c, dimension_info)                
        return tmp_merge_normalized, tmp_normalizer
            
    @staticmethod
    def backward(ctx, do, do2):

        tmp_merge_normalized, alpha_c, dimension_info = ctx.saved_tensors
        B = alpha_c.shape[0]
        N = alpha_c.shape[1] - 1
        w = int(dimension_info[0])
        n = N - w 
        r1 = int(dimension_info[1])
        r2 = int(dimension_info[2])
        r3 = int(dimension_info[3])
        r4 = int(dimension_info[4])
        grid = (triton.next_power_of_2(B), (w-1), int((N-w)*(N-w+1)/2))            

        _kernel_bwd_merge_discontinuous_v1[grid](
            alpha_c,                    
            tmp_merge_normalized, do, 
            w, B, N, 
            alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
            tmp_merge_normalized.stride(0), tmp_merge_normalized.stride(1), tmp_merge_normalized.stride(2),
            r1, r2, r3, r4,
            BLOCK_R3= triton.next_power_of_2(r3),
            BLOCK_R4= triton.next_power_of_2(r4)
        )
    
        return alpha_c, None

class MERGE_D2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_c, alpha_d, dimension_info):                
        B = alpha_c.shape[0]
        N = alpha_c.shape[1] - 1
        w = int(dimension_info[0])
        n = N - w 
        r1 = int(dimension_info[1])
        r2 = int(dimension_info[2])
        r3 = int(dimension_info[3])
        r4 = int(dimension_info[4])
        tmp_merge = alpha_c.new_zeros(B, w-1, int((N-w)*(N-w+1)/2), r4).fill_(-1e9)
        tmp_merge_normalized = alpha_c.new_zeros(B, w-1, int((N-w)*(N-w+1)/2), r4)
        tmp_normalizer =  alpha_c.new_zeros(B, w-1, int((N-w)*(N-w+1)/2)).fill_(-1e9)
        grid =  ( triton.next_power_of_2(B), (w-1), int((N-w)*(N-w+1)/2))            

        _kernel_inside_merge_discontinuous_v2[grid](alpha_c, 
                                                alpha_d,        
                                                tmp_merge, tmp_merge_normalized,  tmp_normalizer,
                                                w, B, N,
                                                alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
                                                alpha_d.stride(0), alpha_d.stride(1), alpha_d.stride(2), alpha_d.stride(3), alpha_d.stride(4),
                                                tmp_merge.stride(0), tmp_merge.stride(1), tmp_merge.stride(2),
                                                tmp_normalizer.stride(0),  tmp_normalizer.stride(1),                 
                                                r1, r2, r3, r4,
                                                BLOCK_R3 =triton.next_power_of_2(r3),  BLOCK_R4 = triton.next_power_of_2(r4)
                                            )        
        
        ctx.save_for_backward(tmp_merge, tmp_merge_normalized, alpha_c, alpha_d, dimension_info)                
        return tmp_merge_normalized, tmp_normalizer
    


    @staticmethod
    def backward(ctx, do, do2):
        tmp_merge, tmp_merge_normalized, alpha_c, alpha_d, dimension_info =  ctx.saved_tensors

        B = alpha_c.shape[0]
        N = alpha_c.shape[1] - 1
        w = int(dimension_info[0])
        n = N - w 
        r1 = int(dimension_info[1])
        r2 = int(dimension_info[2])
        r3 = int(dimension_info[3])
        r4 = int(dimension_info[4])
        # tmp_normalizer =  alpha_c.new_zeros(B, w-1, int((N-w)*(N-w+1)/2)).fill_(-1e9)
        grid = (triton.next_power_of_2(B), (w-1), int((N-w)*(N-w+1)/2))            
        
        _kernel_bwd_merge_discontinuous_v2[grid](alpha_c, 
                                                alpha_d,        
                                                tmp_merge, tmp_merge_normalized,  do,
                                                w, B, N,
                                                alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
                                                alpha_d.stride(0), alpha_d.stride(1), alpha_d.stride(2), alpha_d.stride(3), alpha_d.stride(4),
                                                tmp_merge.stride(0), tmp_merge.stride(1), tmp_merge.stride(2),
                                                r1, r2, r3, r4,
                                                BLOCK_R3 = triton.next_power_of_2(r3),  BLOCK_R4 = triton.next_power_of_2(r4)
                                            )        
        
        return alpha_c, alpha_d, None



_merge_discontinuous_v1 = MERGE_D1.apply
_merge_discontinuous_v2 = MERGE_D2.apply

def merge_discontinuous_v1(
        alpha_c, 
        f_d1,
        dimension_info,
):
    out, normalizer = _merge_discontinuous_v1(alpha_c, dimension_info)

    return ((out @ f_d1) + 1e-9).log() + normalizer[..., None]



def merge_discontinuous_v2(
        alpha_c,
        alpha_d,
        f_d2,
        dimension_info,
):
    out, normalizer = _merge_discontinuous_v2(alpha_c, alpha_d, dimension_info)
    return ((out @ f_d2) + 1e-9).log() + normalizer[..., None]
    # else:
    # return 



# @checkpoint

def _merge_discontinuous(
        alpha_c, 
        alpha_d,
        f_d1,
        f_d2,
        dimension_info
):
    out1 = merge_discontinuous_v1(alpha_c, f_d1, dimension_info)
    out2 = merge_discontinuous_v2(alpha_c, alpha_d, f_d2, dimension_info)

    return torch.logaddexp(out1, out2)






    
    
    

    
