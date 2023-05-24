import pdb
from numpy import dtype
# import statistics
import torch
import triton
import triton.language as tl

@triton.jit
def update_position(a, b, position_a, position_b):
    tmp = a - b
    return tl.where(tmp > 0, position_a , position_b)




@triton.jit
def _kernel_argmax_merge_continuous(
            alpha_c, alpha_d, 
            marginal_c, 
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            stride_alpha_d1, stride_alpha_d2, stride_alpha_d3, stride_alpha_d4, stride_alpha_d5,
            B, w, L,
            ):    
    
    b_idx = tl.program_id(0)
    if b_idx >= B:
        return
    start = tl.program_id(1)
    end = start + w

    # if b_idx >= b:
        # return        
    # [i, k], [k, j] -> [i, j]



    l_ptr = alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (start+1) * stride_alpha_c3 
    r_ptr = alpha_c + b_idx * stride_alpha_c1 + (start +1) * stride_alpha_c2 +  (end) * stride_alpha_c3 
    
    acc1 =  tl.zeros((1,),  dtype=tl.float32)  -1e9
    max_idx =  tl.zeros((1,),dtype=tl.float32) -1

    for split in range(start+1, start+w):
        left = tl.load(l_ptr)
        right = tl.load(r_ptr)
        merge = left + right
        
        max_idx = update_position(acc1, merge, max_idx,   tl.zeros((1,),dtype=tl.float32) +split)
        acc1 = tl.maximum(merge,acc1)

        l_ptr += stride_alpha_c3
        r_ptr += stride_alpha_c2


    # [m, n], [i, m, n, j] -> [i, j]   i:=start, j:=end, m:=gap start, n:= gapend. 
    #  corresponding rank within the last dim of alpha_c. [2r1, 2r1+r2]     
    for gap_start in range(start+1, end-1):
        for gap_end in range(gap_start+1, end):
            ptr_c = alpha_c + b_idx * stride_alpha_c1 + gap_start * stride_alpha_c2 + gap_end * stride_alpha_c3        
            ptr_d = alpha_d + b_idx * stride_alpha_d1 + start * stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + (end) * stride_alpha_d5 
            cont = tl.load(ptr_c)
            disco = tl.load(ptr_d)            
            merge = cont + disco
            max_idx = update_position(acc1, merge, max_idx,   tl.zeros((1,),dtype=tl.float32) -(gap_start * L + gap_end))
            acc1 = tl.maximum(merge,acc1)

                
    tl.store( 
        alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (start+w) * stride_alpha_c3 + tl.arange(0, 1), 
        acc1 + tl.load(marginal_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (start+w) * stride_alpha_c3)
    )

    tl.store(
        alpha_c + b_idx * stride_alpha_c1 + (start+w) * stride_alpha_c2 + (start) * stride_alpha_c3 + tl.arange(0, 1), 
        max_idx
        # tl.full((1,), max_idx, tl.float32)
    )













# 
# For convenience, 
@triton.jit
def _kernel_argmax_merge_discontinuous(
            alpha_c, alpha_d,
            marginal_d,
            w, batch, L, 
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            stride_alpha_d1, stride_alpha_d2, stride_alpha_d3, stride_alpha_d4, stride_alpha_d5,
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

    alpha_c_ptr = alpha_c + b_idx * stride_alpha_c1
    alpha_d_ptr = alpha_d + b_idx * stride_alpha_d1

    max_score = tl.load(alpha_c_ptr + start * stride_alpha_c2 + gap_start) + tl.load(alpha_c_ptr + gap_end * stride_alpha_c2 + end)
    max_idx =  tl.zeros((1,), dtype=tl.float32) -1 

    for split in range(start+1, gap_start):
        #### continuous [i, j], discontinuous [j, k, m, n] -> discontinuous [i, k, m, n]
        c_ptr = alpha_c_ptr + start * stride_alpha_c2 + split * stride_alpha_c3 
        d_ptr = alpha_d_ptr + split * stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5
        
        score = tl.load(c_ptr) +  tl.load(d_ptr)

        max_idx = update_position(max_score, score, max_idx,   tl.zeros((1,),dtype=tl.float32) + split)
        max_score = tl.maximum(score,max_score)

        
        #### continuous [j, k], discontinuous [i, j, m, n] -> discontinuous [i, k, m, n]
        c_ptr = alpha_c_ptr +  split * stride_alpha_c2 + gap_start * stride_alpha_c3 
        d_ptr = alpha_d_ptr + start * stride_alpha_d2 + split * stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5 
        score = tl.load(c_ptr) +  tl.load(d_ptr)

        max_idx = update_position(max_score, score, max_idx,   tl.zeros((1,),dtype=tl.float32) + split + L + 1)
        max_score = tl.maximum(score,max_score)


    for split in range(gap_end+1, end):
        #### continuous [m, j], discontinuous [i, k, j, n] -> discontinuous [i, k, m, n]. 
        c_ptr = alpha_c_ptr + gap_end * stride_alpha_c2 + split * stride_alpha_c3  
        d_ptr = alpha_d_ptr + start * stride_alpha_d2 + gap_start * stride_alpha_d3 + split * stride_alpha_d4 + end * stride_alpha_d5 
        score = tl.load(c_ptr) +  tl.load(d_ptr)
       
        max_idx = update_position(max_score, score, max_idx,   tl.zeros((1,),dtype=tl.float32) + split + 2* (L+1))
        max_score = tl.maximum(score,max_score)

        #### continuous [j, k], discontinuous [i, j, m, n] -> discontinuous [i, k, m, n]
        c_ptr = alpha_c_ptr + split * stride_alpha_c2 + end * stride_alpha_c3 
        d_ptr = alpha_d_ptr + start * stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + split * stride_alpha_d5  
        score = tl.load(c_ptr) +  tl.load(d_ptr)

        max_idx = update_position(max_score, score, max_idx,   tl.zeros((1,),dtype=tl.float32) + split + 3*(L+1))
        max_score = tl.maximum(score,max_score)


    span_score = tl.load(
        marginal_d + b_idx * stride_alpha_d1 + start * stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5          
    )

    tl.store(
        alpha_d + b_idx * stride_alpha_d1 + start * stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5,
        max_score + span_score
    )   

    tl.store(
        alpha_d + b_idx * stride_alpha_d1 + gap_start * stride_alpha_d2 + start * stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5 + tl.arange(0, 1),
        max_idx 
    )   



def argmax_on5_wn(alpha_c_mbr, alpha_d_mbr, marginal_c, marginal_d, 
                  ):
    B, L = alpha_c_mbr.shape[0], alpha_c_mbr.shape[1]    
    for w in range(2, L):
        n = L - w      
        grid = (B, n)        
        _kernel_argmax_merge_continuous[grid](
            alpha_c_mbr, alpha_d_mbr,            
            marginal_c,
            alpha_c_mbr.stride(0), alpha_c_mbr.stride(1), alpha_c_mbr.stride(2),
            alpha_d_mbr.stride(0),alpha_d_mbr.stride(1), alpha_d_mbr.stride(2), alpha_d_mbr.stride(3), alpha_d_mbr.stride(4),
            B, w, L
        )

        if w < L-1:
            grid = (B, (w-1), int((L-w-1)*(L-w)/2))            
            _kernel_argmax_merge_discontinuous[grid](
                alpha_c_mbr, alpha_d_mbr, 
                marginal_d, 
                w, B, L-1, 
                alpha_c_mbr.stride(0), alpha_c_mbr.stride(1), alpha_c_mbr.stride(2),
                alpha_d_mbr.stride(0),alpha_d_mbr.stride(1), alpha_d_mbr.stride(2), alpha_d_mbr.stride(3), alpha_d_mbr.stride(4)
            )


            



