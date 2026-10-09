# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Transformers 4.57.6 native Qwen2 RoPE-buffer exception; checks only."""
def validate_model_precision(model,dtype,torch):
    target=torch.device('cuda:0');name_exact='model.language_model.rotary_emb.inv_freq'
    rotary=model.model.language_model.rotary_emb
    assert type(rotary).__name__=='Qwen2RotaryEmbedding'
    assert type(rotary).__module__=='transformers.models.qwen2.modeling_qwen2'
    assert rotary.rope_type=='default','fixed checkpoint uses default RoPE'
    assert 'inv_freq' in rotary._non_persistent_buffers_set,'native inv_freq must be nonpersistent'
    config=model.config.text_config
    head_dim=getattr(config,'head_dim',None) or config.hidden_size//config.num_attention_heads
    expected_shape=(int(head_dim*getattr(config,'partial_rotary_factor',1.0))//2,)
    parameters=0;buffers=[];exceptions=[];seen=False
    for name,value in model.named_parameters():
        assert value.device==target,('parameter device',name,str(value.device))
        if value.is_floating_point():
            assert value.dtype==dtype,('parameter dtype',name,str(value.dtype),str(dtype))
            parameters+=1
    for name,value in model.named_buffers():
        assert value.device==target,('buffer device',name,str(value.device))
        row={'name':name,'dtype':str(value.dtype),'device':str(value.device),'shape':list(value.shape)}
        if name==name_exact:
            assert value is rotary.inv_freq,'unexpected inv_freq owner'
            assert value.dtype==torch.float32 and tuple(value.shape)==expected_shape,('native RoPE dtype/shape',row)
            row['native_fp32_reason']='HF 4.57.6 default RoPE initializer uses float32; rotary forward computes float32 then casts cos/sin to input dtype'
            seen=True
            if value.dtype!=dtype:exceptions.append(dict(row))
        elif value.is_floating_point():
            assert value.dtype==dtype,('unlisted buffer dtype',name,str(value.dtype),str(dtype))
        buffers.append(row)
    assert seen,'expected native Qwen2 inv_freq buffer missing'
    return {'floating_parameter_tensors_checked':parameters,'runtime_parameter_dtype':str(dtype),
        'all_parameter_and_buffer_devices':'cuda:0','buffers':buffers,
        'native_dtype_exceptions':exceptions,'mutation_performed':False}
