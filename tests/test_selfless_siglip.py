import copy
from types import SimpleNamespace

import pytest
import torch

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from models.modeling_model.modeling_selfless_siglip import SelflessSiglipConfig, SelflessSiglipForCausalLM
from utils.utils import get_selfless_mask


def tiny_config():
    return SelflessSiglipConfig(vocab_size=40, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=9,
        mask_token_id=7, image_mask_token_id=8, boi_token_id=11, eoi_token_id=12,
        image_latent_dim=4, image_tokens_per_img=4, image_flow_width=32, image_flow_depth=2,
        image_flow_num_sampling_steps='10', image_flow_batch_mul=4, image_flow_time_scale=1000.,
        image_flow_time_sampling='uniform', image_flow_time_eps=1e-5, image_flow_time_uniform_mix=0.,
        image_flow_solver='heun', image_uncond_prob=.1, image_input_noise_strength=0.,
        training_objective='selfless_dual_stream', architecture_variant='selfless_siglip',
        dual_stream_attention_contract='xlnet_content_diagonal', flow_head_attention_contract='xlnet_content_diagonal',
        flow_condition_contract='backbone_xt_query_backbone_x0_content', training_image_sigma_order='random',
        lambda_text=.05, lambda_image=1., use_flex_attention=False, tie_word_embeddings=True,
        b_siglip_width=16, b_siglip_intermediate=32, b_siglip_depth=2, b_siglip_heads=2,
        b_siglip_initialization_seed=42, b_siglip_gradient_checkpointing=True)


def model():
    torch.manual_seed(42)
    m=SelflessSiglipForCausalLM(tiny_config())
    with torch.no_grad():
        for block in m.image_flow_head.net.blocks:
            block.adaLN_modulation[-1].weight.normal_(0.,.1)
            block.adaLN_modulation[-1].bias.normal_(0.,.1)
        m.image_flow_head.net.final_layer.linear.weight.normal_(0.,.1)
        m.image_flow_head.net.final_layer.linear.bias.normal_(0.,.1)
    return m


def batch():
    ids=torch.tensor([[3,11,8,8,8,8,12,4,5],[4,5,11,8,8,8,8,12,6]])
    types=torch.tensor([[0,2,1,1,1,1,2,0,0],[0,0,2,1,1,1,1,2,0]])
    sigma=torch.tensor([[0,1,5,3,6,4,2,7,8],[0,1,2,6,4,7,5,3,8]])
    latents=torch.randn(2,9,4)
    return ids,types,sigma,latents


def backbone(m,ids,types,sigma,latents,drop=False):
    masks=dict(sigma=sigma,seq_len=ids.shape[1],device=ids.device,input_ids=ids,token_types=types,
        boi_token_id=11,image_uncond_rows=torch.full((ids.shape[0],),drop,dtype=torch.bool))
    return m.model(X0_input_ids=ids, token_types=types,image_latents=latents,
        attention_mask=get_selfless_mask(**masks),content_attention_mask=get_selfless_mask(**masks,include_diagonal=True),
        calculate_likelihood=True,return_x0_hidden_state=True)


def test_all_common_b_parameters_and_rng_are_identical():
    config=tiny_config();base_config=copy.deepcopy(config);base_config.architecture_variant='selfless_contextual'
    torch.manual_seed(64);baseline=Qwen3ForCausalLM(base_config);after_baseline=torch.get_rng_state()
    torch.manual_seed(64);extra=SelflessSiglipForCausalLM(config);after_extra=torch.get_rng_state()
    for name, p in baseline.state_dict().items():
        torch.testing.assert_close(p,extra.state_dict()[name],rtol=0,atol=0,msg=name)
    assert torch.equal(after_baseline,after_extra)


def test_semantic_initialization_does_not_reseed_accelerators(monkeypatch):
    calls=[]
    monkeypatch.setattr(torch.random,'_seed_custom_device',lambda seed:calls.append(seed))
    SelflessSiglipForCausalLM(tiny_config())
    assert calls==[]


def test_future_clean_latents_cannot_affect_earlier_queries_or_content():
    m=model().eval();ids,types,sigma,latents=batch()
    original=backbone(m,ids,types,sigma,latents)
    changed=latents.clone();changed[(types==1)&(sigma>=5)]+=50*torch.randn_like(changed[(types==1)&(sigma>=5)])
    other=backbone(m,ids,types,sigma,changed)
    strict=sigma<=5;content=sigma<5
    torch.testing.assert_close(original.last_hidden_state[strict],other.last_hidden_state[strict],rtol=0,atol=0)
    torch.testing.assert_close(original['x0_last_hidden_state'][content],other['x0_last_hidden_state'][content],rtol=0,atol=0)
    assert (original.last_hidden_state[sigma>5]-other.last_hidden_state[sigma>5]).abs().max()>1e-5


def test_cfg_semantic_front_does_not_reintroduce_text_condition():
    m=model().eval();ids,types,sigma,latents=batch();other_ids=ids.clone();other_ids[types==0]+=15
    a=backbone(m,ids,types,sigma,latents,True);b=backbone(m,other_ids,types,sigma,latents,True)
    torch.testing.assert_close(a.last_hidden_state[types==1],b.last_hidden_state[types==1],rtol=0,atol=0)
    torch.testing.assert_close(a['x0_last_hidden_state'][types==1],b['x0_last_hidden_state'][types==1],rtol=0,atol=0)


def test_real_joint_losses_and_text_step_connect_every_parameter():
    m=model().train();ids,types,sigma,latents=batch();masks=dict(sigma=sigma,seq_len=9,device=ids.device)
    labels=ids.masked_fill(types!=0,-100)
    result=m(X0_input_ids=ids,labels=labels,token_types=types,image_latents=latents,
        image_loss_mask=types==1,image_span_table=torch.tensor([[0,0,2,6],[1,0,3,7]]),flow_sigma=sigma,
        attention_mask=get_selfless_mask(**masks),content_attention_mask=get_selfless_mask(**masks,include_diagonal=True))
    assert torch.isfinite(result.loss);result.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert sum(p.grad.float().square().sum() for p in m.model.semantic_encoder.parameters())>0
    m.zero_grad();text=ids[:,:2].contiguous();sig=torch.arange(2)[None].expand(2,-1)
    m(X0_input_ids=text,labels=text,token_types=torch.zeros_like(text),attention_mask=get_selfless_mask(sig,2,ids.device),
      content_attention_mask=get_selfless_mask(sig,2,ids.device,include_diagonal=True),
      compute_image_loss=False).loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


@torch.no_grad()
def test_cached_generation_matches_full_with_cfg_and_random_spatial_order(tmp_path):
    m=model().eval();ids,types,sigma,_=batch()
    kwargs=dict(input_ids=ids,token_types=types,sigma=sigma,spans=[(0,2,6),(1,3,7)],
        initial_noise_bank=torch.arange(32).float().reshape(2,4,4)/17,
        flow_temperature=.7,flow_cfg=2.5,flow_solver='heun',flow_num_steps=2,
        parallel_rate=1,order_strategy='spatial_halton',return_trace=True)
    cached,trace=m.generate_image(**kwargs,use_cache=True)
    full,full_trace=m.generate_image(**kwargs,use_cache=False)
    torch.testing.assert_close(cached,full,rtol=2e-5,atol=2e-6)
    assert trace['semantic_cache_peak_bytes']>0 and trace['semantic_forward_calls']>0
    m.save_pretrained(tmp_path,safe_serialization=True)
    loaded=SelflessSiglipForCausalLM.from_pretrained(tmp_path).eval()
    actual,_=loaded.generate_image(**kwargs,use_cache=True)
    torch.testing.assert_close(actual,cached,rtol=0,atol=0)


@torch.no_grad()
def test_i2t_and_text_cached_generation_match_full():
    m=model().eval();ids,types,sigma,latents=batch()
    for image in [True,False]:
        kwargs=dict(input_ids=ids,token_types=types if image else torch.zeros_like(types),
            max_new_tokens=4,temperature=0.,eos_token_id=-1,return_trace=True)
        if image:kwargs.update(image_latents=latents,image_latent_mask=types==1,sigma=sigma)
        a,trace=m.generate_text(**kwargs,use_cache=True);b,_=m.generate_text(**kwargs,use_cache=False)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert bool(trace['semantic_forward_calls'])==image


def test_fresh_qwen_loading_keeps_b_initial_modules_and_rng(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM as HFQwen
    base_config=Qwen3Config(vocab_size=40,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=True)
    torch.manual_seed(123);HFQwen(base_config).save_pretrained(tmp_path)
    config=tiny_config();b_config=copy.deepcopy(config);b_config.architecture_variant='selfless_contextual'
    torch.manual_seed(42);b=Qwen3ForCausalLM.from_pretrained(tmp_path,config=b_config);b.reset_image_modules();b_rng=torch.get_rng_state()
    torch.manual_seed(42);extra=SelflessSiglipForCausalLM.from_pretrained(tmp_path,config=config);extra.reset_image_modules();extra_rng=torch.get_rng_state()
    for name,value in b.state_dict().items():
        torch.testing.assert_close(value,extra.state_dict()[name],rtol=0,atol=0,msg=name)
    assert torch.equal(b_rng,extra_rng)


def test_text_only_microbatch_does_not_add_random_draws():
    config=tiny_config();config.image_input_noise_strength=.01
    bconfig=copy.deepcopy(config);bconfig.architecture_variant='selfless_contextual'
    torch.manual_seed(42);b=Qwen3ForCausalLM(bconfig).train()
    torch.manual_seed(42);extra=SelflessSiglipForCausalLM(config).train()
    ids=torch.tensor([[3,4,5,6]]);sigma=torch.arange(4)[None]
    kwargs=dict(X0_input_ids=ids,labels=ids,token_types=torch.zeros_like(ids),image_latents=torch.zeros(1,4,4),
        image_span_table=torch.empty(0,4,dtype=torch.long),compute_image_loss=False,
        attention_mask=get_selfless_mask(sigma,4,ids.device),content_attention_mask=get_selfless_mask(sigma,4,ids.device,include_diagonal=True))
    torch.manual_seed(52);first=b(**kwargs).loss;first_rng=torch.get_rng_state()
    torch.manual_seed(52);second=extra(**kwargs).loss;second_rng=torch.get_rng_state()
    torch.testing.assert_close(first,second,rtol=0,atol=0);assert torch.equal(first_rng,second_rng)


@torch.no_grad()
def test_image_likelihood_prefix_cache_matches_full_scoring():
    from utils.evaluation.native_understanding import score_text_candidates, score_text_candidates_cached_prefix
    m=model().eval();latent=torch.randn(4,4)
    class Tokenizer:
        eos_token_id=9
        def encode(self,text,add_special_tokens=False):return [3+ord(c)%4 for c in text]
    kwargs=dict(model=m,tokenizer=Tokenizer(),cache=SimpleNamespace(sample=lambda _:latent),image_id=1,
        item_id='test',prompt='Describe',candidates=['red','blue'],device=torch.device('cpu'),
        image_sigma_order='random',attention_contract='xlnet_content_diagonal',evaluation_task='toy',
        args=SimpleNamespace(batch_size_per_rank=2,request_chunk_size=4,max_length=64,lm_head_chunk_tokens=2,seed=42))
    a=score_text_candidates(**kwargs);b=score_text_candidates_cached_prefix(**kwargs)
    for first,second in zip(a,b):torch.testing.assert_close(first,second,rtol=1e-6,atol=2e-6)


def test_extra_frontend_checkpointing_preserves_gradients():
    first=model().train();second=copy.deepcopy(first);first.model.semantic_encoder.gradient_checkpointing=False
    ids,types,sigma,latents=batch()
    for m in [first,second]:backbone(m,ids,types,sigma,latents).last_hidden_state.square().sum().backward()
    for (name,a),(_,b) in zip(first.named_parameters(),second.named_parameters()):
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0,msg=name)


def test_ema_loader_preserves_semantic_contract_and_rejects_missing_branch(tmp_path):
    import json
    from omegaconf import OmegaConf
    from safetensors.torch import load_file, save_file
    from utils.evaluation_model_source import configure_model_source, resolve_evaluation_model_source
    m=model();m.save_pretrained(tmp_path)
    (tmp_path/'tokenizer.json').write_text('{}')
    weights=load_file(tmp_path/'model.safetensors')
    metadata=dict(schema='selfless_ema_hf_export_v1',floating_dtype='float32',source_global_step=6,
        source_world_size=16,state_key_count=len(m.state_dict()),stored_weight_key_count=len(weights),export_kind='training')
    (tmp_path/'ema_export_metadata.json').write_text(json.dumps(metadata))
    config=OmegaConf.create(dict(model=dict(architecture_variant='selfless_contextual'),training=dict(from_scratch=False)))
    configure_model_source(config,resolve_evaluation_model_source(tmp_path))
    assert config.model.architecture_variant=='selfless_siglip'
    assert config.model.b_siglip_width==16
    assert config.model.b_siglip_visibility_contract=='same_image_native_x0_sigma_causal'
    assert config.model.flow_condition_contract=='backbone_xt_query_backbone_x0_content'
    weights={k:v for k,v in weights.items() if not k.startswith('model.semantic_encoder.')}
    save_file(weights,tmp_path/'model.safetensors')
    metadata['stored_weight_key_count']=len(weights)
    (tmp_path/'ema_export_metadata.json').write_text(json.dumps(metadata))
    with pytest.raises((RuntimeError,ValueError),match='semantic_encoder'):
        resolve_evaluation_model_source(tmp_path)


def test_resume_contract_rejects_semantic_or_optimizer_changes():
    from omegaconf import OmegaConf
    from utils.b_siglip_protocol import config_path
    from utils.selfless_training_runtime import build_resume_contract,validate_resume_contract,RESUME_SCHEMA,RESUME_CONTRACT_VERSION
    config=OmegaConf.load(config_path())
    build=lambda c:build_resume_contract(c,world_size=16,gradient_accumulation_steps=4)
    metadata=dict(schema=RESUME_SCHEMA,config_contract_version=RESUME_CONTRACT_VERSION,config_contract=build(config))
    config.training.stop_after_steps=8
    validate_resume_contract(metadata,current_contract=build(config))
    for field,value in [('model.b_siglip_visibility_contract','full_image'),('optimizer.params.semantic_learning_rate',5e-5)]:
        changed=copy.deepcopy(config);OmegaConf.update(changed,field,value)
        with pytest.raises(RuntimeError,match='differs'):validate_resume_contract(metadata,current_contract=build(changed))


def test_shared_mc_content_matches_full_model_loss_and_gradients():
    first=model().train();second=copy.deepcopy(first)
    second.config.image_flow_share_content=True
    ids,types,sigma,latents=batch();masks=dict(sigma=sigma,seq_len=9,device=ids.device)
    kwargs=dict(X0_input_ids=ids,labels=ids.masked_fill(types!=0,-100),token_types=types,image_latents=latents,
        image_loss_mask=types==1,image_span_table=torch.tensor([[0,0,2,6],[1,0,3,7]]),flow_sigma=sigma,
        attention_mask=get_selfless_mask(**masks),content_attention_mask=get_selfless_mask(**masks,include_diagonal=True))
    torch.manual_seed(52);a=first(**kwargs).loss;a.backward()
    torch.manual_seed(52);b=second(**kwargs).loss;b.backward()
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    for (name,p),(_,q) in zip(first.named_parameters(),second.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6,msg=name)


def test_launch_contract_supports_formal_and_same_run_smoke_resume(tmp_path,monkeypatch):
    import json
    from omegaconf import OmegaConf
    from scripts.launch_unified_b_siglip import launch_plan
    from utils.b_siglip_protocol import config_path
    import utils.training_checkpoint
    environment=dict(PET_NNODES='4',PET_NODE_RANK='3',PET_NPROC_PER_NODE='16',PET_MASTER_ADDR='127.0.0.1',PET_MASTER_PORT='12345')
    formal=launch_plan('b-siglip',smoke=False,label='r1',steps=6,environment=environment)
    assert formal['world_size']==64 and formal['rank']==3
    assert formal['experiment_identity']['id']=='b_siglip'
    fresh=launch_plan('b-siglip',smoke=True,label='test',steps=6,environment={})
    assert fresh['experiment_identity']['purpose']=='temporary'
    config=OmegaConf.load(config_path());config.training.stop_after_steps=6
    monkeypatch.chdir(tmp_path)
    # Keep the frozen config location explicit while exercising a separate run root.
    import scripts.launch_unified_b_siglip as launcher
    monkeypatch.setattr(launcher,'config_path',lambda _:str(__import__('utils.b_siglip_protocol',fromlist=['ROOT']).ROOT/config_path()))
    root=tmp_path/fresh['output_root'];root.mkdir(parents=True);OmegaConf.save(config,root/'config.yaml')
    ckpt=root/'checkpoint-6';ckpt.mkdir();(ckpt/'metadata.json').write_text(json.dumps(dict(global_step=6,world_size=16)))
    monkeypatch.setattr(utils.training_checkpoint,'_validate_checkpoint_complete',lambda *a,**k:None)
    resumed=launch_plan('b-siglip',smoke=True,label='test',steps=8,environment={},resume_from_checkpoint=str(ckpt))
    assert resumed['resume_step']==6 and 'resume-6-to-8' in resumed['audit_root']
    with pytest.raises(ValueError,match='later stop'):
        launch_plan('b-siglip',smoke=True,label='test',steps=6,environment={},resume_from_checkpoint=str(ckpt))
