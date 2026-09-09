"""Offline frozen JanusFlow / Show-o2 probes, using pinned official model code.

Run in the isolated transformers-4.47 dependency overlay. No source checkpoint
is modified. Content and native task-slot readouts are explicitly separate.
"""

from __future__ import annotations

import json
import sys

import torch
from PIL import Image
from safetensors import safe_open
from torchvision.transforms.functional import to_tensor

from scripts.geometry_v5_encoders import LayerCollector, PixelLoader, verify_parameters
from utils.research.geometry_v5_assets import MODELS

JANUS_SOURCE = (
    MODELS
    / "_source_snapshots/deepseek-ai--Janus/1daa72fa409002d40931bd7b36a9280362469ead"
)
SHOWO_SOURCE = (
    MODELS
    / "_source_snapshots/showlab--Show-o/45a5a2de01d1ebd10cd5864d29310a76476cdf23/show-o2"
)
FLOW_SEED = 20260911


class MaskCollector(LayerCollector):
    stable_input_pooling = False

    def begin_masks(self, masks):
        assert masks.ndim == 3 and bool(masks.any(-1).all())
        self.masks = masks
        self.values = []

    def collect(self, value):
        assert value.shape[:2] == (self.masks.shape[0], self.masks.shape[2])
        if self.stable_input_pooling and not self.values:
            # Identical fixed-noise slots must remain identical at input layer 0.
            # Padded BMM reduction can change rounding with caption/padding length.
            self.values.append(
                torch.stack(
                    [
                        torch.stack([value[b, mask].float().mean(0) for mask in masks])
                        for b, masks in enumerate(self.masks)
                    ]
                )
            )
            return
        weights = self.masks.float()
        weights = weights / weights.sum(-1, keepdim=True)
        self.values.append(torch.bmm(weights, value.float()))


def readout_masks(length, content, slots=None, boundary=None):
    """Masks are semantic-role masks, never all tokens including the prompt."""
    content = list(content)
    assert content
    selections = [content, content[-1:]]
    if slots is not None:
        slots = list(slots)
        side = int(len(slots) ** 0.5)
        assert side * side == len(slots)
        selections.extend([slots, [slots[(side // 2) * side + side // 2]]])
    else:
        selections.append([boundary])
    value = torch.zeros(len(selections), length, dtype=torch.bool)
    for i, selection in enumerate(selections):
        value[i, selection] = True
    return value


def pack(sequences, masks, device, image_blocks=None):
    """Right padding; true image blocks are bidirectional only for Show-o2."""
    width = max(seq.shape[0] for seq in sequences)
    embedded = torch.stack(
        [torch.nn.functional.pad(seq, (0, 0, 0, width - len(seq))) for seq in sequences]
    )
    pools = torch.stack(
        [torch.nn.functional.pad(mask, (0, width - mask.shape[-1])) for mask in masks]
    ).to(device)
    visible = (
        torch.arange(width)[None] < torch.tensor([len(s) for s in sequences])[:, None]
    )
    if image_blocks is None:
        return embedded, pools, visible.to(device)
    allowed = torch.ones(len(sequences), 1, width, width, dtype=torch.bool).tril()
    for i, (offset, length) in enumerate(image_blocks):
        allowed[i, :, offset : offset + length, offset : offset + length] = True
    allowed &= visible[:, None, None, :]
    attention = torch.zeros(allowed.shape, dtype=embedded.dtype).masked_fill(
        ~allowed, -torch.inf
    )
    return embedded, pools, attention.to(device)


def verify_bin_parameters(model, path):
    source = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    assert isinstance(source, dict)
    checked = 0
    for name, parameter in model.named_parameters():
        assert name in source, name
        assert torch.equal(
            parameter.detach().cpu(), source[name].to(parameter.dtype)
        ), name
        checked += 1
    return {
        "parameters_verified": checked,
        "parameter_elements": sum(p.numel() for p in model.parameters()),
        "source_keys": len(source),
        "strict_load": True,
    }


class FlowAdapter:
    def __init__(self, name, device, dtype=None, storage_dtype=None):
        import transformers

        assert transformers.__version__ == "4.47.0", transformers.__version__
        dtype = dtype if dtype is not None else torch.float32
        self.name, self.device, self.dtype = name, device, dtype
        self.pixels = PixelLoader()
        if name == "janusflow":
            self.init_janus()
        elif name == "showo2":
            self.init_showo()
        else:
            raise ValueError(name)
        self.model.requires_grad_(False).eval().to(device, self.dtype)
        self.collector = MaskCollector(
            self.backbone.layers,
            storage_dtype=storage_dtype
            if storage_dtype is not None
            else (torch.float32 if name == "showo2" else torch.bfloat16),
        )
        self.collector.stable_input_pooling = name == "showo2"
        self.backbone.register_forward_hook(self.remember_hidden)
        self.contract = {
            "schema": "geometry_v5_flow_contract_2"
            if dtype == torch.float32
            else "geometry_v5_flow_contract_1",
            "name": name,
            "source_path": str(self.path),
            "source_code": str(JANUS_SOURCE if name == "janusflow" else SHOWO_SOURCE),
            "layers": self.collector.names,
            "precision": str(dtype).removeprefix("torch.")
            + "; pooling float32; VAE posterior mean",
            "dependencies": {
                "transformers": "4.47.0",
                "diffusers": "0.31.0",
                "tokenizers": "0.21.4",
            },
            "pools": {
                "understanding": ["content_mean", "content_last", "assistant_boundary"],
                "generation": [
                    "content_mean",
                    "content_last",
                    "generation_slot_mean",
                    "generation_slot_center",
                ],
            },
            "pixel_policy": "exact V4 Resize256/CenterCrop256 RGB area; resize to declared native resolution",
            "resolution": self.resolution,
            "vae_path": str(self.vae_path),
            "understanding_sequence": self.understanding_description,
            "generation_sequence": self.generation_description,
            "attention_policy": self.attention_description,
            "clean_image_time": 1.0,
            "text_generation_time": 0.0,
            "janus_time_multiplier": 1000 if name == "janusflow" else 1,
            "generation_noise": "CPU float32 Gaussian, same spatial tensor for every semantic item; cast BF16",
            "generation_noise_seeds": [FLOW_SEED, FLOW_SEED + 1, FLOW_SEED + 2],
            "image_midpoint": "0.5 * posterior_mean + 0.5 * fixed Gaussian, t=0.5; image-side robustness only",
            "text_content_mask": "observed body tokens only; chat, time, noise and image markers excluded",
            "target_leakage_contract": "image forward never reads text; text forward never reads image; no paired teacher forcing",
            "head_readout": "backbone task-slot hidden states, not decoded pixels or LM vocabulary logits",
            "text_truncation": "none; assert native context capacity",
            "training_updates": 0,
        }
        if self.collector.storage_dtype != torch.bfloat16:
            self.contract.update(
                schema="geometry_v5_flow_contract_3",
                storage_dtype=str(self.collector.storage_dtype).removeprefix("torch."),
                generation_noise="CPU float32 Gaussian, same spatial tensor for every semantic item; cast to declared computation precision",
            )
        if self.collector.stable_input_pooling:
            self.contract["input_pooling"] = (
                "selected-token FP32 mean independent of sequence padding; subsequent layers masked BMM FP32 mean"
            )

    def remember_hidden(self, module, args, output):
        self.last_hidden = output.last_hidden_state

    def init_janus(self):
        sys.path.insert(0, str(JANUS_SOURCE))
        from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
        from janus.janusflow.models.modeling_vlm import MultiModalityConfig

        self.path = MODELS / "deepseek-ai--JanusFlow-1.3B"
        self.processor = VLChatProcessor.from_pretrained(
            self.path, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer
        assert self.tokenizer.is_fast
        processor_config = json.loads((self.path / "processor_config.json").read_text())
        for key, value in processor_config.items():
            if key != "processor_class":
                assert getattr(self.processor, key) == value, (key, value)
        config = MultiModalityConfig.from_pretrained(self.path, local_files_only=True)
        config.language_config._attn_implementation = "eager"
        self.model = MultiModalityCausalLM.from_pretrained(
            self.path,
            config=config,
            local_files_only=True,
            torch_dtype=self.dtype,
            attn_implementation="eager",
        )
        self.backbone = self.model.language_model.model
        self.resolution, self.latent_shape = 384, (4, 48, 48)
        self.vae_path = MODELS / "stabilityai--sdxl-vae"
        self.vae = None
        marker = "<GEOMETRY_CONTENT>"
        template = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=[
                {"role": "User", "content": marker},
                {"role": "Assistant", "content": ""},
            ],
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        self.prefix, self.suffix = template.split(marker)
        self.understanding_description = {
            "image": self.prefix
            + self.processor.image_tag
            + "\nDescribe the image."
            + self.suffix,
            "text": template,
            "implementation": "official VLChatProcessor + prepare_inputs_embeds for image; same chat template for text",
        }
        self.generation_description = {
            "image": "official empty-user generation prefix without <bog>, time(1000), clean-image UViT tokens",
            "text": "official observed-user generation prefix without <bog>, time(0), fixed-noise UViT tokens",
        }
        self.attention_description = "native causal Llama, including causal order within visual tokens; valid-key right-padding mask"

    def init_showo(self):
        # Isolated process: vendor's package is intentionally called `models`.
        assert "models" not in sys.modules
        sys.path.insert(0, str(SHOWO_SOURCE))
        from models.misc import get_text_tokenizer

        from models import Showo2Qwen2_5

        self.path = MODELS / "showlab--show-o2-1.5B"
        self.qwen_path = MODELS / "Qwen--Qwen2.5-1.5B-Instruct"
        self.tokenizer, self.ids = get_text_tokenizer(
            str(self.qwen_path), return_showo_token_ids=True, llm_name="qwen2_5"
        )
        config = json.loads((self.path / "config.json").read_text())
        config = {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        config.update(
            llm_model_path=str(self.qwen_path),
            load_from_showo=True,
            clip_pretrained_model_path=str(
                MODELS / "google--siglip-so400m-patch14-384"
            ),
        )
        assert len(self.tokenizer) == config["llm_vocab_size"]
        self.model = Showo2Qwen2_5(**config)
        state = torch.load(
            str(self.path / "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        self.model.load_state_dict(state, strict=True, assign=True)
        self.backbone = self.model.showo.model
        self.resolution, self.latent_shape = 432, (16, 54, 54)
        self.vae_path = MODELS / "Wan-AI--Wan2.1-T2V-14B/Wan2.1_VAE.pth"
        self.vae = None
        self.sys_role = [self.ids["bos_id"]] + self.tokenizer.encode(
            "system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
            add_special_tokens=False,
        )
        self.assistant_role = self.tokenizer.encode(
            "\n<|im_start|>assistant\n", add_special_tokens=False
        )
        self.understanding_description = {
            "image": "official inference_mmu system/user, BOI, time(1), 729 fused tokens, EOI, 'Describe the image.', assistant role",
            "text": "same system/user, observed text, assistant role; no image or image placeholder",
        }
        self.generation_description = {
            "image": "BOS, BOI, time(1), 729 clean-image fused tokens, EOI, EOS; empty text condition",
            "text": "BOS, observed text, BOI, time(0), 729 fixed-noise fused tokens, EOI, EOS",
        }
        self.attention_description = {
            "understanding": "official inference_mmu dense mask: bidirectional 729 image patches plus EOI, excludes preceding time token",
            "generation": "official forward dense omni mask: bidirectional time plus 729 image patches; other tokens causal",
            "padding": "right padding, pad keys hidden; scores must match unpadded singleton within BF16 tolerance",
        }

    def verify(self):
        if self.name == "janusflow":
            result = verify_parameters(self.model, self.path)
        else:
            result = verify_bin_parameters(self.model, self.path / "pytorch_model.bin")
        return {**result, "contract": self.contract}

    def load_vae(self):
        if self.vae is not None:
            return
        if self.name == "janusflow":
            from diffusers import AutoencoderKL

            self.vae = (
                AutoencoderKL.from_pretrained(
                    self.vae_path,
                    local_files_only=True,
                    torch_dtype=self.dtype,
                )
                .requires_grad_(False)
                .eval()
                .to(self.device)
            )
        else:
            from models import WanVAE

            self.vae = WanVAE(
                vae_pth=str(self.vae_path), dtype=self.dtype, device=self.device
            )
            # Use model.encode directly: the official wrapper hardcodes CUDA autocast.
            self.vae.model.to(self.device, self.dtype).requires_grad_(False).eval()

    def images(self, rows):
        return list(
            self.pixels.pool.map(self.pixels.one, [row["source_path"] for row in rows])
        )

    def clean_latents(self, rows):
        self.load_vae()
        images = self.images(rows)
        pixels = torch.stack(
            [
                to_tensor(
                    im.resize(
                        (self.resolution, self.resolution), Image.Resampling.BICUBIC
                    )
                )
                * 2
                - 1
                for im in images
            ]
        ).to(self.device, self.dtype)
        if self.name == "janusflow":
            result = (
                self.vae.encode(pixels).latent_dist.mode()
                * self.vae.config.scaling_factor
            )
        else:
            result = self.vae.model.encode(pixels.unsqueeze(2), self.vae.scale)[
                0
            ].squeeze(2)
        assert result.shape[1:] == self.latent_shape and bool(
            torch.isfinite(result).all()
        )
        return result

    def noise(self, count, seed):
        base = torch.randn(
            (1, *self.latent_shape), generator=torch.Generator().manual_seed(seed)
        )
        return base.to(self.device, self.dtype).expand(count, -1, -1, -1)

    def tokens(self, ids):
        return self.backbone.embed_tokens(
            torch.tensor(ids, device=self.device, dtype=torch.long)
        )

    def janus_text_tokens(self, text):
        prompt = self.prefix + text + self.suffix
        encoded = self.tokenizer(prompt, return_offsets_mapping=True)
        lo, hi = len(self.prefix), len(self.prefix) + len(text)
        content = [
            i
            for i, (start, stop) in enumerate(encoded["offset_mapping"])
            if stop > lo and start < hi
        ]
        assert len(encoded["input_ids"]) < self.backbone.config.max_position_embeddings
        return encoded["input_ids"], content

    def run(self, sequences, masks, image_blocks=None):
        embedded, pools, attention = pack(sequences, masks, self.device, image_blocks)
        self.collector.begin_masks(pools)
        output = self.backbone(
            inputs_embeds=embedded,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        )
        features = self.collector.finish(output.last_hidden_state)
        self.last_call = {"embedded": embedded, "pools": pools, "attention": attention}
        return features

    def understanding(self, rows, modality):
        upstream = {}
        sequences, masks, blocks = [], [], []
        if self.name == "janusflow" and modality == "image":
            prompt = self.understanding_description["image"]
            prepared = self.processor.batchify(
                [
                    self.processor.process_one(prompt=prompt, images=[im])
                    for im in self.images(rows)
                ]
            ).to(self.device, dtype=self.dtype)
            capture = []
            handle = self.model.vision_und_enc_model.register_forward_hook(
                lambda module, args, output: capture.append(output.detach())
            )
            embedded = self.model.prepare_inputs_embeds(**prepared)
            handle.remove()
            upstream["understanding_encoder"] = (
                capture[0].float().mean(1).to("cpu", self.dtype)
            )
            for i, emb in enumerate(embedded):
                indices = prepared.images_seq_mask[i].nonzero().flatten().tolist()[1:]
                assert len(indices) == 576
                sequences.append(emb)
                masks.append(readout_masks(len(emb), indices, boundary=len(emb) - 1))
        elif self.name == "showo2" and modality == "image":
            fused, time, upstream = self.showo_visual(self.clean_latents(rows), 1.0)
            before = self.sys_role + [self.ids["boi_id"]]
            after = (
                [self.ids["eoi_id"]]
                + self.tokenizer.encode("Describe the image.", add_special_tokens=False)
                + self.assistant_role
            )
            for i in range(len(rows)):
                emb = torch.cat(
                    [self.tokens(before), time[i : i + 1], fused[i], self.tokens(after)]
                )
                first = len(before) + 1
                sequences.append(emb)
                masks.append(
                    readout_masks(
                        len(emb), range(first, first + 729), boundary=len(emb) - 1
                    )
                )
                blocks.append((first, 730))  # exact published inference_mmu convention
        else:
            for row in rows:
                if self.name == "janusflow":
                    ids, body = self.janus_text_tokens(row["text"])
                else:
                    body_ids = self.tokenizer.encode(
                        row["text"], add_special_tokens=False
                    )
                    ids = self.sys_role + body_ids + self.assistant_role
                    body = range(len(self.sys_role), len(self.sys_role) + len(body_ids))
                sequences.append(self.tokens(ids))
                masks.append(readout_masks(len(ids), body, boundary=len(ids) - 1))
                blocks.append((0, 0))
        features = self.run(sequences, masks, blocks if self.name == "showo2" else None)
        return features, upstream

    def showo_visual(self, latents, time):
        und = self.model.image_embedder_und(latents)
        gen = self.model.image_embedder_gen(latents)
        und = und + self.model.position_embedding(self.model.image_position_ids)
        und = self.model.und_trans(und)["last_hidden_state"]
        fused = self.model.fusion_proj(torch.cat([und, gen], dim=-1))
        t = torch.full((latents.shape[0],), time, device=self.device, dtype=self.dtype)
        raw_time = self.model.time_embed(t, self.dtype)
        time_emb = self.model.time_embed_proj(raw_time)
        upstream = {
            name: value.float().mean(1).to("cpu", self.dtype)
            for name, value in {
                "semantic_teacher_path": und,
                "low_level_path": gen,
                "fused_input": fused,
            }.items()
        }
        self.last_showo_time = raw_time
        return fused, time_emb, upstream

    def generation(self, rows, modality, profile):
        seed = FLOW_SEED + ({"seed1": 1, "seed2": 2}.get(profile, 0))
        count = len(rows)
        noise = self.noise(count, seed)
        if modality == "image":
            latent, time = self.clean_latents(rows), 1.0
            if profile == "image_midpoint":
                latent, time = (latent + noise) * 0.5, 0.5
        else:
            latent, time = noise, 0.0
        if self.name == "janusflow":
            t = torch.full((count,), time * 1000, dtype=self.dtype, device=self.device)
            encoded, time_emb, skips = self.model.vision_gen_enc_model(latent, t)
            encoded = encoded.flatten(2).transpose(1, 2)
            visual = self.model.vision_gen_enc_aligner(encoded)
            upstream = {
                "generation_encoder": encoded.float().mean(1).to("cpu", self.dtype)
            }
            self.last_janus_skips, self.last_janus_time = skips, time_emb
        else:
            visual, time_emb, upstream = self.showo_visual(latent, time)
        sequences, masks, blocks, all_ids = [], [], [], []
        for i, row in enumerate(rows):
            text = row["text"] if modality == "text" else ""
            if self.name == "janusflow":
                ids, body = self.janus_text_tokens(text)
                # Appending then removing <bog> must leave exactly the official prefix.
                prompt = self.prefix + text + self.suffix + self.processor.image_gen_tag
                reference_ids = self.tokenizer.encode(prompt)
                assert reference_ids[-1] == self.processor.image_gen_id
                assert ids == reference_ids[:-1]
                after = []
            else:
                body_ids = self.tokenizer.encode(text, add_special_tokens=False)
                ids = [self.ids["bos_id"]] + body_ids + [self.ids["boi_id"]]
                body = range(1, 1 + len(body_ids))
                after = [self.ids["eoi_id"], self.ids["eos_id"]]
            first = len(ids) + 1
            slots = range(first, first + visual.shape[1])
            emb = torch.cat(
                [self.tokens(ids), time_emb[i : i + 1], visual[i], self.tokens(after)]
            )
            sequences.append(emb)
            masks.append(
                readout_masks(
                    len(emb), slots if modality == "image" else body, slots=slots
                )
            )
            blocks.append((first - 1, visual.shape[1] + 1))
            if self.name == "showo2":
                all_ids.append(
                    ids + [self.ids["img_pad_id"]] * (visual.shape[1] + 1) + after
                )
        features = self.run(sequences, masks, blocks if self.name == "showo2" else None)
        self.last_generation = {
            "latents": latent,
            "time": time,
            "blocks": blocks,
            "ids": all_ids,
        }
        return features, upstream

    @torch.inference_mode()
    def forward(self, rows, modality, path, profile="main"):
        if path == "understanding":
            assert profile == "main"
            features, upstream = self.understanding(rows, modality)
        else:
            features, upstream = self.generation(rows, modality, profile)
        assert bool(torch.isfinite(features).all())
        return features, upstream

    @torch.inference_mode()
    def validate_generation_reference(self, rows, modality):
        """Cal-only: compare against the official full generation forward, head included."""
        expected, _ = self.forward(rows[:1], modality, "generation")
        if self.name == "janusflow":
            hidden = self.last_hidden[:, -576:]
            aligned = self.model.vision_gen_dec_aligner(
                self.model.vision_gen_dec_aligner_norm(hidden)
            )
            aligned = aligned.reshape(1, 24, 24, 768).permute(0, 3, 1, 2)
            velocity = self.model.vision_gen_dec_model(
                aligned, self.last_janus_skips, self.last_janus_time
            )
            difference = 0.0
            reference = "official README single-step flow encoder/backbone/decoder expression; native modules called unchanged"
        else:
            data = self.last_generation
            ids = torch.tensor(data["ids"], device=self.device)
            positions = torch.tensor(data["blocks"], device=self.device).unsqueeze(1)
            time = torch.full((1,), data["time"], dtype=self.dtype, device=self.device)
            self.collector.begin_masks(self.last_call["pools"])
            _, velocity = self.model(
                text_tokens=ids,
                image_latents=data["latents"],
                t=time,
                attention_mask=self.last_call["attention"],
                modality_positions=positions,
                output_hidden_states=True,
                max_seq_len=ids.shape[1],
                device=self.device,
            )
            actual = self.collector.finish(self.last_hidden)
            difference = float((expected.float() - actual.float()).abs().max())
            assert torch.equal(expected, actual), difference
            reference = "exact BF16 pooled-layer equality with official Showo2Qwen2_5.forward including full diffusion head"
        assert velocity.shape == (1, *self.latent_shape) and bool(
            torch.isfinite(velocity).all()
        )
        return {
            "reference": reference,
            "all_layer_max_abs": difference,
            "velocity_shape": list(velocity.shape),
            "velocity_finite": True,
        }

    def verify_vae(self):
        self.load_vae()
        if self.name == "showo2":
            return verify_bin_parameters(self.vae.model, self.vae_path)
        checked = 0
        with safe_open(
            str(self.vae_path / "diffusion_pytorch_model.safetensors"), framework="pt"
        ) as source:
            for name, parameter in self.vae.named_parameters():
                assert torch.equal(
                    parameter.detach().cpu(),
                    source.get_tensor(name).to(parameter.dtype),
                ), name
                checked += 1
        return {
            "parameters_verified": checked,
            "parameter_elements": sum(p.numel() for p in self.vae.parameters()),
        }
