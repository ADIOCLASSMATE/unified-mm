import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from torch import nn
from transformers import Qwen3Config

from models.modeling_model.mar_flowloss import FlowLoss
from models.modeling_model.modeling_selfless_flow import ImageTokenEmbedder, Qwen3ForCausalLM, Qwen3Model
from utils.dataset_combined_flow import CombinedBatchDataLoader, TextArrowDataset, collate_text_arrow
from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset, collate_imagenet_flow_cache
from utils.utils import get_selfless_mask


def tiny_qwen3_config():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    return config


class CaptureLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        X0_hidden_states,
        XT_hidden_states,
        attention_mask,
        **kwargs,
    ):
        self.calls.append(
            {
                "x0": X0_hidden_states.detach().clone(),
                "xt": None if XT_hidden_states is None else XT_hidden_states.detach().clone(),
                "attention_mask": attention_mask,
            }
        )
        return X0_hidden_states, XT_hidden_states


class FakeLinear:
    def __init__(self, dtype=torch.float32):
        self.weight = torch.empty(1, dtype=dtype)


class FakeFinalLayer:
    def __init__(self):
        self.linear = FakeLinear()


class FakeImageFlowHead:
    def __init__(self, latent_dim):
        self.net = types.SimpleNamespace(final_layer=FakeFinalLayer())
        self.latent_dim = latent_dim
        self.sample_calls = []

    def sample(self, z, temperature=1.0, cfg=1.0, solver=None, num_steps=None):
        self.sample_calls.append(
            {
                "z": z.detach().clone(),
                "temperature": temperature,
                "cfg": cfg,
                "solver": solver,
                "num_steps": num_steps,
            }
        )
        out_batch = z.shape[0] // 2 if cfg != 1.0 else z.shape[0]
        return torch.zeros(out_batch, self.latent_dim, device=z.device, dtype=z.dtype)


class FakeCallableFlowHead(nn.Module):
    def __init__(self, loss_value=5.0):
        super().__init__()
        self.loss_value = float(loss_value)
        self.last_forward_stats = {}

    def forward(self, target, z):
        loss = target.float().sum() * 0.0 + z.float().sum() * 0.0 + self.loss_value
        self.last_forward_stats = {
            "flow/v_mse": loss.detach(),
            "flow/v_pred_rms": loss.detach() + 1.0,
        }
        return loss


class FakeConditionEmbedder(nn.Module):
    def add_diffusion_pos(self, z, image_local_positions):
        return z


class FakeCausalInnerModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.image_token_embedder = FakeConditionEmbedder()

    def forward(self, X0_input_ids, **kwargs):
        bsz, seq_len = X0_input_ids.shape
        hidden = torch.ones(
            bsz,
            seq_len,
            self.hidden_size,
            device=X0_input_ids.device,
            dtype=torch.float32,
        )
        return types.SimpleNamespace(last_hidden_state=hidden, past_key_values=None, hidden_states=None, attentions=None)

    def image_local_positions(self, token_types):
        is_image = token_types == 1
        positions = is_image.long().cumsum(dim=1) - 1
        return positions.masked_fill(~is_image, 0)


class FakeTokenizer:
    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = 100 + len(self.vocab)
            ids.append(self.vocab[word])
        return ids


class FakeInnerModel:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.calls = []

    def __call__(self, **kwargs):
        attention_mask = kwargs["attention_mask"]
        image_uncond = bool(
            torch.is_tensor(attention_mask)
            and attention_mask.numel() > 0
            and attention_mask.detach().float().max().item() >= 100.0
        )
        self.calls.append(
            {
                "calculate_likelihood": kwargs["calculate_likelihood"],
                "sigma": attention_mask,
                "image_latent_mask": kwargs["image_latent_mask"].detach().clone(),
                "image_latents": kwargs["image_latents"].detach().clone(),
                "token_types": kwargs["token_types"].detach().clone(),
                "image_uncond": image_uncond,
            }
        )
        input_ids = kwargs["X0_input_ids"]
        hidden = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.hidden_size,
            device=input_ids.device,
            dtype=kwargs["image_latents"].dtype,
        )
        hidden.fill_(2.0 if image_uncond else 1.0)
        return types.SimpleNamespace(last_hidden_state=hidden)


class FakeTextBackbone:
    def __init__(self, token_plan):
        self.token_plan = list(token_plan)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["X0_input_ids"]
        token_idx = min(len(self.calls) - 1, len(self.token_plan) - 1)
        hidden = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            1,
            device=input_ids.device,
            dtype=torch.float32,
        )
        hidden[:, -1, 0] = float(self.token_plan[token_idx])
        return types.SimpleNamespace(last_hidden_state=hidden)


class FakeTextHead:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, hidden):
        token_id = int(hidden[0, 0].item())
        logits = torch.full((hidden.shape[0], self.vocab_size), -1000.0, device=hidden.device)
        logits[:, token_id] = 1000.0
        return logits


class SelflessFlowBehaviorTest(unittest.TestCase):
    def test_full_imagenet_cache_collate_trains_boi_and_eos_but_not_eoi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latents = torch.arange(16, dtype=torch.float16).view(2, 4, 2)
            torch.save({"latents": latents, "img_ids": torch.tensor([1, 2])}, root / "latents.pt")
            (root / "manifest.jsonl").write_text(
                '{"img_id": 1, "synset": "n00000001"}\n'
                '{"img_id": 2, "synset": "n00000002"}\n'
            )
            (root / "mapping.txt").write_text(
                "n00000001 test class, alternate name\n"
                "n00000002 other class\n"
            )

            dataset = ImageNetFlowCacheDataset(
                cache_path=str(root / "latents.pt"),
                tokenizer=FakeTokenizer(),
                boi_token_id=11,
                eoi_token_id=12,
                mask_token_id=13,
                eos_token_id=14,
                image_tokens_per_img=4,
                image_latent_dim=2,
                manifest_jsonl=str(root / "manifest.jsonl"),
                prompt_template="a photo of {class_name}",
                synset_mapping_path=str(root / "mapping.txt"),
                max_seq_length=16,
            )

            batch = collate_imagenet_flow_cache([dataset[0]], pad_to_length=16)

        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        token_types = batch["token_types"][0]
        sigma = batch["sigma"][0]
        image_latents = batch["image_latents"][0]

        prompt_len = 5
        image_start = prompt_len + 1
        eoi_pos = image_start + 4
        eos_pos = eoi_pos + 1

        self.assertEqual(input_ids[prompt_len].item(), 11)
        self.assertEqual(input_ids[eoi_pos].item(), 12)
        self.assertEqual(input_ids[eos_pos].item(), 14)

        self.assertEqual(labels[prompt_len].item(), 11)
        self.assertEqual(labels[eoi_pos].item(), -100)
        self.assertEqual(labels[eos_pos].item(), 14)
        self.assertTrue(torch.equal(labels[image_start:eoi_pos], torch.full((4,), -100)))

        self.assertTrue(torch.equal(token_types[image_start:eoi_pos], torch.ones(4, dtype=torch.uint8)))
        self.assertTrue(torch.equal(image_latents[image_start:eoi_pos], latents[0]))

        self.assertEqual(sigma[prompt_len].item(), prompt_len)
        self.assertEqual(sigma[eoi_pos].item(), prompt_len + 1)
        self.assertTrue(torch.all(sigma[image_start:eoi_pos] > sigma[eoi_pos]))
        self.assertTrue(torch.all(sigma[eos_pos] > sigma[image_start:eoi_pos]))

    def test_full_imagenet_cache_collate_supports_image_first_templates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latents = torch.arange(8, dtype=torch.float16).view(1, 4, 2)
            torch.save({"latents": latents, "img_ids": torch.tensor([1])}, root / "latents.pt")
            (root / "manifest.jsonl").write_text('{"img_id": 1, "synset": "n00000001"}\n')
            (root / "mapping.txt").write_text("n00000001 test class\n")

            tokenizer = FakeTokenizer()
            dataset = ImageNetFlowCacheDataset(
                cache_path=str(root / "latents.pt"),
                tokenizer=tokenizer,
                boi_token_id=11,
                eoi_token_id=12,
                mask_token_id=13,
                eos_token_id=14,
                image_tokens_per_img=4,
                image_latent_dim=2,
                manifest_jsonl=str(root / "manifest.jsonl"),
                prompt_templates=["{image} this image shows {class_name}"],
                synset_mapping_path=str(root / "mapping.txt"),
                max_seq_length=16,
            )

            batch = collate_imagenet_flow_cache([dataset[0]], pad_to_length=16)

        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        token_types = batch["token_types"][0]
        sigma = batch["sigma"][0]

        image_start = 1
        eoi_pos = image_start + 4
        suffix_start = eoi_pos + 1
        eos_pos = suffix_start + 5

        self.assertEqual(input_ids[0].item(), 11)
        self.assertEqual(input_ids[eoi_pos].item(), 12)
        self.assertEqual(input_ids[eos_pos].item(), 14)
        self.assertTrue(torch.equal(token_types[suffix_start:eos_pos], torch.zeros(5, dtype=torch.uint8)))
        self.assertTrue(torch.equal(labels[suffix_start:eos_pos], input_ids[suffix_start:eos_pos]))
        self.assertEqual(labels[eoi_pos].item(), -100)
        self.assertTrue(torch.all(sigma[image_start:eoi_pos] > sigma[eoi_pos]))
        self.assertTrue(torch.all(sigma[suffix_start:eos_pos] > sigma[image_start:eoi_pos].max()))
        self.assertTrue(torch.all(sigma[eos_pos] > sigma[suffix_start:eos_pos]))

    def test_full_imagenet_cache_uses_category_prompt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latents = torch.arange(8, dtype=torch.float16).view(1, 4, 2)
            torch.save({"latents": latents, "img_ids": torch.tensor([1])}, root / "latents.pt")
            (root / "manifest.jsonl").write_text('{"img_id": 1, "synset": "n00000001"}\n')
            (root / "mapping.txt").write_text("n00000001 tabby cat\n")
            (root / "prompts.yaml").write_text(
                "fallback_templates:\n"
                "  - 'fallback object {image}'\n"
                "groups:\n"
                "  - name: cats\n"
                "    keywords: ['cat']\n"
                "    templates:\n"
                "      - 'cat specific prefix {class_name} {image} cat specific suffix'\n"
            )

            tokenizer = FakeTokenizer()
            dataset = ImageNetFlowCacheDataset(
                cache_path=str(root / "latents.pt"),
                tokenizer=tokenizer,
                boi_token_id=11,
                eoi_token_id=12,
                mask_token_id=13,
                eos_token_id=14,
                image_tokens_per_img=4,
                image_latent_dim=2,
                manifest_jsonl=str(root / "manifest.jsonl"),
                prompt_templates_path=str(root / "prompts.yaml"),
                synset_mapping_path=str(root / "mapping.txt"),
                max_seq_length=32,
            )

            batch = collate_imagenet_flow_cache([dataset[0]], pad_to_length=32)

        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        token_types = batch["token_types"][0]
        text_tokens = input_ids[token_types == 0].tolist()
        text_labels = labels[(token_types == 0) & (labels != -100)].tolist()

        expected = tokenizer.encode("cat specific prefix tabby cat cat specific suffix", add_special_tokens=False)
        self.assertEqual(text_tokens, expected)
        self.assertEqual(text_labels, expected)
        self.assertEqual(
            batch["input_ids"][0, 0].item(),
            tokenizer.encode("cat", add_special_tokens=False)[0],
        )

    def test_imagenet_latent_hflip_augmentation_is_train_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latents = torch.tensor(
                [
                    [[1.0], [2.0], [3.0], [4.0]],
                    [[10.0], [20.0], [30.0], [40.0]],
                ],
                dtype=torch.float16,
            )
            torch.save({"latents": latents, "img_ids": torch.tensor([1, 2])}, root / "latents.pt")

            dataset = ImageNetFlowCacheDataset(
                cache_path=str(root / "latents.pt"),
                tokenizer=FakeTokenizer(),
                boi_token_id=11,
                eoi_token_id=12,
                mask_token_id=13,
                eos_token_id=14,
                image_tokens_per_img=4,
                image_latent_dim=1,
                latent_hflip_prob=1.0,
            )
            dataset.set_augmentation_train_size(1)

            train_item = dataset[0]
            val_item = dataset[1]

        expected_flipped = torch.tensor([[2.0], [1.0], [4.0], [3.0]], dtype=torch.float16)
        self.assertTrue(torch.equal(train_item["image_latents"], expected_flipped))
        self.assertTrue(torch.equal(val_item["image_latents"], latents[1]))

    def test_image_token_embedder_handles_bfloat16_weights(self):
        projector = ImageTokenEmbedder(latent_dim=4, hidden_size=8, projector_width=16, image_tokens_per_img=4).to(dtype=torch.bfloat16)
        latents = torch.randn(5, 4, dtype=torch.float32)
        positions = torch.tensor([0, 1, 2, 3, 0])
        out = projector(latents, positions)
        self.assertEqual(tuple(out.shape), (5, 8))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_x0_image_latent_assignment_casts_projector_dtype_to_embedding_dtype(self):
        config = tiny_qwen3_config()
        model = Qwen3Model(config)
        model.embed_tokens.to(dtype=torch.bfloat16)
        model.image_token_embedder.to(dtype=torch.float32)

        input_ids = torch.tensor([[1, 2, 3, 4]])
        token_types = torch.tensor([[0, 1, 1, 2]])
        image_latents = torch.randn(1, 4, 4, dtype=torch.float32)

        out = model._build_x0_inputs_embeds(
            input_ids=input_ids,
            token_types=token_types,
            image_latents=image_latents,
            image_latent_mask=None,
        )

        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(tuple(out.shape), (1, 4, config.hidden_size))

    def test_text_arrow_dataset_slices_and_collates_as_text_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            HFDataset.from_dict({"input_ids": [list(range(20)), list(range(100, 112))]}).save_to_disk(root)
            dataset = TextArrowDataset(
                tokenized_path=str(root),
                max_seq_length=8,
                pad_token_id=0,
                seed=123,
            )
            batch = collate_text_arrow([dataset[0], dataset[1]], pad_token_id=0, pad_to_multiple_of=8)

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertTrue(torch.equal(batch["token_types"], torch.zeros(2, 8, dtype=torch.uint8)))
        self.assertTrue(torch.equal(batch["sigma"][0], torch.arange(8)))
        self.assertTrue(torch.equal(batch["labels"], batch["input_ids"]))
        self.assertEqual(batch["pack_stats"].tolist(), [16, 0, 0, 8])

    def test_text_arrow_collate_supports_fixed_pad_to_length(self):
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "token_types": torch.zeros(3, dtype=torch.uint8),
                "sigma": torch.arange(3, dtype=torch.long),
                "labels": torch.tensor([1, 2, 3], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([4, 5], dtype=torch.long),
                "token_types": torch.zeros(2, dtype=torch.uint8),
                "sigma": torch.arange(2, dtype=torch.long),
                "labels": torch.tensor([4, 5], dtype=torch.long),
            },
        ]

        out = collate_text_arrow(batch, pad_token_id=0, pad_to_length=8, pad_to_multiple_of=8)

        self.assertEqual(tuple(out["input_ids"].shape), (2, 8))
        self.assertEqual(out["pack_stats"].tolist(), [5, 0, 11, 8])
        self.assertTrue(torch.equal(out["token_types"][:, 3:], torch.full((2, 5), 3, dtype=torch.uint8)))

    def test_combined_loader_mixes_whole_batches(self):
        image_batch = {
            "input_ids": torch.ones(2, 6, dtype=torch.long),
            "token_types": torch.ones(2, 6, dtype=torch.uint8),
            "sigma": torch.zeros(2, 6, dtype=torch.long),
            "labels": torch.full((2, 6), -100, dtype=torch.long),
            "image_latents": torch.zeros(2, 6, 4),
        }
        text_batch = {
            "input_ids": torch.ones(2, 8, dtype=torch.long) * 2,
            "token_types": torch.zeros(2, 8, dtype=torch.uint8),
            "sigma": torch.arange(8).unsqueeze(0).expand(2, 8),
            "labels": torch.ones(2, 8, dtype=torch.long) * 2,
        }
        image_loader = DataLoader([image_batch, image_batch], batch_size=None)
        text_loader = DataLoader([text_batch, text_batch], batch_size=None)
        combined = CombinedBatchDataLoader(
            image_loader=image_loader,
            text_loader=text_loader,
            text_batch_ratio=1.0,
            seed=1,
            mode="train",
        )

        batches = list(combined)
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(batch["batch_source"] == "text" for batch in batches))
        self.assertTrue(all("image_latents" not in batch for batch in batches))
        self.assertTrue(all(torch.equal(batch["token_types"], torch.zeros_like(batch["token_types"])) for batch in batches))

    def test_combined_loader_accumulation_schedule_is_deterministic(self):
        image_batch = {
            "input_ids": torch.ones(2, 6, dtype=torch.long),
            "token_types": torch.ones(2, 6, dtype=torch.uint8),
            "sigma": torch.zeros(2, 6, dtype=torch.long),
            "labels": torch.full((2, 6), -100, dtype=torch.long),
            "image_latents": torch.zeros(2, 6, 4),
        }
        text_batch = {
            "input_ids": torch.ones(2, 8, dtype=torch.long) * 2,
            "token_types": torch.zeros(2, 8, dtype=torch.uint8),
            "sigma": torch.arange(8).unsqueeze(0).expand(2, 8),
            "labels": torch.ones(2, 8, dtype=torch.long) * 2,
        }
        image_loader = DataLoader([image_batch] * 8, batch_size=None)
        text_loader = DataLoader([text_batch] * 2, batch_size=None)
        combined = CombinedBatchDataLoader(
            image_loader=image_loader,
            text_loader=text_loader,
            text_batch_ratio=0.25,
            seed=1,
            mode="train",
            batch_schedule="accumulation",
            accumulation_steps=4,
            text_batches_per_accumulation=1,
        )

        sources = [batch["batch_source"] for batch in combined]
        self.assertEqual(
            sources,
            ["image", "image", "image", "text", "image", "image", "image", "text"],
        )

    def test_combined_loader_ratio_accumulation_matches_long_run_ratio(self):
        image_batch = {
            "input_ids": torch.ones(2, 6, dtype=torch.long),
            "token_types": torch.ones(2, 6, dtype=torch.uint8),
            "sigma": torch.zeros(2, 6, dtype=torch.long),
            "labels": torch.full((2, 6), -100, dtype=torch.long),
            "image_latents": torch.zeros(2, 6, 4),
        }
        text_batch = {
            "input_ids": torch.ones(2, 8, dtype=torch.long) * 2,
            "token_types": torch.zeros(2, 8, dtype=torch.uint8),
            "sigma": torch.arange(8).unsqueeze(0).expand(2, 8),
            "labels": torch.ones(2, 8, dtype=torch.long) * 2,
        }
        image_loader = DataLoader([image_batch] * 40, batch_size=None)
        text_loader = DataLoader([text_batch] * 8, batch_size=None)
        combined = CombinedBatchDataLoader(
            image_loader=image_loader,
            text_loader=text_loader,
            text_batch_ratio=0.15,
            seed=1,
            mode="train",
            batch_schedule="accumulation",
            accumulation_steps=2,
            text_batches_per_accumulation="ratio",
        )

        sources = [batch["batch_source"] for batch in combined]

        self.assertEqual(sources.count("text"), 6)
        self.assertEqual(sources.count("image"), 34)
        for group_start in range(0, len(sources), 2):
            self.assertLessEqual(sources[group_start:group_start + 2].count("text"), 1)

    def test_mar_flow_head_predicts_velocity_and_samples_latents(self):
        head = FlowLoss(
            target_channels=4,
            z_channels=8,
            depth=1,
            width=8,
            num_sampling_steps="2",
            time_sampling="uniform",
        )
        target = torch.randn(3, 4)
        z = torch.randn(3, 8)
        t = torch.full((3,), 0.5)
        out = head.velocity(target, t, z)
        loss = head(target, z)
        sample = head.sample(z, temperature=1.0, cfg=1.0)

        self.assertEqual(tuple(out.shape), (3, 4))
        self.assertEqual(tuple(sample.shape), (3, 4))
        self.assertEqual(loss.dim(), 0)
        self.assertIn("flow/v_mse", head.last_forward_stats)

    def test_causal_lm_keeps_multimodal_logging_fields_in_model_output_mapping(self):
        from accelerate.utils.operations import convert_to_fp32

        config = tiny_qwen3_config()
        config.lambda_text = 0.0
        config.lambda_image = 1.0
        model = Qwen3ForCausalLM(config)
        model.model = FakeCausalInnerModel(config.hidden_size)
        model.image_flow_head = FakeCallableFlowHead(loss_value=5.0)

        input_ids = torch.tensor([[1, 8, 8, 2]])
        token_types = torch.tensor([[0, 1, 1, 2]])
        labels = torch.full_like(input_ids, -100)
        image_latents = torch.randn(1, 4, config.image_latent_dim)

        output = model(
            X0_input_ids=input_ids,
            attention_mask=object(),
            labels=labels,
            token_types=token_types,
            image_latents=image_latents,
            calculate_likelihood=True,
            return_logits=False,
        )
        converted = convert_to_fp32(output)

        self.assertIn("last_hidden_state", converted)
        self.assertIn("per_modality_loss", converted)
        self.assertIn("flow_debug_stats", converted)
        self.assertAlmostEqual(converted.per_modality_loss["image_loss"].item(), 5.0)
        self.assertAlmostEqual(converted.flow_debug_stats["flow/v_mse"].item(), 5.0)

    def test_training_likelihood_uses_xt_all_mask_and_x0_real_content(self):
        config = tiny_qwen3_config()
        model = Qwen3Model(config)
        capture = CaptureLayer()
        model.layers = nn.ModuleList([capture])
        model.norm = nn.Identity()
        model.train()

        input_ids = torch.tensor([[1, 2, 3, 4]])
        token_types = torch.tensor([[0, 1, 1, 2]])
        image_latents = torch.arange(16, dtype=torch.float32).view(1, 4, 4)

        output = model(
            X0_input_ids=input_ids,
            attention_mask=object(),
            token_types=token_types,
            image_latents=image_latents,
            calculate_likelihood=True,
        )

        call = capture.calls[-1]
        expected_xt = model.embed_tokens(
            torch.full_like(input_ids, config.mask_token_id)
        )
        image_positions = model.image_local_positions(token_types)
        image_mask = token_types == 1
        image_mask_embedding = model._image_mask_embedding(
            input_ids.device,
            model.image_token_embedder.weight_dtype,
        )
        expected_xt[image_mask] = model.image_token_embedder.embed_mask(
            image_positions[image_mask],
            image_mask_embedding,
        )
        self.assertTrue(torch.allclose(call["xt"], expected_xt))

        expected_x0 = model.embed_tokens(input_ids.masked_fill(image_mask, config.image_mask_token_id))
        expected_x0[image_mask] = model.image_token_embedder(
            image_latents[image_mask],
            image_positions[image_mask],
        )
        self.assertTrue(torch.allclose(call["x0"], expected_x0))

        self.assertTrue(torch.allclose(output.last_hidden_state, expected_xt))

    def test_image_token_ids_do_not_leak_into_backbone_embeddings(self):
        config = tiny_qwen3_config()
        model = Qwen3Model(config)
        capture = CaptureLayer()
        model.layers = nn.ModuleList([capture])
        model.norm = nn.Identity()
        model.train()

        token_types = torch.tensor([[0, 1, 1, 2]])
        image_latents = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
        attention_mask = object()

        first_ids = torch.tensor([[1, 2, 3, 4]])
        second_ids = torch.tensor([[1, 20, 21, 4]])

        model(
            X0_input_ids=first_ids,
            attention_mask=attention_mask,
            token_types=token_types,
            image_latents=image_latents,
            calculate_likelihood=True,
        )
        first_call = capture.calls[-1]

        model(
            X0_input_ids=second_ids,
            attention_mask=attention_mask,
            token_types=token_types,
            image_latents=image_latents,
            calculate_likelihood=True,
        )
        second_call = capture.calls[-1]

        self.assertIs(first_call["attention_mask"], attention_mask)
        self.assertIs(second_call["attention_mask"], attention_mask)
        self.assertTrue(torch.allclose(first_call["x0"], second_call["x0"]))
        self.assertTrue(torch.allclose(first_call["xt"], second_call["xt"]))

    def test_single_stream_unfilled_image_tokens_use_same_mask_embedding(self):
        config = tiny_qwen3_config()
        model = Qwen3Model(config)
        capture = CaptureLayer()
        model.layers = nn.ModuleList([capture])
        model.norm = nn.Identity()
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4]])
        token_types = torch.tensor([[0, 1, 1, 2]])
        image_latents = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
        image_latent_mask = torch.tensor([[False, True, False, False]])

        model(
            X0_input_ids=input_ids,
            attention_mask=object(),
            token_types=token_types,
            image_latents=image_latents,
            image_latent_mask=image_latent_mask,
            calculate_likelihood=False,
        )

        call = capture.calls[-1]
        self.assertIsNone(call["xt"])

        image_positions = model.image_local_positions(token_types)
        image_mask = token_types == 1
        image_mask_embedding = model._image_mask_embedding(
            input_ids.device,
            model.image_token_embedder.weight_dtype,
        )
        expected_x0 = model.embed_tokens(input_ids.masked_fill(image_mask, config.image_mask_token_id))
        expected_x0[image_mask] = model.image_token_embedder.embed_mask(
            image_positions[image_mask],
            image_mask_embedding,
        )
        visible_image = image_mask & image_latent_mask
        expected_x0[visible_image] = model.image_token_embedder(
            image_latents[visible_image],
            image_positions[visible_image],
        )
        self.assertTrue(torch.allclose(call["x0"], expected_x0))
        self.assertTrue(torch.allclose(call["x0"][:, 2], expected_x0[:, 2]))

    def test_image_uncond_mask_image_queries_only_see_same_image_tokens(self):
        boi = 11
        eoi = 12
        input_ids = torch.tensor(
            [
                [101, boi, 8, 8, eoi, 102, boi, 8, 8, 8, eoi, 103],
                [101, boi, 8, 8, eoi, 102, boi, 8, 8, 8, eoi, 103],
            ]
        )
        token_types = torch.tensor(
            [
                [0, 2, 1, 1, 2, 0, 2, 1, 1, 1, 2, 0],
                [0, 2, 1, 1, 2, 0, 2, 1, 1, 1, 2, 0],
            ]
        )
        sigma = torch.arange(input_ids.shape[1]).unsqueeze(0).repeat(2, 1)
        mask = get_selfless_mask(
            sigma=sigma,
            seq_len=input_ids.shape[1],
            device="cpu",
            input_ids=input_ids,
            token_types=token_types,
            boi_token_id=boi,
            image_uncond_rows=torch.tensor([True, False]),
        )

        def allowed(batch, q_idx, kv_idx):
            value = mask.mask_mod(
                torch.tensor(batch),
                torch.tensor(0),
                torch.tensor(q_idx),
                torch.tensor(kv_idx),
            )
            return bool(value.item() if torch.is_tensor(value) else value)

        second_image_third_token = 9
        self.assertEqual(
            [idx for idx in range(input_ids.shape[1]) if allowed(0, second_image_third_token, idx)],
            [7, 8],
        )
        self.assertFalse(allowed(0, second_image_third_token, 0))   # text before first BOI
        self.assertFalse(allowed(0, second_image_third_token, 2))   # previous image token
        self.assertFalse(allowed(0, second_image_third_token, 5))   # text before current BOI
        self.assertFalse(allowed(0, second_image_third_token, 6))   # current BOI
        self.assertFalse(allowed(0, second_image_third_token, 10))  # current EOI
        self.assertEqual([idx for idx in range(input_ids.shape[1]) if allowed(0, 7, idx)], [])
        self.assertEqual([idx for idx in range(input_ids.shape[1]) if allowed(0, 11, idx)], list(range(11)))
        self.assertEqual(
            [idx for idx in range(input_ids.shape[1]) if allowed(1, second_image_third_token, idx)],
            list(range(second_image_third_token)),
        )

    def test_image_flow_cfg_requires_paired_unconditional_condition(self):
        latent_dim = 4
        head = FakeImageFlowHead(latent_dim=latent_dim)
        dummy = types.SimpleNamespace(image_flow_head=head)
        z = torch.ones(3, latent_dim)
        z_uncond = torch.full_like(z, 2.0)

        out = Qwen3ForCausalLM.sample_image_flow_with_cfg(
            dummy,
            z,
            z_uncond=z_uncond,
            temperature=0.7,
            cfg=3.0,
            solver="euler",
            num_steps=7,
        )

        self.assertEqual(tuple(out.shape), (3, latent_dim))
        call = head.sample_calls[-1]
        self.assertEqual(call["cfg"], 3.0)
        self.assertEqual(call["temperature"], 0.7)
        self.assertEqual(call["solver"], "euler")
        self.assertEqual(call["num_steps"], 7)
        self.assertTrue(torch.equal(call["z"][:3], z))
        self.assertTrue(torch.equal(call["z"][3:], z_uncond))

    def test_single_stream_helper_uses_single_stream_masks_and_original_sigma(self):
        latent_dim = 4
        hidden_size = 8
        dummy = types.SimpleNamespace(
            config=types.SimpleNamespace(image_tokens_per_img=4, boi_token_id=11),
            model=FakeInnerModel(hidden_size=hidden_size),
            image_flow_head=FakeImageFlowHead(latent_dim=latent_dim),
        )

        input_ids = torch.tensor([[10, 11, 7, 7, 7, 7, 12, 13]])
        token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 2]])
        sigma = torch.tensor([[0, 1, 4, 5, 3, 6, 2, 7]])
        spans = [(0, 2, 6)]

        def fake_mask(sigma, seq_len, device, **kwargs):
            mask = sigma.detach().clone()
            if kwargs.get("image_uncond_rows") is not None:
                self.assertTrue(kwargs["image_uncond_rows"].all())
                self.assertTrue(torch.equal(kwargs["input_ids"], input_ids))
                self.assertTrue(torch.equal(kwargs["token_types"], token_types))
                self.assertEqual(kwargs["boi_token_id"], 11)
                mask = mask + 100
            return mask

        with patch("utils.utils.get_selfless_mask", side_effect=fake_mask):
            generated = Qwen3ForCausalLM.sample_image_latents_single_stream(
                dummy,
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
                spans=spans,
                image_latent_dim=latent_dim,
                flow_temperature=0.5,
                parallel_rate=1,
                order_strategy="sigma",
            )

        self.assertEqual(tuple(generated.shape), (1, latent_dim, 2, 2))
        self.assertEqual(len(dummy.model.calls), 4)
        self.assertEqual(len(dummy.image_flow_head.sample_calls), 4)

        for call in dummy.model.calls:
            self.assertFalse(call["calculate_likelihood"])
            self.assertTrue(torch.equal(call["sigma"], sigma))

        expected_masks = [
            [False, False, False, False],
            [False, False, True, False],
            [True, False, True, False],
            [True, True, True, False],
        ]
        for call, expected in zip(dummy.model.calls, expected_masks):
            self.assertEqual(call["image_latent_mask"][0, 2:6].tolist(), expected)

    def test_single_stream_treats_initial_image_latent_mask_as_already_filled(self):
        latent_dim = 4
        hidden_size = 8
        dummy = types.SimpleNamespace(
            config=types.SimpleNamespace(image_tokens_per_img=4, boi_token_id=11),
            model=FakeInnerModel(hidden_size=hidden_size),
            image_flow_head=FakeImageFlowHead(latent_dim=latent_dim),
        )

        input_ids = torch.tensor([[10, 11, 7, 7, 7, 7, 12, 13]])
        token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 2]])
        sigma = torch.tensor([[0, 1, 4, 5, 3, 6, 2, 7]])
        spans = [(0, 2, 6)]
        initial_latents = torch.zeros(1, 8, latent_dim)
        initial_latents[0, 2] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        initial_latents[0, 4] = torch.tensor([5.0, 6.0, 7.0, 8.0])
        initial_mask = torch.zeros(1, 8, dtype=torch.bool)
        initial_mask[0, [2, 4]] = True

        with patch("utils.utils.get_selfless_mask", side_effect=lambda sigma, seq_len, device, **kwargs: sigma.detach().clone()):
            generated = Qwen3ForCausalLM.sample_image_latents_single_stream(
                dummy,
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
                spans=spans,
                image_latent_dim=latent_dim,
                initial_image_latents=initial_latents,
                initial_image_latent_mask=initial_mask,
                flow_temperature=0.5,
                parallel_rate=1,
                order_strategy="sigma",
            )

        self.assertEqual(tuple(generated.shape), (1, latent_dim, 2, 2))
        self.assertEqual(len(dummy.model.calls), 2)
        self.assertEqual(len(dummy.image_flow_head.sample_calls), 2)
        self.assertTrue(torch.equal(generated[0, :, 0, 0], initial_latents[0, 2]))
        self.assertTrue(torch.equal(generated[0, :, 1, 0], initial_latents[0, 4]))

        first_call = dummy.model.calls[0]
        self.assertEqual(first_call["image_latent_mask"][0, 2:6].tolist(), [True, False, True, False])
        self.assertTrue(torch.equal(first_call["image_latents"][0, 2], initial_latents[0, 2]))
        self.assertTrue(torch.equal(first_call["image_latents"][0, 4], initial_latents[0, 4]))

    def test_single_stream_cfg_uses_cond_and_uncond_hidden_pairs(self):
        latent_dim = 4
        hidden_size = 8
        dummy = types.SimpleNamespace(
            config=types.SimpleNamespace(image_tokens_per_img=4, boi_token_id=11),
            model=FakeInnerModel(hidden_size=hidden_size),
            image_flow_head=FakeImageFlowHead(latent_dim=latent_dim),
        )

        input_ids = torch.tensor([[10, 11, 7, 7, 7, 7, 12, 13]])
        token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 2]])
        sigma = torch.tensor([[0, 1, 4, 5, 3, 6, 2, 7]])
        spans = [(0, 2, 6)]

        def fake_mask(sigma, seq_len, device, **kwargs):
            mask = sigma.detach().clone()
            if kwargs.get("image_uncond_rows") is not None:
                self.assertTrue(kwargs["image_uncond_rows"].all())
                self.assertTrue(torch.equal(kwargs["input_ids"], input_ids))
                self.assertTrue(torch.equal(kwargs["token_types"], token_types))
                self.assertEqual(kwargs["boi_token_id"], 11)
                mask = mask + 100
            return mask

        with patch("utils.utils.get_selfless_mask", side_effect=fake_mask):
            generated = Qwen3ForCausalLM.sample_image_latents_single_stream(
                dummy,
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
                spans=spans,
                image_latent_dim=latent_dim,
                flow_cfg=2.0,
                parallel_rate=1,
                order_strategy="sigma",
            )

        self.assertEqual(tuple(generated.shape), (1, latent_dim, 2, 2))
        self.assertEqual(len(dummy.model.calls), 8)
        self.assertEqual(len(dummy.image_flow_head.sample_calls), 4)
        for cond_call, uncond_call in zip(dummy.model.calls[0::2], dummy.model.calls[1::2]):
            self.assertFalse(cond_call["image_uncond"])
            self.assertTrue(uncond_call["image_uncond"])
            self.assertTrue(torch.equal(cond_call["sigma"], sigma))
            self.assertTrue(torch.equal(uncond_call["sigma"], sigma + 100))
        expected_cfg = [1.25, 1.5, 1.75, 2.0]
        for sample_call, cfg in zip(dummy.image_flow_head.sample_calls, expected_cfg):
            self.assertAlmostEqual(sample_call["cfg"], cfg)
            self.assertEqual(tuple(sample_call["z"].shape), (2, hidden_size))
            self.assertTrue(torch.all(sample_call["z"][0] == 1.0))
            self.assertTrue(torch.all(sample_call["z"][1] == 2.0))

    def test_text_generate_is_single_stream_and_has_no_discrete_image_branch(self):
        dummy = types.SimpleNamespace(
            device=torch.device("cpu"),
            config=types.SimpleNamespace(
                mask_token_id=7,
                eos_token_id=9,
                image_tokens_per_img=4,
                image_latent_dim=4,
            ),
            image_latent_dim=4,
            model=FakeTextBackbone(token_plan=[5, 9]),
            lm_head=FakeTextHead(vocab_size=32),
            unified_head=False,
        )
        dummy._sample_from_logits = types.MethodType(Qwen3ForCausalLM._sample_from_logits, dummy)

        with patch("utils.utils.get_selfless_mask", side_effect=lambda sigma, seq_len, device, **kwargs: sigma.detach().clone()):
            seq, token_types, image_latents, generated_images, image_spans = Qwen3ForCausalLM._generate_one(
                dummy,
                prompt_ids=torch.tensor([1, 2]),
                gen_length=4,
                prompt_task="ar",
                block_size=1,
                temperature=0.0,
                ratio=None,
                parallel_rate=None,
                decode_strategy="confidence",
            )

        self.assertEqual(seq.tolist(), [1, 2, 5, 9])
        self.assertEqual(token_types.tolist(), [0, 0, 0, 0])
        self.assertEqual(tuple(image_latents.shape), (4, 4))
        self.assertEqual(tuple(generated_images.shape), (0, 4, 2, 2))
        self.assertEqual(tuple(image_spans.shape), (0, 2))
        self.assertEqual(len(dummy.model.calls), 2)
        for call in dummy.model.calls:
            self.assertFalse(call["calculate_likelihood"])
            self.assertNotIn("image_latents", call)
            self.assertNotIn("image_latent_mask", call)

    def test_generate_one_expands_boi_into_image_span_and_continues_text(self):
        dummy = types.SimpleNamespace(
            device=torch.device("cpu"),
            config=types.SimpleNamespace(
                mask_token_id=7,
                image_mask_token_id=8,
                eos_token_id=9,
                boi_token_id=11,
                eoi_token_id=12,
                image_tokens_per_img=4,
                image_latent_dim=4,
            ),
            image_latent_dim=4,
            model=FakeTextBackbone(token_plan=[11, 5, 9]),
            lm_head=FakeTextHead(vocab_size=32),
        )
        dummy._sample_from_logits = types.MethodType(Qwen3ForCausalLM._sample_from_logits, dummy)

        sampler_calls = []

        def fake_sampler(self, **kwargs):
            sampler_calls.append(
                {
                    "input_ids": kwargs["input_ids"].detach().clone(),
                    "token_types": kwargs["token_types"].detach().clone(),
                    "sigma": kwargs["sigma"].detach().clone(),
                    "spans": list(kwargs["spans"]),
                    "initial_image_latent_mask": kwargs["initial_image_latent_mask"].detach().clone(),
                    "flow_cfg_schedule": kwargs["flow_cfg_schedule"],
                    "parallel_rate": kwargs["parallel_rate"],
                    "order_strategy": kwargs["order_strategy"],
                }
            )
            return torch.arange(16, dtype=torch.float32).view(1, 4, 2, 2)

        dummy.sample_image_latents_single_stream = types.MethodType(fake_sampler, dummy)

        with patch("utils.utils.get_selfless_mask", side_effect=lambda sigma, seq_len, device, **kwargs: sigma.detach().clone()):
            seq, token_types, image_latents, generated_images, image_spans = Qwen3ForCausalLM._generate_one(
                dummy,
                prompt_ids=torch.tensor([1, 2]),
                gen_length=3,
                prompt_task="ar",
                block_size=1,
                temperature=0.0,
                ratio=None,
                parallel_rate=None,
                decode_strategy="confidence",
                flow_cfg_schedule="constant",
                image_parallel_rate=2,
            )

        self.assertEqual(seq.tolist(), [1, 2, 11, 8, 8, 8, 8, 12, 5, 9])
        self.assertEqual(token_types.tolist(), [0, 0, 2, 1, 1, 1, 1, 2, 0, 0])
        self.assertEqual(image_spans.tolist(), [[3, 7]])
        self.assertEqual(tuple(generated_images.shape), (1, 4, 2, 2))
        expected_flat = torch.arange(16, dtype=torch.float32).view(4, 2, 2).permute(1, 2, 0).reshape(4, 4)
        self.assertTrue(torch.equal(image_latents[3:7], expected_flat))

        self.assertEqual(len(sampler_calls), 1)
        call = sampler_calls[0]
        self.assertEqual(call["spans"], [(0, 3, 7)])
        self.assertEqual(call["input_ids"].tolist(), [[1, 2, 11, 8, 8, 8, 8, 12]])
        self.assertEqual(call["token_types"].tolist(), [[0, 0, 2, 1, 1, 1, 1, 2]])
        self.assertEqual(call["sigma"].tolist(), [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        self.assertFalse(call["initial_image_latent_mask"].any())
        self.assertEqual(call["flow_cfg_schedule"], "constant")
        self.assertEqual(call["parallel_rate"], 2)
        self.assertEqual(call["order_strategy"], "confidence")

        self.assertEqual(len(dummy.model.calls), 3)
        after_image_call = dummy.model.calls[1]
        self.assertTrue(torch.equal(after_image_call["image_latent_mask"][0, 3:7], torch.ones(4, dtype=torch.bool)))
        self.assertTrue(torch.equal(after_image_call["image_latents"][0, 3:7], expected_flat))


if __name__ == "__main__":
    unittest.main()
