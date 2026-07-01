from __future__ import annotations

import os
from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin, WhisperForConditionalGeneration
from transformers.cache_utils import EncoderDecoderCache, StaticCache
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

from .configuration_mapperatorinator import MapperatorinatorConfig
from .spectrogram import MelSpectrogram

LABEL_IGNORE_ID = -100


def get_backbone_model(config: MapperatorinatorConfig):
    name = config.backbone_model_name
    b_config = config.backbone_config

    if name.startswith("google/t5"):
        from transformers import T5Config, T5ForConditionalGeneration
        if isinstance(b_config, dict):
            b_config = T5Config(**b_config)
        model_cls = T5ForConditionalGeneration
    elif name.startswith("OliBomby/nwhisper"):
        from .custom_transformers import NWhisperConfig, NWhisperForConditionalGeneration
        if isinstance(b_config, dict):
            b_config = NWhisperConfig(**b_config)
        model_cls = NWhisperForConditionalGeneration
    elif name.startswith("Tiger14n/ropewhisper"):
        from .custom_transformers import RoPEWhisperConfig, RoPEWhisperForConditionalGeneration
        if isinstance(b_config, dict):
            b_config = RoPEWhisperConfig(**b_config)
        model_cls = RoPEWhisperForConditionalGeneration
    elif name.startswith("openai/whisper"):
        from transformers import WhisperConfig, WhisperForConditionalGeneration
        if isinstance(b_config, dict):
            b_config = WhisperConfig(**b_config)
        model_cls = WhisperForConditionalGeneration
    elif name.startswith("UsefulSensors/moonshine-tiny"):
        from .custom_transformers import MoonshineConfig, MoonshineForConditionalGeneration
        if isinstance(b_config, dict):
            b_config = MoonshineConfig(**b_config)
        model_cls = MoonshineForConditionalGeneration
    elif name.startswith("OliBomby/varwhisper"):
        from .custom_transformers import VarWhisperConfig, VarWhisperForConditionalGeneration
        if isinstance(b_config, dict):
            b_config = VarWhisperConfig(**b_config)
        model_cls = VarWhisperForConditionalGeneration
    else:
        raise NotImplementedError

    b_config._attn_implementation = config._attn_implementation
    b_config.dtype = config.dtype
    model = model_cls(b_config)

    return model


class Mapperatorinator(PreTrainedModel, GenerationMixin):
    __slots__ = [
        "spectrogram",
        "decoder_embedder",
        "encoder_embedder",
        "transformer",
        "style_embedder",
        "num_classes",
        "_mapperatorinator_preallocated_sample_used",
    ]
    config_class = MapperatorinatorConfig
    base_model_prefix = "model"
    main_input_name = "frames"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, config: MapperatorinatorConfig):
        super().__init__(config)
        self._mapperatorinator_preallocated_sample_used = False

        if not config.input_raw_wave:
            self.spectrogram = MelSpectrogram(
                config.spectrogram_implementation,
                config.spectrogram_log_scale,
                config.sample_rate,
                config.n_fft,
                config.n_mels,
                config.hop_length,
                config.f_min,
                config.f_max,
                config.pad_mode,
            )

        self.transformer: WhisperForConditionalGeneration = get_backbone_model(config)

        self.num_classes = config.num_classes
        self.input_features = config.input_features
        self.input_raw_wave = config.input_raw_wave
        self.project_encoder_input = config.project_encoder_input
        self.embed_decoder_input = config.embed_decoder_input

        self.do_style_embed = config.do_style_embed
        self.do_difficulty_embed = config.do_difficulty_embed
        self.do_mapper_embed = config.do_mapper_embed
        self.do_song_position_embed = config.do_song_position_embed
        d_model = config.hidden_size

        if self.do_style_embed:
            self.style_embedder = LabelEmbedder(self.num_classes, d_model)
            nn.init.normal_(self.style_embedder.embedding_table.weight, std=config.init_std)

        if self.do_difficulty_embed:
            self.difficulty_embedder = DifficultyEmbedder(
                hidden_size=config.cond_dim,
                max_difficulty=10,
            )

        if self.do_mapper_embed:
            self.mapper_embedder = MapperStyleEmbedder(
                embedding_dim=config.cond_dim,
                num_mappers=config.num_mappers,
            )

        if self.do_song_position_embed:
            self.song_pos_embedder = SongPositionEmbedder(
                hidden_size=config.cond_dim,
                num_basis=10,
            )

        if self.project_encoder_input:
            self.encoder_embedder = nn.Linear(config.n_mels + config.cond_size, d_model)

        if self.embed_decoder_input:
            self.decoder_embedder = nn.Embedding(config.vocab_size_in, d_model)
            self.decoder_embedder.weight.data.normal_(mean=0.0, std=config.init_std)

        class_weights = torch.ones(config.vocab_size)
        class_weights[config.rhythm_token_start:config.rhythm_token_end] = config.rhythm_weight
        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none",
            ignore_index=LABEL_IGNORE_ID,
            label_smoothing=config.label_smoothing
        )

    def forward(
            self,
            frames: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            beatmap_idx: Optional[torch.Tensor] = None,
            difficulty: Optional[torch.Tensor] = None,
            mapper_idx: Optional[torch.Tensor] = None,
            song_position: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            sample_weights: Optional[torch.FloatTensor] = None,
            **kwargs
    ) -> Seq2SeqLMOutput:
        """
        frames: B x L_encoder x mel_bins, float32
        decoder_input_ids: B x L_decoder, int64
        beatmap_idx: B, int64
        beatmap_id: B, int64
        encoder_outputs: B x L_encoder x D, float32
        """
        if beatmap_idx is None and self.do_style_embed:
            batch_size = frames.shape[0] if frames is not None else decoder_input_ids.shape[0]
            device = frames.device if frames is not None else decoder_input_ids.device
            beatmap_idx = torch.full([batch_size], self.num_classes, dtype=torch.long, device=device)

        inputs = dict(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs, **kwargs
        )

        inputs_embeds = None
        if encoder_outputs is None and frames is not None:
            if not self.input_raw_wave:
                frames = self.spectrogram(frames)  # (N, L, M)
                frames = frames.to(dtype=self.transformer.dtype)  # Ensure correct dtype for the model
                conds = []

                if self.do_style_embed:
                    style_embedding = self.style_embedder(beatmap_idx)  # (N, D)
                    style_embedding = style_embedding.unsqueeze(1).repeat((1, frames.shape[1], 1))
                    conds.append(style_embedding)
                if self.do_difficulty_embed:
                    difficulty_embedding = self.difficulty_embedder(difficulty)
                    conds.append(difficulty_embedding)
                if self.do_mapper_embed:
                    mapper_embedding = self.mapper_embedder(mapper_idx)
                    conds.append(mapper_embedding)
                if self.do_song_position_embed:
                    song_position_embedding = self.song_pos_embedder(song_position)
                    conds.append(song_position_embedding)

                conds_expanded = [c.unsqueeze(1).expand((-1, frames.shape[1], -1)) for c in conds]
                inputs_embeds = torch.concatenate([frames] + conds_expanded, dim=-1)

            if self.project_encoder_input:
                inputs_embeds = self.encoder_embedder(inputs_embeds) if inputs_embeds is not None else None

            if self.input_raw_wave:
                inputs["input_values"] = frames.reshape(frames.shape[0], -1)
            elif self.input_features:
                inputs["input_features"] = torch.swapaxes(inputs_embeds, 1, 2) if inputs_embeds is not None else None
            else:
                inputs["inputs_embeds"] = inputs_embeds

        if self.embed_decoder_input:
            inputs["decoder_inputs_embeds"] = self.decoder_embedder(decoder_input_ids)
            del inputs["decoder_input_ids"]

        output = self.transformer.forward(**inputs)

        loss = None
        if labels is not None:
            unreduced_loss = self.loss_fn(torch.swapaxes(output.logits, 1, -1), labels)
            if sample_weights is not None:
                unreduced_loss *= sample_weights.unsqueeze(1)
            loss = unreduced_loss.sum() / (labels != LABEL_IGNORE_ID).sum()

        return Seq2SeqLMOutput(
            loss=loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            decoder_hidden_states=output.decoder_hidden_states,
            decoder_attentions=output.decoder_attentions,
            cross_attentions=output.cross_attentions,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=output.encoder_hidden_states,
            encoder_attentions=output.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_attention_mask=None,
        cache_position=None,
        negative_prompt=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        # Add negative prompt to the input for classifier free guidance
        if negative_prompt is not None:
            decoder_input_ids = decoder_input_ids.repeat((2, 1))
            decoder_input_ids[:decoder_input_ids.shape[0] // 2, :negative_prompt.shape[1]] = negative_prompt

            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.repeat((2, 1))
                if negative_prompt_attention_mask is not None:
                    decoder_attention_mask[:decoder_attention_mask.shape[0] // 2, :negative_prompt_attention_mask.shape[1]] = negative_prompt_attention_mask

            if encoder_outputs is not None:
                encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state.repeat((2, 1, 1)))

        model_inputs = self.transformer.prepare_inputs_for_generation(
            input_ids=decoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            cache_position=cache_position,
            **kwargs,
        )

        # 7. Forward ALL kwargs that are uninitialized (e.g. `beatmap_idx`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        return model_inputs

    def _sample(
            self,
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            synced_gpus: bool = False,
            streamer=None,
            **model_kwargs,
    ):
        self._mapperatorinator_preallocated_sample_used = False
        if not self._can_use_preallocated_sample(
                input_ids,
                logits_processor,
                generation_config,
                synced_gpus,
                streamer,
                model_kwargs,
        ):
            return GenerationMixin._sample(
                self,
                input_ids,
                logits_processor,
                stopping_criteria,
                generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        return self._preallocated_sample(
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            **model_kwargs,
        )

    def _can_use_preallocated_sample(
            self,
            input_ids,
            logits_processor,
            generation_config,
            synced_gpus,
            streamer,
            model_kwargs,
    ) -> bool:
        if not bool(getattr(generation_config, "mapperatorinator_preallocated_sample", False)):
            return False

        past_key_values = model_kwargs.get("past_key_values")
        uses_static_encoder_decoder_cache = (
                isinstance(past_key_values, EncoderDecoderCache)
                and isinstance(past_key_values.self_attention_cache, StaticCache)
                and isinstance(past_key_values.cross_attention_cache, StaticCache)
        )
        has_cfg_processor = any(processor.__class__.__name__ == "ClassifierFreeGuidanceLogitsProcessor"
                                for processor in logits_processor)
        pad_token_id = generation_config._pad_token_tensor

        return (
                self.config.is_encoder_decoder
                and input_ids.shape[0] == 1
                and not synced_gpus
                and streamer is None
                and generation_config.do_sample
                and generation_config.num_beams == 1
                and not generation_config.return_dict_in_generate
                and not generation_config.output_attentions
                and not generation_config.output_hidden_states
                and not generation_config.output_scores
                and not generation_config.output_logits
                and generation_config.prefill_chunk_size is None
                and generation_config.max_length is not None
                and pad_token_id is not None
                and pad_token_id.numel() == 1
                and model_kwargs.get("decoder_attention_mask") is not None
                and uses_static_encoder_decoder_cache
                and model_kwargs.get("use_cache", True)
                and model_kwargs.get("negative_prompt") is None
                and model_kwargs.get("negative_prompt_attention_mask") is None
                and not has_cfg_processor
                and (generation_config.guidance_scale is None or generation_config.guidance_scale == 1)
                and "token_type_ids" not in model_kwargs
        )

    def _preallocated_sample(
            self,
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            **model_kwargs,
    ):
        pad_token_id = generation_config._pad_token_tensor
        pad_token_value = int(pad_token_id.item()) if pad_token_id is not None else 0
        batch_size, cur_len = input_ids.shape[:2]
        max_length = int(generation_config.max_length)

        if cur_len >= max_length:
            return input_ids

        self._mapperatorinator_preallocated_sample_used = True
        input_ids_full = input_ids.new_full((batch_size, max_length), pad_token_value)
        input_ids_full[:, :cur_len] = input_ids

        decoder_attention_mask = model_kwargs.pop("decoder_attention_mask")
        decoder_attention_mask_full = decoder_attention_mask.new_zeros((batch_size, max_length))
        decoder_attention_mask_full[:, :cur_len] = decoder_attention_mask

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        if self._valid_auto_compile_criteria(model_kwargs, generation_config):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

        this_peer_finished = False
        while not this_peer_finished:
            current_input_ids = input_ids_full[:, :cur_len]
            model_kwargs["decoder_attention_mask"] = decoder_attention_mask_full[:, :cur_len]
            model_inputs = self.prepare_inputs_for_generation(current_input_ids, **model_kwargs)

            if cur_len == input_ids.shape[1]:
                outputs = self(**model_inputs, return_dict=True)
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            if getattr(outputs, "past_key_values", None) is not None:
                model_kwargs["past_key_values"] = outputs.past_key_values
            if model_kwargs.get("use_cache", True):
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
            else:
                return GenerationMixin._sample(
                    self,
                    current_input_ids,
                    logits_processor,
                    stopping_criteria,
                    generation_config,
                    **model_kwargs,
                )

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(current_input_ids, next_token_logits)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids_full[:, cur_len] = next_tokens
            decoder_attention_mask_full[:, cur_len] = 1
            cur_len += 1

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids_full[:, :cur_len], None)
            this_peer_finished = bool(unfinished_sequences.max() == 0 or cur_len >= max_length)
            del outputs

        return input_ids_full[:, :cur_len]

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: torch.Tensor,
        device: torch.device = None,
    ):
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        if device is None:
            device = self.device
        if decoder_start_token_id.ndim == 1:
            if decoder_start_token_id.shape[0] != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expected to have length {batch_size} but got {decoder_start_token_id.shape[0]}"
                )
            decoder_start_token_id = decoder_start_token_id.view(-1, 1)
        else:
            decoder_start_token_id = (
                torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
            )

        # Mapperatorinator handles the task-specific decoder input externally
        if decoder_input_ids is None:
            decoder_input_ids = decoder_start_token_id

        return decoder_input_ids, model_kwargs

    def can_generate(self) -> bool:
        return True

    def tie_weights(self):
        self.transformer.tie_weights()

    def get_input_embeddings(self):
        if self.embed_decoder_input:
            return self.decoder_embedder
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value):
        if self.embed_decoder_input:
            self.decoder_embedder = value
            return
        self.transformer.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.transformer.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.transformer.set_output_embeddings(new_embeddings)

    def get_encoder(self):
        return OsuTEncoder(
            self.transformer.get_encoder(),
            self.spectrogram if not self.input_raw_wave else None,
            self.style_embedder if self.do_style_embed else None,
            self.difficulty_embedder if self.do_difficulty_embed else None,
            self.mapper_embedder if self.do_mapper_embed else None,
            self.song_pos_embedder if self.do_song_position_embed else None,
            self.encoder_embedder if self.project_encoder_input else None,
            self.num_classes,
            self.input_features,
            self.input_raw_wave,
            self.project_encoder_input,
            self.do_style_embed,
            self.do_difficulty_embed,
            self.do_mapper_embed,
            self.do_song_position_embed,
        )

    def get_decoder(self):
        return self.transformer.get_decoder()


class OsuTEncoder(nn.Module):
    def __init__(
            self,
            base_encoder: nn.Module,
            spectrogram: MelSpectrogram,
            style_embedder: LabelEmbedder,
            difficulty_embedder: DifficultyEmbedder,
            mapper_embedder: MapperStyleEmbedder,
            song_pos_embedder: SongPositionEmbedder,
            encoder_embedder: nn.Linear,
            num_classes: int,
            input_features: bool,
            input_raw_wave: bool,
            project_encoder_input: bool,
            do_style_embed: bool,
            do_difficulty_embed: bool,
            do_mapper_embed: bool,
            do_song_position_embed: bool,
    ):
        super().__init__()
        self.base = base_encoder
        self.spectrogram = spectrogram
        self.style_embedder = style_embedder
        self.difficulty_embedder = difficulty_embedder
        self.mapper_embedder = mapper_embedder
        self.song_pos_embedder = song_pos_embedder
        self.encoder_embedder = encoder_embedder
        self.num_classes = num_classes
        self.input_features = input_features
        self.input_raw_wave = input_raw_wave
        self.project_encoder_input = project_encoder_input
        self.do_style_embed = do_style_embed
        self.do_difficulty_embed = do_difficulty_embed
        self.do_mapper_embed = do_mapper_embed
        self.do_song_position_embed = do_song_position_embed

    def forward(
            self,
            frames: torch.FloatTensor,
            beatmap_idx: Optional[torch.Tensor] = None,
            difficulty: Optional[torch.Tensor] = None,
            mapper_idx: Optional[torch.Tensor] = None,
            song_position: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = False
    ):
        if beatmap_idx is None and self.do_style_embed:
            batch_size = frames.shape[0]
            device = frames.device
            beatmap_idx = torch.full([batch_size], self.num_classes, dtype=torch.long, device=device)

        if self.input_raw_wave:
            inputs_embeds = frames.reshape(frames.shape[0], -1)
        else:
            frames = self.spectrogram(frames)  # (N, L, M)
            frames = frames.to(dtype=self.base.dtype)  # Ensure correct dtype for the model
            conds = []

            if self.do_style_embed:
                style_embedding = self.style_embedder(beatmap_idx)  # (N, D)
                style_embedding = style_embedding.unsqueeze(1).repeat((1, frames.shape[1], 1))
                conds.append(style_embedding)
            if self.do_difficulty_embed:
                difficulty_embedding = self.difficulty_embedder(difficulty)
                conds.append(difficulty_embedding)
            if self.do_mapper_embed:
                mapper_embedding = self.mapper_embedder(mapper_idx)
                conds.append(mapper_embedding)
            if self.do_song_position_embed:
                song_position_embedding = self.song_pos_embedder(song_position)
                conds.append(song_position_embedding)

            conds_expanded = [c.unsqueeze(1).expand((-1, frames.shape[1], -1)) for c in conds]
            inputs_embeds = torch.concatenate([frames] + conds_expanded, dim=-1)

            if self.project_encoder_input:
                inputs_embeds = self.encoder_embedder(inputs_embeds) if inputs_embeds is not None else None

            if self.input_features:
                inputs_embeds = torch.swapaxes(inputs_embeds, 1, 2) if inputs_embeds is not None else None

        return self.base.forward(
            inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(
            num_classes + 1,
            hidden_size,
        )

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class DifficultyEmbedder(nn.Module):
    def __init__(self, hidden_size=64, max_difficulty=10.0, num_basis=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_difficulty = max_difficulty
        self.num_basis = num_basis

        # Learnable basis centers
        self.register_parameter(
            'basis_centers',
            nn.Parameter(torch.linspace(0, 1, num_basis))
        )

        # Learnable basis widths
        self.register_parameter(
            'basis_widths',
            nn.Parameter(torch.ones(num_basis) * 0.1)
        )

        self.difficulty_proj = nn.Sequential(
            nn.Linear(num_basis, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Initialize with smaller weights
        for m in self.difficulty_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Reduce gain
                nn.init.zeros_(m.bias)

    def compute_basis_functions(self, diff_normalized):
        # Compute RBF basis functions
        diff_expanded = diff_normalized.unsqueeze(-1)  # [B, 1]
        centers = self.basis_centers.view(1, -1)       # [1, N]
        widths = self.basis_widths.view(1, -1)        # [1, N]

        # Gaussian RBF
        basis = torch.exp(
            -(diff_expanded - centers).pow(2) / (2 * widths.pow(2))
        )
        return basis

    def forward(self, difficulty):
        # Normalize difficulty
        diff_normalized = difficulty / self.max_difficulty

        # Compute basis functions
        basis = self.compute_basis_functions(diff_normalized)

        # Project to embedding space
        return self.difficulty_proj(basis)


class MapperStyleEmbedder(nn.Module):
    """
    Embedding layer for mapper styles
    """
    def __init__(self, num_mappers: int, embedding_dim: int = 64, dropout_prob: float = 0.1):
        """
        Args:
            num_mappers: Total number of unique mappers.
            embedding_dim: Size of the embedding vector.
            dropout_prob: Dropout probability for regularization.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_mappers = num_mappers

        # Embedding table: num_mappers rows for actual mappers + 1 row for default style (-1)
        self.embedding = nn.Embedding(num_embeddings=num_mappers + 1, embedding_dim=embedding_dim)

        # Initialize embeddings with small random values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Dropout for regularization to help small mappers generalize
        self.dropout = nn.Dropout(p=dropout_prob)

        # Layer normalization to stabilize embeddings (especially for mappers with few maps)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, mapper_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            mapper_ids: Tensor of shape [B] with mapper IDs (long/int), where:
                - IDs >= 0 correspond to specific mappers (0 to num_mappers-1).
                - ID = -1 triggers the default style.

        Returns:
            Embedding tensor of shape [B, embedding_dim] if mapper_ids is provided,
        """
        if mapper_ids is None:
            return None  # No conditioning applied

        # Map -1 to the last index (default style) and ensure IDs are valid
        mapper_ids = torch.where(
            mapper_ids == -1,
            torch.tensor(self.num_mappers, device=mapper_ids.device),
            mapper_ids
        )

        # Ensure mapper_ids are within bounds (0 to num_mappers)
        mapper_ids = torch.clamp(mapper_ids, min=0, max=self.num_mappers)

        # Retrieve embeddings: [B] -> [B, embedding_dim]
        embeddings = self.embedding(mapper_ids)

        # Apply dropout and normalization
        embeddings = self.dropout(embeddings)
        embeddings = self.layer_norm(embeddings)

        return embeddings


class SongPositionEmbedder(nn.Module):
    """
    Generates an embedding vector representing the global position and duration
    context of an audio chunk within a larger song.

    It takes a normalized start and end position of an audio chunk
    (e.g., [0.25, 0.30] meaning the chunk covers 25% to 30% of the total song)

    This allows the model to be aware of:
    - Where the current audio chunk begins within the song.
    - Where the current audio chunk ends within the song.
    - Implicitly, the duration or extent of the chunk relative to the song.
    This information can help the model make decisions appropriate for different
    song sections (e.g., intro, verse, chorus, outro) and varying chunk lengths.
    """
    def __init__(self, hidden_size=64, num_basis=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_basis = num_basis

        # Learnable basis centers from 0 to 1
        self.register_parameter(
            'basis_centers',
            nn.Parameter(torch.linspace(0, 1, num_basis))
        )

        # Learnable basis widths
        self.register_parameter(
            'basis_widths',
            nn.Parameter(torch.ones(num_basis) * 0.1)
        )

        self.position_proj = nn.Sequential(
            nn.Linear(num_basis * 2, hidden_size * 2),  # start and end positions
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),  # Reduce back to original hidden size
            nn.LayerNorm(hidden_size),
        )


        # Initialize weights
        for m in self.position_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def compute_basis_functions(self, position):
        # Compute RBF basis functions
        position_expanded = position.unsqueeze(-1)  # [B, 1]
        centers = self.basis_centers.view(1, -1)    # [1, N]
        widths = self.basis_widths.view(1, -1)      # [1, N]

        # Gaussian RBF
        basis = torch.exp(
            -(position_expanded - centers).pow(2) / (2 * widths.pow(2))
        )
        return basis

    def forward(self, position_range):
        """
        Args:
            position_range: Tensor of shape [B, 2] containing normalized start and end positions
                            position_range[:, 0] is the start position (0 to 1)
                            position_range[:, 1] is the end position (0 to 1)
        """
        # Split start and end positions
        start_pos = position_range[:, 0]
        end_pos = position_range[:, 1]

        # Compute basis functions for both positions
        start_basis = self.compute_basis_functions(start_pos)  # [B, num_basis]
        end_basis = self.compute_basis_functions(end_pos)     # [B, num_basis]

        # Concatenate bases
        combined_basis = torch.cat([start_basis, end_basis], dim=1)  # [B, num_basis*2]

        # Project to embedding space
        return self.position_proj(combined_basis)
