from __future__ import annotations
from copy import deepcopy
from typing import Iterable
import torch
from transformers import DynamicCache, Cache
from .configuration_tsrt import TSRTConfig


class TSRTDecoderCache(DynamicCache):

    def __init__(
        self,
        ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
        config: TSRTConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        cache_config = None

        if config is not None:
            cache_config = deepcopy(config)

            cache_config.num_hidden_layers = (
                config.num_decoder_layers
                + config.num_tsrt_layers
            )

            cache_config.layer_types = (
                config.layer_types[
                    config.num_encoder_layers:
                ]
                if config.layer_types is not None
                else None
            )

            self.cache_config = cache_config

        super().__init__(
            ddp_cache_data=ddp_cache_data,
            config=cache_config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )


class TSRTDocumentCache(DynamicCache):
    def __init__(
        self,
        ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
        config: TSRTConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        cache_config = None

        if config is not None:
            cache_config = deepcopy(config)
            cache_config.num_hidden_layers = config.num_tsrt_layers

            if config.layer_types is not None:
                cache_config.layer_types = config.layer_types[
                    : config.num_tsrt_layers
                ]
            else:
                cache_config.layer_types = None

        self.cache_config = cache_config

        super().__init__(
            ddp_cache_data=ddp_cache_data,
            config=cache_config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

        self.encoder_state: torch.Tensor | None = None

    # =========================
    # KV CACHE UTILS
    # =========================

    def has_kv(self, layer_idx: int) -> bool:
        layer = self.layers[layer_idx]
        return layer.is_initialized and layer.get_seq_length() > 0

    def get_kv(self, layer_idx: int):
        layer = self.layers[layer_idx]

        if not self.has_kv(layer_idx):
            return None, None

        return layer.keys, layer.values

    def reset_kv(self, layer_idx: int):
        layer = self.layers[layer_idx]
        layer.keys = None
        layer.values = None
        layer.is_initialized = False

    def reset_all(self):
        for layer in self.layers:
            layer.keys = None
            layer.values = None
            layer.is_initialized = False

    # =========================
    # ENCODER STATE
    # =========================

    def has_encoder_state(self) -> bool:
        return self.encoder_state is not None

    def update_encoder_state(self, new_state: torch.Tensor):
        """
        shape: (B, D, L, H)
        concat theo D
        """

        if self.encoder_state is None:
            self.encoder_state = new_state
            return

        # ===== FIX dtype =====
        if self.encoder_state.dtype != new_state.dtype:
            new_state = new_state.to(self.encoder_state.dtype)

        # ===== FIX device =====
        if self.encoder_state.device != new_state.device:
            new_state = new_state.to(self.encoder_state.device)

        self.encoder_state = torch.cat(
            [self.encoder_state, new_state],
            dim=1,
        )

        return self.encoder_state

    def reset_encoder_state(self):
        self.encoder_state = None

    def get_encoder_state(self):
        return self.encoder_state

class TSRTEmbeddingCache:
    def __init__(
        self,
        weights: torch.Tensor | None = None,
        hidden_embs: torch.Tensor | None = None,
        doc_embs: torch.Tensor | None = None,
    ):
        self.weights = weights
        self.hidden_embs = hidden_embs
        self.doc_embs = doc_embs

        # infer dtype/device từ tensor đầu tiên có sẵn
        ref = self._get_ref_tensor()
        self.dtype = ref.dtype if ref is not None else None
        self.device = ref.device if ref is not None else None

    # ========================
    # internal utils
    # ========================
    def _get_ref_tensor(self) -> torch.Tensor | None:
        for t in (self.weights, self.hidden_embs, self.doc_embs):
            if t is not None:
                return t
        return None

    def _ensure_compat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure dtype + device consistency
        """
        if self.dtype is None:
            self.dtype = x.dtype
        if self.device is None:
            self.device = x.device

        return x.to(device=self.device, dtype=self.dtype)

    def _concat(self, old: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
        """
        Concat theo dim=1 (giống KV cache: batch, seq, dim)
        """
        new = self._ensure_compat(new)

        if old is None:
            return new

        return torch.cat([old, new], dim=1)

    # ========================
    # update functions
    # ========================
    def update(self, weights: torch.Tensor, hidden_embs: torch.Tensor):
        """
        Concat weights và hidden_embs (giống KV cache growth theo seq)
        """
        self.weights = self._concat(self.weights, weights)
        self.hidden_embs = self._concat(self.hidden_embs, hidden_embs)

        return self.weights, self.hidden_embs

    def update_doc_embs(self, doc_embs: torch.Tensor):
        """
        Set / overwrite doc_embs (không concat)
        """
        doc_embs = self._ensure_compat(doc_embs)
        self.doc_embs = doc_embs

        return self.doc_embs

    # ========================
    # state check
    # ========================
    def has_doc_embs(self) -> bool:
        return self.doc_embs is not None and self.doc_embs.numel() > 0

    # ========================
    # reset
    # ========================
    def reset_doc_embs(self):
        """
        Reset doc_embs về trạng thái rỗng
        """
        self.doc_embs = None

    def reset_all(self):
        """
        Optional: reset toàn bộ cache
        """
        self.weights = None
        self.hidden_embs = None
        self.doc_embs = None
        self.dtype = None
        self.device = None
    
    def get_doc_embs(self) -> torch.Tensor | None:
        """
        Return cached document embeddings
        """
        return self.doc_embs


class TSRTCache(Cache):
    def __init__(
        self,
        ddp_cache_data=None,
        config=None,
        offloading=False,
        offload_only_non_sliding=False,
    ):
        self.decoder_cache = TSRTDecoderCache(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

        self.document_cache = TSRTDocumentCache(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

        self.embedding_cache = TSRTEmbeddingCache()

    # =========================
    # CORE (HF sẽ gọi)
    # =========================

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.decoder_cache.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        self.decoder_cache.reorder_cache(beam_idx)
        self.document_cache.reorder_cache(beam_idx)

        if self.embedding_cache.weights is not None:
            self.embedding_cache.weights = self.embedding_cache.weights.index_select(0, beam_idx)

        if self.embedding_cache.hidden_embs is not None:
            self.embedding_cache.hidden_embs = self.embedding_cache.hidden_embs.index_select(0, beam_idx)

        if self.embedding_cache.doc_embs is not None:
            self.embedding_cache.doc_embs = self.embedding_cache.doc_embs.index_select(0, beam_idx)

        return self

    def to_legacy_cache(self):
        return self

    # =========================
    # UTIL
    # =========================

    def reset(self):
        self.decoder_cache.reset()
        self.document_cache.reset_all()
        self.embedding_cache.reset_all()

    def get_decoder_cache(self):
        return self.decoder_cache

    def get_document_cache(self):
        return self.document_cache

    def get_embedding_cache(self):
        return self.embedding_cache

    def has_cache(self) -> bool:
        return any(
            layer.is_initialized for layer in self.decoder_cache.layers
        )