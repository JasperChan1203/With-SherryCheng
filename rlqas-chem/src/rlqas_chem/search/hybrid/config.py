"""Configuration helpers for Phase 3 hybrid search."""
from typing import Dict, Any


class HybridSearchConfig:
    """Default configuration provider for hybrid HEA+UCC search.

    Merges user-supplied config with sensible defaults for each sub-section.
    """

    DEFAULTS: Dict[str, Dict] = {
        "environment": {
            "max_depth": 15,
            "max_blocks": 6,
            "encoding_method": "matrix",
            "run_classical_opt": True,
            "complexity_penalty": 0.0,
            "operator_type": "fermion",
            "entanglement_patterns": ["linear", "circular"],
        },
        "controller": {
            "n_episodes": 500,
            "early_stop_threshold": 1.6e-3,
            "log_frequency": 10,
            "checkpoint_frequency": 0,
            "checkpoint_dir": "checkpoints",
            "seed": 42,
        },
        "fusion": {
            "fusion_mode": "sequential",
            "min_ucc_components": 1,
            "max_ucc_components": 5,
            "hea_layers_per_block": 2,
        },
    }

    # Keys that live in the search sub-section and map into the fusion section
    _SEARCH_TO_FUSION_KEYS = {"fusion_mode", "encoding_method"}

    def __init__(self, config: Dict = None):
        self._config = config or {}

    def get_section(self, section: str) -> Dict:
        """Return merged defaults + user values for the requested section.

        Also promotes matching top-level keys into the section dict so callers
        can specify e.g. ``max_depth`` at the top level of the config dict.
        """
        defaults = dict(self.DEFAULTS.get(section, {}))
        user_section = dict(self._config.get(section, {}))
        merged = {**defaults, **user_section}

        # Promote top-level keys that belong to this section
        for k in list(defaults.keys()):
            if k in self._config and k not in user_section:
                merged[k] = self._config[k]

        # Special case: search sub-section can override fusion keys
        if section == "fusion":
            search_cfg = self._config.get("search", {})
            for k in self._SEARCH_TO_FUSION_KEYS:
                if k in search_cfg:
                    merged[k] = search_cfg[k]

        return merged
