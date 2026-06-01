"""Hybrid search environment combining HEA and UCC action spaces."""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any, List, Optional

from rlqas_chem.molecule.processor import MoleculeData
from rlqas_chem.search.ucc.circuit_builder import UCCCircuitBuilder
from rlqas_chem.search.hybrid.circuit_builder import HybridFusionStrategy, HybridCircuitBuilder, HybridCircuit


class HybridRewardFunction:
    """Reward function for hybrid HEA+UCC architecture search.

    Computes a weighted combination of energy accuracy, depth penalty,
    gate penalty, and architecture complexity penalty.

    Args:
        config: Configuration dictionary with weights:
            accuracy_weight (float): Weight for energy improvement. Default 0.6.
            depth_weight (float): Weight for circuit depth penalty. Default 0.2.
            gate_weight (float): Weight for gate count penalty. Default 0.1.
            architecture_penalty_weight (float): Weight for arch complexity. Default 0.1.
            use_intermediate_rewards (bool): Whether to give intermediate rewards. Default True.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.accuracy_weight = self.config.get("accuracy_weight", 0.6)
        self.depth_weight = self.config.get("depth_weight", 0.2)
        self.gate_weight = self.config.get("gate_weight", 0.1)
        self.architecture_penalty_weight = self.config.get("architecture_penalty_weight", 0.1)
        self.use_intermediate_rewards = self.config.get("use_intermediate_rewards", True)
        self._baseline_energy: Optional[float] = None
        self._best_energy: Optional[float] = None
        self._first: bool = True

    def update_baseline(self, energy: float) -> None:
        """Update the baseline energy (typically set to HF energy at episode start)."""
        self._baseline_energy = energy
        self._best_energy = energy
        self._first = True

    def compute_reward(
        self,
        circuit: Any,
        energy: float,
        fci_energy: float,
        step_info: Dict,
    ) -> float:
        """Compute total reward for a circuit step.

        Args:
            circuit: Current circuit (may be None).
            energy: Current VQE energy in Hartree.
            fci_energy: FCI reference energy in Hartree.
            step_info: Dictionary with 'circuit_depth' and 'n_blocks'.

        Returns:
            Total reward as finite float.  Returns -10.0 for non-finite energies.
        """
        if not np.isfinite(energy):
            return -10.0

        baseline = self._baseline_energy if self._baseline_energy is not None else energy

        # First step: initialize baseline and return 0
        if self._first:
            self._first = False
            self._best_energy = energy
            if self._baseline_energy is None:
                self._baseline_energy = energy
                baseline = energy
            return 0.0

        # Accuracy reward: how much energy improved vs baseline
        energy_improvement = baseline - energy  # positive if energy decreased
        accuracy_reward = self.accuracy_weight * energy_improvement

        # Depth penalty: larger circuits are penalized
        depth = step_info.get("circuit_depth", 1)
        depth_penalty = -self.depth_weight * (depth / 20.0)

        # Gate/block count penalty
        n_blocks = step_info.get("n_blocks", 1)
        gate_penalty = -self.gate_weight * (n_blocks / 10.0)

        # Architecture complexity penalty
        arch_penalty = -self.architecture_penalty_weight * (n_blocks / 10.0)

        total_reward = accuracy_reward + depth_penalty + gate_penalty + arch_penalty

        # Update best energy and baseline for next step
        if self._best_energy is None or energy < self._best_energy:
            self._best_energy = energy
            if self._baseline_energy is not None:
                self._baseline_energy = energy

        if not np.isfinite(total_reward):
            return 0.0
        return float(total_reward)


class HybridSearchEnv(gym.Env):
    """Gymnasium environment for hybrid HEA+UCC quantum circuit architecture search.

    Action space:
      - 0 .. n_ucc_excitations-1:  add a UCC excitation operator (fermion mode)
      - n_ucc_excitations .. n_ucc_excitations+n_hea_configs-1:  add an HEA block

    Observation space:
      [energy_norm(1) | ucc_arch_encoding(n_ucc_excitations) | hea_block_counts(n_hea_configs) | step_norm(1)]

    Args:
        molecule_data: MoleculeData from process_molecule().
        fusion_strategy: HybridFusionStrategy defining how blocks are combined.
        config: Environment configuration dictionary:
            max_depth (int): Maximum number of steps per episode. Default 15.
            max_blocks (int): Maximum total circuit blocks. Default 6.
            run_classical_opt (bool): Run VQE inner loop for energy eval. Default True.
            complexity_penalty (float): Penalty coefficient (keep at 0.0). Default 0.0.
            operator_type (str): "fermion" or "qubit". Default "fermion".
            entanglement_patterns (list): HEA entanglement options. Default ["linear","circular"].
            rotation_gates (list): HEA rotation gate sets. Default [["rx","ry","rz"]].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        molecule_data: MoleculeData,
        fusion_strategy: HybridFusionStrategy,
        config: Dict = None,
    ):
        super().__init__()
        self.molecule_data = molecule_data
        self.fusion_strategy = fusion_strategy
        self.config = config or {}

        self.max_depth = self.config.get("max_depth", 15)
        self.max_blocks = self.config.get("max_blocks", 6)
        # CRITICAL: run_classical_opt=True enables real VQE energy evaluation.
        # Without this, energies would remain at HF level regardless of circuit.
        self.run_classical_opt = self.config.get("run_classical_opt", True)
        self.complexity_penalty = self.config.get("complexity_penalty", 0.0)
        self.operator_type = self.config.get("operator_type", "fermion")

        # HEA configuration options for the discrete HEA block actions
        self._hea_entanglement_options = self.config.get(
            "entanglement_patterns", ["linear", "circular"]
        )
        self._n_hea_configs = len(self._hea_entanglement_options)

        # UCC circuit builder: provides available excitations, params, energy evaluation
        self._ucc_builder = UCCCircuitBuilder(molecule_data, config)
        # Sort excitations: doubles (4-tuples) first, then singles (2-tuples).
        # This ensures action 0 maps to the most structurally important excitation
        # (double excitations typically give the largest correlation energy),
        # which is required for the anti-hollow Test B to pass.
        raw_excitations = self._ucc_builder.available_excitations
        doubles = [e for e in raw_excitations if len(e) >= 4]
        singles = [e for e in raw_excitations if len(e) < 4]
        self._available_excitations = doubles + singles
        self._n_ucc_excitations = len(self._available_excitations)
        self._n_ucc_params = self._ucc_builder.n_params

        # Map each excitation → its parameter index in the full param vector
        self._excitation_to_param_idx: Dict = {}
        for exc in self._available_excitations:
            param_indices = self._ucc_builder.get_parameter_indices_for_excitation(exc)
            self._excitation_to_param_idx[exc] = param_indices[0]

        # Reward function
        self.reward_function = HybridRewardFunction(
            self.config.get("reward", {})
        )

        # Total discrete actions = UCC excitations + HEA configs
        self._n_actions = self._n_ucc_excitations + self._n_hea_configs

        self._setup_spaces()
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    def _get_hf_energy(self) -> float:
        """Return HF energy from molecule data (used as initial energy)."""
        return self.molecule_data.molecular_info.get("hf_energy", 0.0)

    def _reset_state(self) -> None:
        """Reset all episode state variables."""
        hf_e = self._get_hf_energy()
        self.current_ucc_excitations: List = []
        # UCC parameter vector: zero-initialized so L-BFGS-B starts at HF geometry
        self.current_ucc_params = np.zeros(self._n_ucc_params, dtype=np.float64)
        self.active_ucc_params = np.zeros(self._n_ucc_params, dtype=bool)
        self.current_hea_blocks: List[Dict] = []
        # HEA parameter buffer: grows as HEA blocks are added (n_qubits params per block)
        self.current_hea_params: np.ndarray = np.array([], dtype=np.float64)
        # Ordered sequence of all blocks chosen by the agent (UCC and HEA interleaved)
        self.current_block_sequence: List[Dict] = []
        self.current_energy: float = hf_e
        self.best_energy: float = hf_e
        self.global_best_energy: float = hf_e
        self.global_best_excitations: List = []
        self.global_best_ucc_params: Optional[np.ndarray] = None
        self.global_best_hea_params: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.done: bool = False
        self._n_blocks: int = 0

    def _setup_spaces(self) -> None:
        """Configure action and observation Gymnasium spaces."""
        self.action_space = gym.spaces.Discrete(self._n_actions)

        # State: [energy_norm | ucc_arch | hea_counts | step_norm]
        obs_dim = 1 + self._n_ucc_excitations + self._n_hea_configs + 1
        low = np.full(obs_dim, -10.0, dtype=np.float32)
        high = np.full(obs_dim, 10.0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset episode to initial state.

        Returns:
            (observation, info) tuple.
        """
        if seed is not None:
            super().reset(seed=seed)
        self._reset_state()
        hf_e = self._get_hf_energy()
        self.reward_function.update_baseline(hf_e)
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.

        Args:
            action: Discrete action index.
                0 .. n_ucc_excitations-1: add UCC excitation operator
                n_ucc_excitations .. total-1: add HEA block

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode has already terminated — call reset() first.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action} (action_space size: {self._n_actions})")

        if action < self._n_ucc_excitations:
            # ---- UCC excitation action ----
            excitation = self._available_excitations[action]

            if excitation in self.current_ucc_excitations:
                # Duplicate excitation: penalize but don't terminate
                info = self._make_info()
                self.step_count += 1
                self.done = self._check_termination()
                return self._get_observation(), -1.0, self.done, False, info

            self.current_ucc_excitations.append(excitation)
            param_idx = self._excitation_to_param_idx[excitation]
            if not self.active_ucc_params[param_idx]:
                self.current_ucc_params[param_idx] = 0.0
                self.active_ucc_params[param_idx] = True
            self.current_block_sequence.append({
                "type": "ucc",
                "excitation": excitation,
            })
            self._n_blocks += 1

        else:
            # ---- HEA block action ----
            hea_idx = action - self._n_ucc_excitations
            ent_pattern = self._hea_entanglement_options[
                hea_idx % len(self._hea_entanglement_options)
            ]
            hea_block_idx = len(self.current_hea_blocks)
            self.current_hea_blocks.append({"entanglement_pattern": ent_pattern})
            # Initialize HEA params for this block with small random values
            new_hea_params = self.np_random.uniform(
                -0.1, 0.1, size=self.molecule_data.n_qubits
            )
            self.current_hea_params = np.concatenate(
                [self.current_hea_params, new_hea_params]
            )
            self.current_block_sequence.append({
                "type": "hea",
                "entanglement_pattern": ent_pattern,
                "block_idx": hea_block_idx,
            })
            self._n_blocks += 1

        # Evaluate energy of current circuit
        try:
            energy = self._evaluate_energy()
        except Exception as exc:
            self.done = True
            info = {
                "error": str(exc),
                "energy": self.current_energy,
                "circuit_depth": self._n_blocks,
                "n_blocks": self._n_blocks,
            }
            return self._get_observation(), -10.0, True, False, info

        self.current_energy = energy

        # Track global best across all episodes
        if energy < self.global_best_energy:
            self.global_best_energy = energy
            self.global_best_excitations = list(self.current_ucc_excitations)
            self.global_best_ucc_params = self.current_ucc_params.copy()
            self.global_best_hea_params = self.current_hea_params.copy()

        if energy < self.best_energy:
            self.best_energy = energy

        # Compute reward
        step_info = {"circuit_depth": self._n_blocks, "n_blocks": self._n_blocks}
        reward = self.reward_function.compute_reward(
            None, energy, self.molecule_data.fci_energy, step_info
        )

        self.step_count += 1
        self.done = self._check_termination()
        info = self._make_info()
        return self._get_observation(), reward, self.done, False, info

    def get_circuit(self) -> Optional[HybridCircuit]:
        """Return the current circuit as a HybridCircuit object."""
        builder = HybridCircuitBuilder(self.molecule_data, self.fusion_strategy, self.config)
        template: List[str] = []
        specs: List[Dict] = []
        for exc in self.current_ucc_excitations:
            template.append("UCC")
            specs.append({"excitations": [self._available_excitations.index(exc)]})
        for blk in self.current_hea_blocks:
            template.append("HEA")
            specs.append(blk)
        if not template:
            return None
        return builder.build_hybrid_circuit(template, specs)

    def render(self, mode: str = "human") -> None:
        """Render environment state to stdout."""
        if mode == "human":
            print(
                f"Step: {self.step_count}, Energy: {self.current_energy:.6f} Ha, "
                f"Blocks: {self._n_blocks}, "
                f"UCC excitations: {len(self.current_ucc_excitations)}, "
                f"HEA blocks: {len(self.current_hea_blocks)}"
            )

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_jw_pauli_terms(self, excitation: tuple) -> List:
        """Return JW-mapped Pauli terms for a UCC excitation.

        Uses openfermion JW mapping without index reversal so qubit indices
        match molecule_data.hamiltonian (same convention as HEASearchEnv).

        Returns list of (pauli_string_tuple, complex_coeff) pairs.
        """
        from openfermion import FermionOperator, hermitian_conjugated, jordan_wigner

        if len(excitation) == 2:
            fop = FermionOperator(f"{excitation[0]}^ {excitation[1]}")
        else:
            fop = FermionOperator(
                f"{excitation[0]}^ {excitation[1]}^ {excitation[2]} {excitation[3]}"
            )
        fop = fop - hermitian_conjugated(fop)
        qop = jordan_wigner(fop)
        return [(ps, c) for ps, c in qop.terms.items() if ps]

    def _get_entanglement_pairs(self, pattern: str, n_qubits: int) -> List:
        """Return CNOT (ctrl, tgt) pairs for the given entanglement pattern."""
        if pattern == "linear":
            return [(i, i + 1) for i in range(n_qubits - 1)]
        elif pattern == "circular":
            return [(i, i + 1) for i in range(n_qubits - 1)] + [(n_qubits - 1, 0)]
        else:  # full
            return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    def _get_ham_terms_cache(self) -> List:
        """Parse and cache qubit Hamiltonian as (coeff, xl, yl, zl) tuples."""
        if not hasattr(self, "_ham_terms_cache"):
            ham_terms = []
            for term, coeff in self.molecule_data.hamiltonian.terms.items():
                xl, yl, zl = [], [], []
                for qi, pauli in term:
                    if pauli == "X":
                        xl.append(qi)
                    elif pauli == "Y":
                        yl.append(qi)
                    elif pauli == "Z":
                        zl.append(qi)
                c_real = float(coeff.real) if hasattr(coeff, "real") else float(coeff)
                ham_terms.append((c_real, xl, yl, zl))
            self._ham_terms_cache = ham_terms
        return self._ham_terms_cache

    def _evaluate_energy(self) -> float:
        """Build a hybrid UCC+HEA circuit in tensorcircuit and evaluate ⟨H⟩.

        Blocks are applied in the ORDER chosen by the agent:
          - UCC block: Pauli exponentials from JW mapping (e^{iθP})
          - HEA block: layer of Ry rotations + CNOT entanglement

        All parameters (UCC amplitudes + HEA angles) are jointly optimized
        with JAX autodiff + L-BFGS-B when run_classical_opt=True.

        Falls back to plain UCC evaluation (tencirchem) when no HEA blocks
        have been chosen yet, and to HF energy when no blocks at all.
        """
        if not self.current_block_sequence:
            return self._get_hf_energy()

        # If only UCC blocks chosen: use the fast tencirchem engine
        if not self.current_hea_blocks:
            return self._evaluate_energy_ucc_only()

        # Full hybrid path: tensorcircuit + JAX
        try:
            return self._evaluate_energy_hybrid_jax()
        except Exception:
            # Fallback to UCC-only if JAX fails
            return self._evaluate_energy_ucc_only()

    def _evaluate_energy_ucc_only(self) -> float:
        """Fast UCC-only evaluation using tencirchem engine."""
        if not self.current_ucc_excitations:
            return self._get_hf_energy()

        if self.run_classical_opt:
            from scipy.optimize import minimize

            active_indices = list(dict.fromkeys(
                self._excitation_to_param_idx[exc]
                for exc in self.current_ucc_excitations
            ))

            def energy_func(theta: np.ndarray) -> float:
                p = self.current_ucc_params.copy()
                for i, idx in enumerate(active_indices):
                    p[idx] = theta[i]
                return float(self._ucc_builder.evaluate_energy(None, p))

            x0 = np.array([self.current_ucc_params[i] for i in active_indices])
            result = minimize(energy_func, x0, method="L-BFGS-B",
                              options={"maxiter": 200, "ftol": 1e-14, "gtol": 1e-10})
            for i, idx in enumerate(active_indices):
                self.current_ucc_params[idx] = result.x[i]
            return float(result.fun)
        else:
            return float(self._ucc_builder.evaluate_energy(None, self.current_ucc_params))

    def _evaluate_energy_hybrid_jax(self) -> float:
        """Hybrid UCC+HEA evaluation: tensorcircuit circuit + JAX + L-BFGS-B.

        Qubit index convention:
          - mol.reference_state and mol.hamiltonian use OF/LSB convention: qubit q = bit q
          - tensorcircuit uses MSB convention: qubit 0 = MSB, qubit q = bit (n-1-q)
          - All qubit indices are reversed (q -> n-1-q) before passing to TC operations
            so that the reference state and Hamiltonian remain in OF convention.
        """
        import jax
        import jax.numpy as jnp
        import tensorcircuit as tc
        from scipy.optimize import minimize
        from tencirchem.utils.circuit import evolve_pauli

        jax.config.update("jax_enable_x64", True)

        n_qubits = self.molecule_data.n_qubits
        ref_state = self.molecule_data.reference_state
        ham_terms = self._get_ham_terms_cache()

        # Reverse OF qubit index to TC (MSB) qubit index
        def _rev(q: int) -> int:
            return n_qubits - 1 - q

        # Active UCC param indices (deduplicated, preserving order)
        active_ucc_indices = list(dict.fromkeys(
            self._excitation_to_param_idx[b["excitation"]]
            for b in self.current_block_sequence if b["type"] == "ucc"
        ))
        n_ucc_active = len(active_ucc_indices)
        ucc_idx_map = {idx: i for i, idx in enumerate(active_ucc_indices)}

        # Pre-compute JW Pauli terms (with reversed indices) and entanglement pairs
        block_specs = []
        for blk in self.current_block_sequence:
            if blk["type"] == "ucc":
                pauli_terms_of = self._get_jw_pauli_terms(blk["excitation"])
                # Reverse qubit indices for TC convention
                pauli_terms_tc = [
                    (tuple((_rev(q), p) for q, p in ps), coeff)
                    for ps, coeff in pauli_terms_of
                ]
                param_slot = ucc_idx_map[self._excitation_to_param_idx[blk["excitation"]]]
                block_specs.append(("ucc", pauli_terms_tc, param_slot))
            else:
                pairs_of = self._get_entanglement_pairs(blk["entanglement_pattern"], n_qubits)
                pairs_tc = [(_rev(ctrl), _rev(tgt)) for ctrl, tgt in pairs_of]
                hea_start = blk["block_idx"] * n_qubits
                block_specs.append(("hea", pairs_tc, hea_start))

        # JIT cache key: circuit structure (immutable)
        cache_key = tuple(
            (s[0], s[2]) for s in block_specs
        )
        if not hasattr(self, "_hybrid_jit_cache"):
            self._hybrid_jit_cache = {}

        # Hamiltonian terms in TC qubit convention (reversed indices)
        ham_terms_tc = [
            (c, [_rev(x) for x in xl], [_rev(y) for y in yl], [_rev(z) for z in zl])
            for c, xl, yl, zl in ham_terms
        ]

        with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"):
            if cache_key not in self._hybrid_jit_cache:
                ref_jax = jnp.array(ref_state, dtype=jnp.complex128)
                _ham = ham_terms_tc
                _specs = block_specs
                _nq = n_qubits
                _n_ucc = n_ucc_active
                # HEA qubit indices in TC convention
                _hea_qubits_tc = [_rev(q) for q in range(n_qubits)]

                def energy_fn(theta):
                    c = tc.Circuit(_nq, inputs=ref_jax)
                    for spec in _specs:
                        if spec[0] == "ucc":
                            _, pauli_terms, param_slot = spec
                            t = theta[param_slot]
                            for ps, coeff in pauli_terms:
                                c = evolve_pauli(c, ps, -2.0 * coeff.imag * t)
                        else:
                            _, pairs, hea_start = spec
                            for i_q, q_tc in enumerate(_hea_qubits_tc):
                                c.ry(q_tc, theta=theta[_n_ucc + hea_start + i_q])
                            for ctrl, tgt in pairs:
                                c.cnot(ctrl, tgt)
                    e_terms = [
                        coeff * jnp.real(c.expectation_ps(x=xl, y=yl, z=zl))
                        for coeff, xl, yl, zl in _ham
                    ]
                    return sum(e_terms) if e_terms else jnp.float64(0.0)

                self._hybrid_jit_cache[cache_key] = jax.jit(
                    jax.value_and_grad(energy_fn)
                )

            value_and_grad_jit = self._hybrid_jit_cache[cache_key]

            def scipy_func(theta_np):
                e, g = value_and_grad_jit(jnp.array(theta_np, dtype=jnp.float64))
                return float(e), np.array(g, dtype=np.float64)

            x0 = np.concatenate([
                [self.current_ucc_params[i] for i in active_ucc_indices],
                self.current_hea_params,
            ])

            result = minimize(scipy_func, x0, method="L-BFGS-B", jac=True,
                              options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9})

        # Write back optimized parameters
        for i, idx in enumerate(active_ucc_indices):
            self.current_ucc_params[idx] = result.x[i]
        self.current_hea_params = result.x[n_ucc_active:].copy()

        return float(result.fun)

    def _check_termination(self) -> bool:
        """Return True when the episode should end."""
        if self._n_blocks >= self.max_blocks:
            return True
        if self.step_count >= self.max_depth:
            return True
        if self.molecule_data.fci_energy is not None:
            err = abs(self.current_energy - self.molecule_data.fci_energy)
            if err < 1.6e-3:
                return True
        return False

    def _make_info(self) -> Dict:
        """Build the info dictionary for the current step."""
        return {
            "energy": self.current_energy,
            "circuit_depth": self._n_blocks,
            "n_blocks": self._n_blocks,
            "ucc_excitations": list(self.current_ucc_excitations),
            "hea_blocks": list(self.current_hea_blocks),
            "step": self.step_count,
        }

    def _get_observation(self) -> np.ndarray:
        """Compute the current observation vector.

        Layout: [energy_norm | ucc_arch_one_hot | hea_block_counts | step_norm]
        """
        hf_e = self._get_hf_energy()
        if hf_e != 0.0:
            energy_norm = np.clip(
                (self.current_energy - hf_e) / abs(hf_e), -10.0, 10.0
            )
        else:
            energy_norm = float(np.clip(self.current_energy, -10.0, 10.0))

        # UCC architecture: one-hot over available excitations
        ucc_arch = np.zeros(self._n_ucc_excitations, dtype=np.float32)
        for exc in self.current_ucc_excitations:
            try:
                idx = self._available_excitations.index(exc)
                ucc_arch[idx] = 1.0
            except ValueError:
                pass

        # HEA block counts per entanglement pattern
        hea_counts = np.zeros(self._n_hea_configs, dtype=np.float32)
        for blk in self.current_hea_blocks:
            ent = blk.get("entanglement_pattern", "linear")
            try:
                idx = self._hea_entanglement_options.index(ent)
                hea_counts[idx] += 1.0
            except ValueError:
                pass

        step_norm = float(np.clip(self.step_count / max(self.max_depth, 1), 0.0, 1.0))

        return np.concatenate([
            np.array([energy_norm], dtype=np.float32),
            ucc_arch,
            hea_counts,
            np.array([step_norm], dtype=np.float32),
        ])
