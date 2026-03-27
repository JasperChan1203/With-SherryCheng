"""
RLAgent abstract base class for reinforcement learning agents.

This module defines the abstract base class RLAgent, which provides a generic
interface compatible with OpenAI Gym environments. It includes abstract methods
for core agent operations and helper methods for UCC compatibility.

Copied from Task 003, with minimal adaptations for integration.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import numpy as np


class RLAgent(ABC):
    """Abstract base class for reinforcement learning agents.

    This class defines the interface that all RL agents must implement to be
    compatible with the RLQAS framework. The interface is generic and works
    with any OpenAI Gym environment, with additional helper methods for
    UCC (Unitary Coupled Cluster) quantum chemistry tasks.

    Abstract Methods:
        act(state): Select action given current state
        learn(experience): Learn from experience batch
        save(path): Save agent to disk
        load(path): Load agent from disk

    Helper Methods (non-abstract):
        format_ucc_state(energy, circuit_params): Format UCC state for agent input
        parse_ucc_action(action_idx): Parse action index to UCC excitation operator
    """

    @abstractmethod
    def act(self, state: np.ndarray) -> Tuple[int, Dict]:
        """Select action given current state.

        Args:
            state: Current observation from environment as a numpy array.

        Returns:
            Tuple of (action, info_dict) where:
                - action: Integer action index selected by the agent
                - info_dict: Dictionary containing optional additional information
                  (e.g., action probabilities, value function estimate)
        """
        pass

    @abstractmethod
    def learn(self, experience: Dict) -> Dict:
        """Learn from experience batch.

        Args:
            experience: Dictionary containing experience data with keys:
                - states: Batch of states (np.ndarray)
                - actions: Batch of actions (np.ndarray)
                - rewards: Batch of rewards (np.ndarray)
                - next_states: Batch of next states (np.ndarray)
                - dones: Batch of terminal flags (np.ndarray)
                Additional keys may be present depending on the agent.

        Returns:
            Dictionary containing learning metrics (e.g., loss values,
            entropy, learning rate, etc.)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent to disk.

        Args:
            path: File path to save agent. The format depends on the
                  specific agent implementation.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent from disk.

        Args:
            path: File path to load agent from. Must be compatible with
                  the format used by save().
        """
        pass

    def format_ucc_state(self, energy: float, circuit_params: np.ndarray) -> np.ndarray:
        """Format UCC state for agent input.

        This helper method formats quantum chemistry state information
        for use with the agent's act() method. The default implementation
        concatenates energy and circuit parameters into a single vector.

        Args:
            energy: Current circuit energy (float)
            circuit_params: Circuit parameter values as a 1D numpy array

        Returns:
            Formatted state array suitable for passing to act()
        """
        # Default implementation concatenates energy and params
        return np.concatenate([[energy], circuit_params])

    def parse_ucc_action(self, action_idx: int) -> Dict:
        """Parse action index to UCC excitation operator.

        This helper method parses an action index (output of act()) into
        a dictionary describing a UCC excitation operator.

        Args:
            action_idx: Action index from agent (integer)

        Returns:
            Dictionary describing excitation operator with keys:
                - excitation_idx: The action index (default implementation)
                Additional keys may be added by subclasses for more detailed
                operator descriptions.
        """
        # Default implementation returns minimal excitation information
        return {"excitation_idx": action_idx}