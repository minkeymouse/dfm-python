"""Tests for config.schema.block module."""

import pytest


class TestBlockSchema:
    """Test suite for block schema."""
    
    def test_block_schema_validation(self):
        """Test block schema validation."""
        import numpy as np
        from dfm_python.config.schema.block import BlockStructure
        N, n_blocks = 5, 2
        block_structure = BlockStructure(
            blocks=np.ones((N, n_blocks), dtype=int),
            r=np.array([1, 1], dtype=np.int32),
            p=1,
            p_plus_one=2,
            n_clock_freq=N,
            idio_indicator=np.ones(N, dtype=int)
        )
        assert block_structure is not None
        assert block_structure.is_valid() is True
        assert block_structure.blocks.shape == (N, n_blocks)
        assert block_structure.r.shape == (n_blocks,)

