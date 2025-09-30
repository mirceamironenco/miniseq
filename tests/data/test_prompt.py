import pytest
import torch

from miniseq.data import PromptBatch


def test_prompt_batch_validation():
    # Test valid batch
    PromptBatch(prompt_ids=[[1, 2], [3, 4]], batch_extras=["a", "b"])

    # Test mismatched prompt_ids and prompt_extras
    with pytest.raises(ValueError):
        PromptBatch(prompt_ids=[[1, 2], [3, 4]], batch_extras=["a"])

    # Test mismatched prompt_ids_pt and prompt_ids
    with pytest.raises(ValueError):
        PromptBatch(
            prompt_ids=[[1, 2], [3, 4]],
            prompt_ids_pt=[torch.tensor([1, 2])],
            batch_extras=["a", "b"],
        )


def test_prompt_batch_to_device():
    batch = PromptBatch(
        prompt_ids=[[1, 2], [3, 4]],
        prompt_ids_pt=[torch.tensor([1, 2]), torch.tensor([3, 4])],
        batch_extras=["a", "b"],
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        batch.to(device)
        assert batch.prompt_ids_pt[0].device.type == "cuda"
        assert batch.prompt_ids_pt[1].device.type == "cuda"


def test_prompt_batch_pin_memory():
    batch = PromptBatch(
        prompt_ids=[[1, 2], [3, 4]],
        prompt_ids_pt=[torch.tensor([1, 2]), torch.tensor([3, 4])],
        batch_extras=["a", "b"],
    )

    if torch.cuda.is_available():
        batch.pin_memory()
        assert batch.prompt_ids_pt[0].is_pinned()
        assert batch.prompt_ids_pt[1].is_pinned()
