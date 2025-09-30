import torch

from miniseq.data import PipelineBuilder


def test_pipeline_builder():
    # 1. Create a simple MapDataset (a list of integers).
    dataset = list(range(10))

    # 2. Create a PipelineBuilder from the dataset.
    builder = PipelineBuilder.from_map_dataset(dataset)
    assert builder.iterator_len == 10

    # 3. Map: square each number.
    builder = builder.map(lambda x: x * x)
    assert builder.iterator_len == 10

    # 4. Batch the results.
    builder = builder.batch(batch_size=2, drop_last=False)

    # 5. Collect the results.
    result = list(builder.as_loader())

    # 6. Verify the output.
    expected = [[0, 1], [4, 9], [16, 25], [36, 49], [64, 81]]
    assert result == expected, f"Expected {expected}, but got {result}"


def test_pipeline_builder_collate():
    # 1. Create a simple MapDataset (a list of integers).
    dataset = list(range(5))

    # 2. Create a PipelineBuilder from the dataset.
    builder = PipelineBuilder.from_map_dataset(dataset)

    # 3. Batch the results.
    builder = builder.batch(batch_size=2, drop_last=False)

    # 4. Collate the batches into a single tensor.
    def collate_fn(batch):
        return torch.tensor(batch)

    builder = builder.collate(fn=collate_fn)

    # 5. Collect the results.
    result = list(builder.as_loader())

    # 6. Verify the output.
    expected = [torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4])]
    for res, exp in zip(result, expected):
        assert torch.equal(res, exp)


def test_pipeline_builder_pin_memory_and_prefetch():
    # This test mainly checks that the pipeline can be constructed with these
    # operations without errors.

    # 1. Create a simple MapDataset.
    dataset = [torch.randn(4) for _ in range(10)]

    # 2. Create a PipelineBuilder.
    builder = PipelineBuilder.from_map_dataset(dataset)

    # 3. Add pin_memory and prefetch.
    if torch.cuda.is_available():
        builder = builder.pin_memory()

    builder = builder.prefetch(prefetch_factor=2)

    # 4. Collect the results.
    result = list(builder.as_loader())

    # 5. Verify the output.
    assert len(result) == 10
    for i in range(10):
        assert torch.equal(result[i], dataset[i])
