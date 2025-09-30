from unittest.mock import MagicMock

import pytest

from miniseq.trainer import Trainer, TrainerState


@pytest.fixture
def mock_trainer_dependencies():
    mock_machine = MagicMock()
    mock_machine.device.type = "cpu"
    return {
        "model_config": MagicMock(),
        "train_unit": MagicMock(),
        "train_loader": MagicMock(),
        "machine": mock_machine,
        "optimizer": MagicMock(),
        "lr_scheduler": MagicMock(),
        "checkpoint_manager": MagicMock(),
        "memory_tracker": MagicMock(),
        "progress_repoter": MagicMock(),
        "validator": MagicMock(),
    }


def test_trainer_constructor_validation(mock_trainer_dependencies):
    with pytest.raises(ValueError, match="grad_accum_steps"):
        Trainer(
            **mock_trainer_dependencies,
            seed=42,
            requires_rollout=False,
            total_steps=100,
            grad_accum_steps=0,
        )


def test_trainer_should_stop_training(mock_trainer_dependencies):
    # Test with total_steps
    trainer = Trainer(
        **mock_trainer_dependencies,
        seed=42,
        requires_rollout=False,
        total_steps=100,
    )
    trainer._global_step = 99
    assert not trainer._should_stop_training()
    trainer._global_step = 100
    assert trainer._should_stop_training()

    # Test with total_epochs
    trainer = Trainer(
        **mock_trainer_dependencies,
        seed=42,
        requires_rollout=False,
        total_steps=100,  # total_epochs should take precedence
        total_epochs=5,
    )
    trainer._epoch_step = 4
    assert not trainer._should_stop_training()
    trainer._epoch_step = 5
    assert trainer._should_stop_training()


def test_trainer_should_validate(mock_trainer_dependencies):
    trainer = Trainer(
        **mock_trainer_dependencies,
        seed=42,
        requires_rollout=False,
        total_steps=100,
        validate_every_n_steps=10,
    )
    trainer._global_step = 9
    assert not trainer._should_validate()
    trainer._global_step = 10
    assert trainer._should_validate()
    trainer._global_step = 11
    assert not trainer._should_validate()

    # Test with no validator
    no_validator_deps = mock_trainer_dependencies.copy()
    del no_validator_deps["validator"]
    trainer = Trainer(
        **no_validator_deps,
        seed=42,
        requires_rollout=False,
        total_steps=100,
        validate_every_n_steps=10,
        validator=None,
    )
    assert not trainer._should_validate()


def test_trainer_state():
    mock_trainer = MagicMock()
    mock_trainer._global_step = 123
    mock_trainer._epoch_step = 456
    mock_trainer._metric_bag.state_dict.return_value = {"loss": 0.5}
    mock_trainer._lr_scheduler.state_dict.return_value = {"lr": 0.001}

    state = TrainerState(mock_trainer)
    state_dict = state.state_dict()

    assert state_dict["global_step"] == 123
    assert state_dict["epoch_step"] == 456
    assert state_dict["metric_bag"] == {"loss": 0.5}
    assert state_dict["lr_scheduler"] == {"lr": 0.001}

    # Test loading state
    new_mock_trainer = MagicMock()
    new_state = TrainerState(new_mock_trainer)
    new_state.load_state_dict(state_dict)

    assert new_mock_trainer._global_step == 123
    assert new_mock_trainer._epoch_step == 456
    new_mock_trainer._metric_bag.load_state_dict.assert_called_once_with({"loss": 0.5})
    new_mock_trainer._lr_scheduler.load_state_dict.assert_called_once_with(
        {"lr": 0.001}
    )
