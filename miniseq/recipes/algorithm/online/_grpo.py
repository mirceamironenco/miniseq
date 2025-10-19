import itertools
from dataclasses import dataclass
from typing import Literal

import torch
from typing_extensions import override

from miniseq.data import CompletionScorer, PromptBatch, SequenceBatch, TrajectoryBatch
from miniseq.generation import Generator
from miniseq.machine import Machine
from miniseq.metric_bag import MetricBag, metrics
from miniseq.recipes.algorithm import (
    model_logps,
    packed_scatter_sum_reduce,
    update_lengths,
    update_seq_batch_metrics,
    update_sum_loss,
)
from miniseq.recipes.algorithm.online._base import OnlineTrainUnit
from miniseq.transformer import CausalTransformerModel
from miniseq.utils import to_tensor


@dataclass(kw_only=True, frozen=True)
class GRPOConfig:
    group_size: int = 4
    mu: int = 1
    std_normalize_advantage: bool = False
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.2
    beta: float = 0.0
    loss: Literal["grpo", "gspo"] = "grpo"
    rollout_correction: bool = True


@torch.inference_mode()
def update_avg_reward(metric_bag: MetricBag, avg_reward: torch.Tensor) -> None:
    metric_bag.get(metrics.Mean, "avg_reward").update(avg_reward, weight=1)


@torch.inference_mode()
def update_reward_groups_same(
    metric_bag: MetricBag, same_group_reward: torch.Tensor, num_groups: int
) -> None:
    assert same_group_reward.numel() == 1

    metric_bag.get(metrics.Mean, "group_reward_same").update(
        same_group_reward / num_groups, weight=num_groups
    )


@torch.inference_mode()
def update_rollout_size(metric_bag: MetricBag, trajectory: TrajectoryBatch) -> None:
    batch_size = to_tensor(trajectory.batch_size, device=metric_bag.device)
    metric_bag.get(metrics.Sum, "batch/rollout_batch_size").update(batch_size)


class GRPOUnit(OnlineTrainUnit):
    _model: CausalTransformerModel
    _machine: Machine
    _generator: Generator
    _reference_model: CausalTransformerModel | None
    _completion_scorer: CompletionScorer
    _group_size: int
    _std_normalize_advantage: bool
    _mu: int
    _clip_eps_low: float
    _clip_eps_high: float
    _beta: float
    _loss: Literal["grpo", "gspo"]
    _rollout_correction: bool
    _packed: bool
    _prefix_sharing: bool
    _pad_index: int

    def __init__(
        self,
        model: CausalTransformerModel,
        machine: Machine,
        generator: Generator,
        trajectory_size: int,
        *,
        reference_model: CausalTransformerModel | None,
        completion_scorer: CompletionScorer,
        group_size: int,
        std_normalize_advantage: bool = True,
        mu: int = 1,
        clip_eps_low: float = 0.0,
        clip_eps_high: float = 0.0,
        beta: float = 0.0,
        loss: Literal["grpo", "gspo"] = "grpo",
        rollout_correction: bool = False,
        packed: bool = False,
        prefix_sharing: bool = False,
        pad_index: int = 0,
    ) -> None:
        super().__init__(
            model, machine, generator, trajectory_size, trajectory_epochs=mu
        )

        if mu < 1:
            raise ValueError(
                "mu (number of GRPO iterations) must be a positive integer."
            )

        if not 0.0 <= clip_eps_low < 1.0:
            raise ValueError(f"clip parameter ({clip_eps_low}) must be in [0,1).")

        if not 0.0 <= clip_eps_high < 1.0:
            raise ValueError(f"clip parameter ({clip_eps_high}) must be in [0,1).")

        if beta > 0.0 and reference_model is None:
            raise ValueError("Reference model cannot be None for KL beta > 0.")

        if beta == 0.0 and reference_model is not None:
            raise ValueError("KL Beta = 0.0, set reference model to None.")

        self._reference_model = reference_model
        self._completion_scorer = completion_scorer
        self._group_size = group_size
        self._std_normalize_advantage = std_normalize_advantage

        # assert mu == 1, "mu > 1 not yet implemented."

        # Number of GRPO iterations.
        self._mu = mu

        # IP ratio clip
        self._clip_eps_low = clip_eps_low
        self._clip_eps_high = clip_eps_high

        # KL Beta
        self._beta = beta

        self._loss = loss

        self._rollout_correction = rollout_correction

        self._packed = packed

        self._prefix_sharing = prefix_sharing

        self._pad_index = pad_index

    def _compute_rewards(
        self, completions: list[list[int]], batch: PromptBatch, metric_bag: MetricBag
    ) -> torch.Tensor:
        rewards = self._completion_scorer(
            completions=completions,
            batch=batch,
            repetitions=self._group_size,
            metric_bag=metric_bag,
        )

        # (bsz * group_size,)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self._machine.device)

        # (bsz, group_size)
        rewards = rewards.reshape(-1, self._group_size)

        return rewards

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        # (bsz, group_size)
        advantages = rewards - rewards.mean(dim=1, keepdim=True)

        if self._std_normalize_advantage:
            advantages /= advantages.std(dim=1, keepdim=True) + 1e-6

        return advantages

    @override
    def generate_and_score(
        self, batch: PromptBatch, metric_bag: MetricBag
    ) -> TrajectoryBatch:
        trajectory_prompt_ids = [
            input_ids[:]
            for prompt_ids in batch.prompt_ids
            for input_ids in itertools.repeat(prompt_ids, self._group_size)
        ]

        completions, rollout_logps = self._generator.generate(trajectory_prompt_ids)

        # (bsz, group_size)
        rewards = self._compute_rewards(completions, batch, metric_bag)

        reward_group_same = torch.all(rewards == rewards[..., 0][..., None], dim=1)

        update_reward_groups_same(
            metric_bag, reward_group_same.float().sum(), rewards.size(0)
        )

        # (bsz, group_size)
        advantages = self._compute_advantages(rewards)

        # Record prompt & completion lengths
        prompt_lens = to_tensor(
            list(map(len, trajectory_prompt_ids)),
            dtype=torch.int64,
            device=metric_bag.device,
        )
        completion_lens = to_tensor(
            list(map(len, completions)), dtype=torch.int64, device=metric_bag.device
        )

        update_lengths(
            metric_bag,
            prompt_lens.sum(),
            num_sequences=prompt_lens.numel(),
            name="prompt_len",
        )

        update_lengths(
            metric_bag,
            completion_lens.sum(),
            num_sequences=completion_lens.numel(),
            name="completion_len",
        )

        return TrajectoryBatch(
            prompt_ids=trajectory_prompt_ids,
            completion_ids=completions,
            advantages=advantages,
            rewards=rewards,
            rollout_logps=rollout_logps,
            pad_idx=self._pad_index,
            packed=self._packed,
            prefix_sharing=self._prefix_sharing,
        )

    # @torch.compile(dynamic=True)
    def _packed_grpo_loss_broadcast(
        self,
        advantage: torch.Tensor,
        input_batch: SequenceBatch,
        *,
        pi_logps: torch.Tensor,
        old_logps: torch.Tensor,
        ref_logps: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # (1, packed_seqlen)
        ip_ratio = (pi_logps - old_logps).exp()

        lengths, seqlen = input_batch.full_lengths(), pi_logps.size(1)

        # (num_seqs * group_size,)
        advantage = advantage.flatten()

        # (1, packed_seqlen)
        advantage = advantage.repeat_interleave(lengths, output_size=seqlen)[None, ...]

        # (1, packed_seqlen)
        policy_adv = ip_ratio * advantage

        if self._clip_eps_high > 0.0:
            clamped_ratio = ip_ratio.clamp(
                1.0 - self._clip_eps_low, 1.0 + self._clip_eps_high
            )
            clamped_policy_adv = clamped_ratio * advantage
            policy_adv = torch.min(policy_adv, clamped_policy_adv)

        if ref_logps is not None:
            # (1, packed_seqlen)
            kl_div = (ref_logps - pi_logps).exp() - (ref_logps - pi_logps) - 1.0

            policy_adv = policy_adv - self._beta * kl_div

        if target_mask is not None:
            policy_adv = policy_adv * target_mask

        loss = -policy_adv.sum()

        del policy_adv

        return loss

    def _grpo_loss(
        self,
        advantage: torch.Tensor,
        input_batch: SequenceBatch,
        *,
        pi_logps: torch.Tensor,
        old_logps: torch.Tensor,
        ref_logps: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # (bsz * group_size, seqlen)
        ip_ratio = (pi_logps - old_logps).exp()

        # (bsz * group_size, 1)
        advantage = advantage.flatten()[..., None]

        policy_adv = ip_ratio * advantage

        if self._clip_eps_high > 0.0:
            clamped_ratio = ip_ratio.clamp(
                1.0 - self._clip_eps_low, 1.0 + self._clip_eps_high
            )
            clamped_policy_adv = clamped_ratio * advantage
            policy_adv = torch.min(policy_adv, clamped_policy_adv)

        if ref_logps is not None:
            kl_div = (ref_logps - pi_logps).exp() - (ref_logps - pi_logps) - 1.0

            policy_adv = policy_adv - self._beta * kl_div

        if target_mask is not None:
            policy_adv = policy_adv * target_mask

        loss = -policy_adv.sum()

        del policy_adv

        return loss

    def _full_gspo(
        self,
        advantage: torch.Tensor,
        input_batch: SequenceBatch,
        *,
        pi_logps: torch.Tensor,
        old_logps: torch.Tensor,
        ref_logps: torch.Tensor | None = None,
        rollout_logps: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_ip_diff = pi_logps - old_logps

        if target_mask is not None:
            log_ip_diff = log_ip_diff * target_mask

        if input_batch.is_packed:
            # Currently not supported.
            assert not input_batch.uses_prefix_sharing()

            # (B * num_groups,)
            log_ip_diff = packed_scatter_sum_reduce(
                log_ip_diff,
                document_ids=input_batch.document_ids,
                num_sequences=input_batch.num_examples,
            )

            if target_mask is not None:
                completion_mask = packed_scatter_sum_reduce(
                    target_mask,
                    document_ids=input_batch.document_ids,
                    num_sequences=input_batch.num_examples,
                )

                log_ip_diff = log_ip_diff / completion_mask
        else:
            log_ip_diff = log_ip_diff.sum(-1)

            if target_mask is not None:
                log_ip_diff = log_ip_diff / target_mask.sum(-1)

        # (B * num_groups,)
        ip_ratio = log_ip_diff.exp()

        advantage = advantage.flatten()

        policy_adv = ip_ratio * advantage

        if self._clip_eps_high > 0.0:
            clamped_ratio = ip_ratio.clamp(
                1.0 - self._clip_eps_low, 1.0 + self._clip_eps_high
            )
            clamped_policy_adv = clamped_ratio * advantage
            policy_adv = torch.min(policy_adv, clamped_policy_adv)

        loss = -policy_adv.sum()

        del policy_adv

        return loss

    def _full_grpo(
        self,
        advantage: torch.Tensor,
        input_batch: SequenceBatch,
        *,
        pi_logps: torch.Tensor,
        old_logps: torch.Tensor,
        ref_logps: torch.Tensor | None = None,
        rollout_logps: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_ip_diff = (pi_logps - old_logps).clamp(min=-20.0, max=20.0)

        ip_ratio = log_ip_diff.exp()

        advantage = advantage.flatten()

        if input_batch.is_packed:
            lengths, seqlen = input_batch.full_lengths(), pi_logps.size(1)

            advantage = advantage.repeat_interleave(lengths, output_size=seqlen)

            advantage.unsqueeze_(0)
        else:
            advantage.unsqueeze_(-1)

        policy_adv = ip_ratio * advantage

        if self._clip_eps_high > 0.0:
            clamped_ratio = ip_ratio.clamp(
                1.0 - self._clip_eps_low, 1.0 + self._clip_eps_high
            )
            clamped_policy_adv = clamped_ratio * advantage
            policy_adv = torch.min(policy_adv, clamped_policy_adv)

        if rollout_logps is not None:
            # Note: flipped sign from standard implementation since we flip again later
            rollout_ip_ratio = torch.exp(rollout_logps - old_logps).clamp(max=2.0)

            policy_adv = policy_adv * rollout_ip_ratio

        if ref_logps is not None:
            kl_div = (ref_logps - pi_logps).exp() - (ref_logps - pi_logps) - 1.0

            policy_adv = policy_adv - self._beta * kl_div

        if target_mask is not None:
            policy_adv = policy_adv * target_mask

        loss = -policy_adv.sum()

        del policy_adv

        return loss

    @override
    def compute_loss(
        self, batch: TrajectoryBatch, metric_bag: MetricBag
    ) -> tuple[torch.Tensor, int | None]:
        input_batch, target_batch = batch.auto_regressive_input()

        # (1, packed_seqlen) or (bsz * group_size, seqlen)
        pi_logps = model_logps(input_batch, self._model, target_batch=target_batch)

        ref_logps: torch.Tensor | None = None
        if self._reference_model is not None:
            with torch.no_grad():
                ref_logps = model_logps(
                    input_batch, self._reference_model, target_batch=target_batch
                )

        old_logps = batch.old_lprobs()
        if old_logps is None:
            old_logps = pi_logps.detach()

        rollout_logps: torch.Tensor | None = None
        if self._rollout_correction:
            rollout_logps = batch.maybe_rollout_lprobs(input_batch)

        match self._loss:
            case "grpo":
                loss = self._full_grpo(
                    batch.advantages,
                    input_batch,
                    pi_logps=pi_logps,
                    old_logps=old_logps,
                    ref_logps=ref_logps,
                    rollout_logps=rollout_logps,
                    target_mask=target_batch.target_mask,
                )
            case "gspo":
                loss = self._full_gspo(
                    batch.advantages,
                    input_batch,
                    pi_logps=pi_logps,
                    old_logps=old_logps,
                    ref_logps=ref_logps,
                    rollout_logps=rollout_logps,
                    target_mask=target_batch.target_mask,
                )
            case _:
                raise ValueError(f"Expected loss type grpo/gspo, got {self._loss}")

        # Cache if mu > 1 (doing multiple GRPO iterations)
        # and not already cached.
        if self._mu > 1 and batch.old_lprobs() is None:
            batch.set_log_probs(old_logps.clone())

        update_sum_loss(
            metric_bag,
            -pi_logps.sum(),
            target_batch.num_target_elements,
            name="nll_loss",
        )

        update_sum_loss(
            metric_bag,
            loss,
            target_batch.num_target_elements,
            name=f"{self._loss}_loss",
        )

        update_avg_reward(metric_bag, batch.rewards.mean())

        update_seq_batch_metrics(metric_bag, target_batch)

        update_rollout_size(metric_bag, batch)

        del pi_logps
        del ref_logps

        return loss, target_batch.num_target_elements
