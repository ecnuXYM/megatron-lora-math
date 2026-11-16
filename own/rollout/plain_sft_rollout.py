import logging

from miles.utils.processing_utils import load_tokenizer

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

TOKENIZER = None
SAMPLE_PRINTED = False


def _encode(text: str) -> list[int]:
    return TOKENIZER(text, add_special_tokens=False)["input_ids"]


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(
            args.hf_checkpoint,
            chat_template_path=args.chat_template_path,
            trust_remote_code=True,
        )

    samples = data_buffer.get_samples(args.rollout_batch_size)

    for i, sample_group in enumerate(samples):
        (sample,) = sample_group
        prompt = sample.prompt
        if not isinstance(prompt, str):
            raise TypeError(f"plain_sft_rollout expects string prompt, got {type(prompt)}")

        completion = sample.metadata.get("completion")
        if completion is None:
            completion = sample.label
        if not isinstance(completion, str) or not completion:
            raise ValueError("plain_sft_rollout requires metadata.completion or label")

        prompt_ids = _encode(prompt)
        completion_ids = _encode(completion)
        if TOKENIZER.eos_token_id is not None:
            completion_ids = completion_ids + [TOKENIZER.eos_token_id]

        token_ids = prompt_ids + completion_ids
        response_length = len(completion_ids)

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = [1] * response_length

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                "plain_sft_rollout example: prompt_len=%s completion_len=%s total_len=%s prompt=%r completion=%r",
                len(prompt_ids),
                len(completion_ids),
                len(token_ids),
                prompt[:300],
                completion[:300],
            )
            SAMPLE_PRINTED = True

    return samples
