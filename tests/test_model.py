import unittest

import torch

from models.transformer.config import GPT2_124M_CONFIG
from models.transformer.model import GPTModel


class TestGPTModel(unittest.TestCase):

    def test_model_construction(self):
        model = GPTModel(GPT2_124M_CONFIG)

        self.assertIsNotNone(model)

    def test_forward_shape(self):
        model = GPTModel(GPT2_124M_CONFIG)

        batch_size = 2
        sequence_length = 8

        input_ids = torch.randint(
            low=0,
            high=GPT2_124M_CONFIG["vocab_size"],
            size=(batch_size, sequence_length),
        )

        with torch.no_grad():
            logits = model(input_ids)

        expected_shape = (
            batch_size,
            sequence_length,
            GPT2_124M_CONFIG["vocab_size"],
        )

        self.assertEqual(
            logits.shape,
            expected_shape,
        )

    def test_pretrained_weight_loading(self):
        try:
            from gpt_download3 import download_and_load_gpt2
        except ImportError:
            self.skipTest(
                "gpt_download3 is not available"
            )

        from inference.model_loader import (
            load_weights_into_gpt,
        )

        _, params = download_and_load_gpt2(
            model_size="124M",
            models_dir="models/checkpoints/pretrained_gpt2",
        )

        model = GPTModel(GPT2_124M_CONFIG)

        load_weights_into_gpt(
            model,
            params,
        )

        self.assertTrue(
            torch.allclose(
                model.tok_emb.weight,
                torch.tensor(
                    params["wte"],
                    dtype=model.tok_emb.weight.dtype,
                ),
            )
        )

        self.assertTrue(
            torch.allclose(
                model.pos_emb.weight,
                torch.tensor(
                    params["wpe"],
                    dtype=model.pos_emb.weight.dtype,
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()