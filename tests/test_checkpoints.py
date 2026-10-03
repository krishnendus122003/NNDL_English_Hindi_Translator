import unittest
from api.model_loader import get_model
from api.config import CHECKPOINTS


class CheckpointTests(unittest.TestCase):
    def check_load(self, name, epoch):
        self.assertTrue(CHECKPOINTS[name].is_file())
        bundle = get_model(name)  # loader must execute strict=True on first load
        self.assertEqual(bundle.epoch, epoch)
        self.assertFalse(bundle.model.training)
        self.assertIs(bundle, get_model(name))

    def test_gru_strict_load(self):
        self.check_load("gru", 3)

    def test_attention_strict_load(self):
        self.check_load("attention", 4)
        self.assertEqual(tuple(get_model("attention").model.decoder.pre_output.weight.shape), (512, 1280))

    def test_transformer_strict_load_epoch19(self):
        self.check_load("transformer", 19)
        self.assertEqual(CHECKPOINTS["transformer"].name, "transformer_1m_improved_best.pt")
