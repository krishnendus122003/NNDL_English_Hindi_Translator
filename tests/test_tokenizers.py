import unittest
from api.model_loader import load_tokenizers
from api.config import TOKENIZER_DIR


class TokenizerTests(unittest.TestCase):
    def test_all_files_exist(self):
        for suffix in ("", "_1m"):
            for language in ("english", "hindi"):
                self.assertTrue((TOKENIZER_DIR / f"{language}_bpe{suffix}.model").is_file())

    def test_tokenizers_load_and_verify_ids(self):
        for improved in (False, True):
            for tokenizer in load_tokenizers(improved):
                self.assertEqual((tokenizer.pad_id(), tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.eos_id()), (0, 1, 2, 3))
                self.assertEqual(tokenizer.vocab_size(), 16000)

    def test_cached_pairs_are_distinct_and_stable(self):
        self.assertIs(load_tokenizers(), load_tokenizers())
        self.assertIsNot(load_tokenizers()[0], load_tokenizers(True)[0])
