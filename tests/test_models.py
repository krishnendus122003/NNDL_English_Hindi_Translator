import ast
import json
import unittest
import torch
from api import model_classes
from api.config import PROJECT_ROOT
from api.model_loader import get_model


class ModelTests(unittest.TestCase):
    def test_class_definitions_match_notebook_asts(self):
        recovered = ast.parse((PROJECT_ROOT / "api/model_classes.py").read_text(encoding="utf-8"))
        actual = {n.name: ast.dump(n, include_attributes=False) for n in recovered.body if isinstance(n, ast.ClassDef)}
        for name in ("04_GRU_Seq2Seq", "05_GRU_Bahdanau_Attention", "07_Transformer_1M_Improved"):
            expected = {}
            notebook = json.loads((PROJECT_ROOT / "notebooks" / (name + ".ipynb")).read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                try:
                    tree = ast.parse("".join(cell["source"]))
                except SyntaxError:
                    continue
                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and node.name != "TranslationDataset":
                        expected[node.name] = ast.dump(node, include_attributes=False)
            for key, value in expected.items():
                self.assertEqual(actual[key], value, key)

    def test_transformer_manual_attention_and_masks(self):
        model = get_model("transformer").model
        self.assertFalse(any(isinstance(m, (torch.nn.Transformer, torch.nn.MultiheadAttention)) for m in model.modules()))
        device = next(model.parameters()).device
        tokens = torch.tensor([[2, 5, 0]], device=device)
        mask = model.create_tgt_mask(tokens)[0, 0]
        self.assertFalse(mask[0, 1].item())
        self.assertFalse(mask[:, 2].any().item())
        self.assertTrue(mask[1, 0].item())

    def test_attention_masks_padding(self):
        model = get_model("attention").model
        device = next(model.parameters()).device
        with torch.no_grad():
            memory, hidden, mask = model.encode(torch.tensor([[2, 10, 3, 0]], device=device))
            _, _, weights = model.decoder(torch.tensor([2], device=device), hidden, memory, mask)
        self.assertEqual(weights[0, -1].item(), 0.0)
        self.assertAlmostEqual(weights.sum().item(), 1, places=5)
