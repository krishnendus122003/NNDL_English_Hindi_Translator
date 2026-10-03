"""Compare application decoding with isolated notebook function definitions.

Only function ASTs execute, never notebook cells, datasets or evaluation loops.
"""
import ast
import json
import unittest
import torch
from api.config import PROJECT_ROOT
from api.decoding import decode
from api.model_loader import get_model


def notebook_functions(filename, names, context):
    notebook = json.loads((PROJECT_ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    selected = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                selected[node.name] = node
    if set(selected) != set(names):
        raise AssertionError("Requested notebook inference definitions were not found")
    for node in selected.values():
        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), context)
    return context


class NotebookDecodingTests(unittest.TestCase):
    def context(self, bundle):
        return dict(torch=torch, F=torch.nn.functional, BOS_ID=2, EOS_ID=3, PAD_ID=0,
                    MAX_SEQUENCE_LENGTH=bundle.max_length, device=next(bundle.model.parameters()).device)

    def test_gru_matches_notebook(self):
        bundle = get_model("gru")
        context = notebook_functions("04_GRU_Seq2Seq.ipynb", ["encode_sentence", "translate_greedy"], self.context(bundle))
        expected = context["translate_greedy"]("I am happy.", bundle.model, bundle.source, bundle.target)
        self.assertEqual(decode(bundle, "I am happy.", "greedy")["translation"], expected.strip())

    def test_attention_text_and_displayed_weights_match_notebook(self):
        bundle = get_model("attention")
        context = notebook_functions("05_GRU_Bahdanau_Attention.ipynb", ["encode_sentence", "translate_with_attention"], self.context(bundle))
        expected, matrix = context["translate_with_attention"]("I am happy.", bundle.model, bundle.source, bundle.target, context["device"])
        actual = decode(bundle, "I am happy.", "greedy")
        self.assertEqual(actual["translation"], expected.strip())
        rows, columns = len(actual["target_tokens"]), len(actual["source_tokens"])
        torch.testing.assert_close(torch.tensor(actual["attention_matrix"]), matrix[:rows, :columns])

    def test_transformer_greedy_and_beam_match_notebook(self):
        bundle = get_model("transformer")
        context = notebook_functions("07_Transformer_1M_Improved.ipynb", ["greedy_translate", "beam_search_translate"], self.context(bundle))
        greedy = context["greedy_translate"](bundle.model, "I am happy.", bundle.source, bundle.target, context["device"])
        beam = context["beam_search_translate"]("I am happy.", bundle.model, bundle.source, bundle.target, context["device"], beam_size=4, length_penalty=.6)
        self.assertEqual(decode(bundle, "I am happy.", "greedy")["translation"], greedy.strip())
        self.assertEqual(decode(bundle, "I am happy.", "beam")["translation"], beam.strip())
