import json
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def test_analytical_notebooks_are_valid_and_present(self):
        root = Path("notebooks")
        for name in ("01_eda.ipynb", "02_error_analysis.ipynb", "03_final_audit.ipynb", "04_article_materials.ipynb"):
            payload = json.loads((root / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["nbformat"], 4)
            code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
            self.assertTrue(code_cells)
            for index, cell in enumerate(code_cells):
                compile("".join(cell["source"]), f"{name}:cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
