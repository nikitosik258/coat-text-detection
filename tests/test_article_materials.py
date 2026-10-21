import tempfile
import unittest
from pathlib import Path

import pandas as pd

from coat_eda.article_materials import run, validate

class ArticleMaterialsTests(unittest.TestCase):
    def test_existing_materials_are_complete_and_consistent(self):
        run(Path("reports/article_materials"), check_only=True)

    def test_generation_creates_fact_based_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "article_materials"
            run(output)
            validate(output)
            metrics = pd.read_csv(output / "tables" / "table_3_final_metrics.csv")
            self.assertEqual(len(metrics), 4)
            self.assertTrue(metrics["F1 (AI=1)"].astype(float).between(0, 1).all())

if __name__ == "__main__":
    unittest.main()
