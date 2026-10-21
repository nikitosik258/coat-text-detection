import unittest
import numpy as np
import pandas as pd
from coat_eda.error_analysis import all_models_wrong, bootstrap, load_frames, metric

class ErrorAnalysisTests(unittest.TestCase):
    def test_confusion_counts_and_rates(self):
        value = metric(np.array([0,0,1,1]), np.array([.1,.9,.2,.8]))
        self.assertEqual((value["tn"],value["fp"],value["fn"],value["tp"]),(1,1,1,1))
        self.assertEqual(value["fpr"], .5); self.assertEqual(value["fnr"], .5)
    def test_paired_bootstrap_is_reproducible(self):
        frame=pd.DataFrame({"id":range(8),"true_label":[0]*4+[1]*4,"predicted_label":[0,0,1,0,1,1,0,1],"probability_ai":[.1,.2,.7,.3,.8,.9,.4,.6]})
        frames={key:frame.copy() for key in ("logreg","mlp_original","mlp_regularized","lstm")}
        self.assertTrue(bootstrap(frames,20,42)[0].equals(bootstrap(frames,20,42)[0]))
    def test_saved_prediction_ids_align(self):
        frames=load_frames(); self.assertEqual(len(frames["lstm"]),24628)
        self.assertTrue(np.array_equal(frames["lstm"].id,frames["logreg"].id))
    def test_four_model_common_error_requires_every_prediction_wrong(self):
        y=np.array([0,1]); base=pd.DataFrame({"id":[10,11],"true_label":y,"probability_ai":[.9,.1]})
        frames={key:base.assign(predicted_label=pred) for key,pred in zip(("logreg","mlp_original","mlp_regularized","lstm"),([1,0],[1,1],[0,0],[1,0]))}
        self.assertEqual(len(all_models_wrong(frames)),0)
    def test_saved_common_errors_cover_all_four_models(self):
        saved=pd.read_csv("reports/error_analysis/all_models_wrong_ids.csv")
        expected=all_models_wrong(load_frames())
        self.assertEqual(saved.columns.tolist(),expected.columns.tolist())
        self.assertTrue(saved.equals(expected.reset_index(drop=True)))

if __name__ == "__main__": unittest.main()
