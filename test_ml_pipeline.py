import unittest
import os
import json


class TestMLPipeline(unittest.TestCase):

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists("heart_model.pkl"))

    def test_accuracy_exists(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertIn("accuracy", metrics)

    def test_accuracy_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
