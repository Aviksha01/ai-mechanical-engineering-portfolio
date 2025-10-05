import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cad_analyzer import CADAnalyzer

class TestCADAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = CADAnalyzer()

    def test_analyze_part(self):
        """Test single part analysis"""
        result = self.analyzer.analyze_part("test_bracket.step")

        # Check that result contains expected keys
        self.assertIn('features', result)
        self.assertIn('optimization_suggestions', result)
        self.assertIn('quality_report', result)
        self.assertIn('quality_score', result)

        # Check that quality score is reasonable
        self.assertGreaterEqual(result['quality_score'], 0)
        self.assertLessEqual(result['quality_score'], 100)

    def test_batch_analyze(self):
        """Test batch analysis"""
        test_files = ["test1.step", "test2.step"]
        results = self.analyzer.batch_analyze(test_files)

        # Check that we get results for all files
        self.assertEqual(len(results), len(test_files))

    def test_feature_extraction(self):
        """Test feature extraction"""
        features = self.analyzer.feature_extractor.extract_features("test_part.step")

        # Check that essential features are present
        self.assertIn('volume', features)
        self.assertIn('surface_area', features)
        self.assertIn('geometric_features', features)
        self.assertIn('quality_metrics', features)

if __name__ == '__main__':
    unittest.main()
