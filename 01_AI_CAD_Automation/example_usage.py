#!/usr/bin/env python3
"""
Example usage of the AI-Driven CAD Automation System
"""

from src.cad_analyzer import CADAnalyzer
import json

def main():
    # Initialize analyzer
    analyzer = CADAnalyzer()

    # Example 1: Single part analysis
    print("=== Single Part Analysis ===")
    result = analyzer.analyze_part("sample_bracket.step")

    print(f"Quality Score: {result.get('quality_score', 'N/A')}")
    print(f"Number of optimization suggestions: {len(result.get('optimization_suggestions', {}).get('suggestions', []))}")

    # Display top suggestions
    if 'optimization_suggestions' in result:
        suggestions = result['optimization_suggestions'].get('priority_suggestions', [])
        print("\nTop 3 Optimization Suggestions:")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"{i}. {suggestion.get('type', 'N/A')}: {suggestion.get('description', suggestion.get('benefit', 'N/A'))}")

    # Example 2: Quality assessment details
    print("\n=== Quality Assessment ===")
    if 'quality_report' in result:
        quality = result['quality_report']
        print(f"Overall Grade: {quality.get('quality_grade', 'N/A')}")
        print("Individual Scores:")
        for category, score in quality.get('individual_scores', {}).items():
            print(f"  {category}: {score:.1f}")

    # Example 3: Batch analysis
    print("\n=== Batch Analysis ===")
    sample_files = ["bracket.step", "gear.step", "housing.step"]
    batch_results = analyzer.batch_analyze(sample_files)
    print(f"Analyzed {len(batch_results)} parts")
    print(f"Average quality score: {batch_results.get('quality_score', pd.Series([0])).mean():.1f}")

if __name__ == "__main__":
    main()
