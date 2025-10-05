import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TestReportGenerator:
    """
    Automated generator for engineering test reports with comprehensive
    data analysis, visualizations, and professional formatting
    """

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.test_data = None
        self.analysis_results = {}
        self.figures = []

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for test report generation"""
        return {
            'company_name': 'Engineering Test Laboratory',
            'report_title': 'Material Testing Report',
            'test_standard': 'ASTM E8/E8M',
            'output_formats': ['pdf', 'html', 'excel'],
            'chart_style': 'seaborn',
            'include_statistical_analysis': True,
            'include_visualizations': True,
            'quality_threshold': 0.95
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._default_config()

    def load_test_data(self, data_source: str, data_type: str = 'csv') -> pd.DataFrame:
        """
        Load test data from various sources

        Args:
            data_source: Path to data file or data itself
            data_type: Type of data source ('csv', 'excel', 'json', 'dict')

        Returns:
            Loaded test data as DataFrame
        """

        if data_type == 'csv':
            self.test_data = pd.read_csv(data_source)
        elif data_type == 'excel':
            self.test_data = pd.read_excel(data_source)
        elif data_type == 'json':
            with open(data_source, 'r') as f:
                data = json.load(f)
            self.test_data = pd.DataFrame(data)
        elif data_type == 'dict':
            self.test_data = pd.DataFrame(data_source)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        print(f"Loaded test data: {len(self.test_data)} records, {len(self.test_data.columns)} columns")
        return self.test_data

    def analyze_test_data(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of test data

        Returns:
            Dictionary containing analysis results
        """

        if self.test_data is None:
            raise ValueError("No test data loaded. Call load_test_data() first.")

        # Basic statistics
        numerical_columns = self.test_data.select_dtypes(include=[np.number]).columns
        basic_stats = self.test_data[numerical_columns].describe()

        # Identify key test parameters
        test_results = self._identify_test_parameters()

        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(numerical_columns)

        # Quality assessment
        quality_assessment = self._assess_data_quality()

        # Pass/fail analysis
        pass_fail_analysis = self._analyze_pass_fail_criteria()

        self.analysis_results = {
            'basic_statistics': basic_stats,
            'test_parameters': test_results,
            'statistical_analysis': statistical_analysis,
            'quality_assessment': quality_assessment,
            'pass_fail_analysis': pass_fail_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }

        return self.analysis_results

    def _identify_test_parameters(self) -> Dict[str, Any]:
        """Identify and categorize test parameters"""

        # Common test parameter patterns
        parameter_patterns = {
            'strength': ['tensile_strength', 'yield_strength', 'ultimate_strength', 'strength'],
            'stress': ['stress', 'sigma', 'tensile_stress', 'yield_stress'],
            'strain': ['strain', 'epsilon', 'elongation', 'deformation'],
            'modulus': ['modulus', 'youngs_modulus', 'elastic_modulus', 'e_modulus'],
            'force': ['force', 'load', 'applied_force', 'maximum_force'],
            'displacement': ['displacement', 'extension', 'deflection'],
            'temperature': ['temperature', 'temp', 'test_temp'],
            'time': ['time', 'duration', 'test_time']
        }

        identified_parameters = {}

        for category, patterns in parameter_patterns.items():
            for column in self.test_data.columns:
                column_lower = column.lower()
                if any(pattern in column_lower for pattern in patterns):
                    if category not in identified_parameters:
                        identified_parameters[category] = []
                    identified_parameters[category].append(column)

        return identified_parameters

    def _perform_statistical_analysis(self, numerical_columns: pd.Index) -> Dict[str, Any]:
        """Perform advanced statistical analysis"""

        analysis = {}

        for column in numerical_columns:
            data = self.test_data[column].dropna()

            if len(data) > 0:
                # Distribution analysis
                from scipy import stats

                # Normality test (if scipy available)
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limit for performance
                    is_normal = shapiro_p > 0.05
                except:
                    is_normal = False
                    shapiro_stat, shapiro_p = None, None

                # Confidence intervals
                confidence_95 = {
                    'lower': np.percentile(data, 2.5),
                    'upper': np.percentile(data, 97.5)
                }

                analysis[column] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'median': float(data.median()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'is_normal_distribution': is_normal,
                    'shapiro_test': {
                        'statistic': float(shapiro_stat) if shapiro_stat else None,
                        'p_value': float(shapiro_p) if shapiro_p else None
                    },
                    'confidence_interval_95': confidence_95,
                    'outliers_count': self._count_outliers(data)
                }

        return analysis

    def _count_outliers(self, data: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return len(outliers)

    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess overall data quality"""

        total_data_points = len(self.test_data)
        missing_data = self.test_data.isnull().sum()

        quality_score = 1.0 - (missing_data.sum() / (len(self.test_data.columns) * total_data_points))

        return {
            'total_records': total_data_points,
            'missing_data_by_column': missing_data.to_dict(),
            'data_completeness': float(quality_score),
            'quality_rating': 'Excellent' if quality_score > 0.95 else 
                           'Good' if quality_score > 0.85 else
                           'Acceptable' if quality_score > 0.75 else 'Poor'
        }

    def _analyze_pass_fail_criteria(self) -> Dict[str, Any]:
        """Analyze test results against pass/fail criteria"""

        # Look for common pass/fail indicators
        pass_fail_columns = []
        for column in self.test_data.columns:
            column_lower = column.lower()
            if any(keyword in column_lower for keyword in ['pass', 'fail', 'result', 'status', 'accept', 'reject']):
                pass_fail_columns.append(column)

        analysis = {}

        if pass_fail_columns:
            for column in pass_fail_columns:
                value_counts = self.test_data[column].value_counts()

                # Try to identify pass/fail values
                pass_values = []
                fail_values = []

                for value in value_counts.index:
                    value_str = str(value).lower()
                    if any(keyword in value_str for keyword in ['pass', 'ok', 'accept', 'good']):
                        pass_values.append(value)
                    elif any(keyword in value_str for keyword in ['fail', 'reject', 'bad', 'nok']):
                        fail_values.append(value)

                total_tests = len(self.test_data)
                passed_tests = sum(value_counts[val] for val in pass_values) if pass_values else 0
                failed_tests = sum(value_counts[val] for val in fail_values) if fail_values else 0

                analysis[column] = {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'pass_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                    'fail_rate': (failed_tests / total_tests) * 100 if total_tests > 0 else 0,
                    'value_distribution': value_counts.to_dict()
                }

        return analysis

    def generate_visualizations(self) -> List[plt.Figure]:
        """Generate comprehensive visualizations for the test data"""

        if self.test_data is None:
            raise ValueError("No test data loaded.")

        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

        figures = []
        numerical_columns = self.test_data.select_dtypes(include=[np.number]).columns

        # 1. Data Distribution Plots
        if len(numerical_columns) > 0:
            n_cols = min(3, len(numerical_columns))
            n_rows = (len(numerical_columns) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, column in enumerate(numerical_columns[:9]):  # Limit to 9 plots
                if i < len(axes):
                    self.test_data[column].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {column}')
                    axes[i].set_xlabel(column)
                    axes[i].set_ylabel('Frequency')

            # Hide empty subplots
            for i in range(len(numerical_columns), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            figures.append(fig)

        # 2. Correlation Matrix
        if len(numerical_columns) > 1:
            correlation_data = self.test_data[numerical_columns].corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(correlation_data, cmap='coolwarm', aspect='auto')

            # Add labels
            ax.set_xticks(range(len(correlation_data.columns)))
            ax.set_yticks(range(len(correlation_data.columns)))
            ax.set_xticklabels(correlation_data.columns, rotation=45, ha='right')
            ax.set_yticklabels(correlation_data.columns)

            # Add values to cells
            for i in range(len(correlation_data.columns)):
                for j in range(len(correlation_data.columns)):
                    text = ax.text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                                 ha='center', va='center', color='black', fontsize=10)

            plt.colorbar(im)
            plt.title('Correlation Matrix of Test Parameters')
            plt.tight_layout()
            figures.append(fig)

        # 3. Box Plots for Outlier Detection
        if len(numerical_columns) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Select up to 6 columns for box plot
            selected_columns = numerical_columns[:6]
            box_data = [self.test_data[col].dropna() for col in selected_columns]

            ax.boxplot(box_data, labels=selected_columns)
            ax.set_title('Box Plots - Outlier Detection')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            figures.append(fig)

        # 4. Time Series Plot (if time column exists)
        time_columns = [col for col in self.test_data.columns 
                       if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp'])]

        if time_columns and len(numerical_columns) > 0:
            time_col = time_columns[0]

            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot first numerical column against time
            first_numerical = numerical_columns[0]
            ax.plot(self.test_data[time_col], self.test_data[first_numerical], 
                   marker='o', markersize=4, linewidth=1)
            ax.set_xlabel(time_col)
            ax.set_ylabel(first_numerical)
            ax.set_title(f'{first_numerical} vs {time_col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            figures.append(fig)

        self.figures = figures
        return figures

    def generate_html_report(self, output_path: str = None) -> str:
        """Generate comprehensive HTML report"""

        if self.analysis_results is None:
            self.analyze_test_data()

        html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>{{ config.report_title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { border-bottom: 3px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        .metric-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .metric-table th, .metric-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .metric-table th { background-color: #f2f2f2; font-weight: bold; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .warning { color: orange; font-weight: bold; }
        .summary-card { background: #f9f9f9; padding: 15px; border-left: 4px solid #007cba; margin: 10px 0; }
        .chart-placeholder { width: 100%; height: 400px; border: 1px solid #ddd; 
                           background: #f5f5f5; display: flex; align-items: center; 
                           justify-content: center; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ config.report_title }}</h1>
        <p><strong>Company:</strong> {{ config.company_name }}</p>
        <p><strong>Test Standard:</strong> {{ config.test_standard }}</p>
        <p><strong>Report Generated:</strong> {{ timestamp }}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-card">
            <h3>Test Overview</h3>
            <p><strong>Total Test Records:</strong> {{ quality.total_records }}</p>
            <p><strong>Data Completeness:</strong> {{ "%.1f%%"|format(quality.data_completeness * 100) }}</p>
            <p><strong>Quality Rating:</strong> <span class="{{ quality.quality_rating.lower() }}">{{ quality.quality_rating }}</span></p>
        </div>
    </div>

    <div class="section">
        <h2>Statistical Analysis Summary</h2>
        <table class="metric-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Outliers</th>
                </tr>
            </thead>
            <tbody>
            {% for param, stats in statistical_analysis.items() %}
                <tr>
                    <td>{{ param }}</td>
                    <td>{{ "%.3f"|format(stats.mean) }}</td>
                    <td>{{ "%.3f"|format(stats.std) }}</td>
                    <td>{{ "%.3f"|format(stats.min) }}</td>
                    <td>{{ "%.3f"|format(stats.max) }}</td>
                    <td>{{ stats.outliers_count }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>

    {% if pass_fail_analysis %}
    <div class="section">
        <h2>Pass/Fail Analysis</h2>
        {% for column, analysis in pass_fail_analysis.items() %}
        <div class="summary-card">
            <h3>{{ column }}</h3>
            <p><strong>Total Tests:</strong> {{ analysis.total_tests }}</p>
            <p><strong>Passed:</strong> <span class="pass">{{ analysis.passed_tests }} ({{ "%.1f%%"|format(analysis.pass_rate) }})</span></p>
            <p><strong>Failed:</strong> <span class="fail">{{ analysis.failed_tests }} ({{ "%.1f%%"|format(analysis.fail_rate) }})</span></p>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="section">
        <h2>Visualizations</h2>
        <div class="chart-placeholder">
            <p>Charts would be embedded here in production version</p>
        </div>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            {% if quality.data_completeness < 0.9 %}
            <li class="warning">Improve data collection procedures to reduce missing data points</li>
            {% endif %}

            {% for param, stats in statistical_analysis.items() %}
                {% if stats.outliers_count > 5 %}
                <li class="warning">Investigate outliers in {{ param }} measurements</li>
                {% endif %}
            {% endfor %}

            <li>Continue monitoring test parameters within specified ranges</li>
            <li>Regular calibration of measurement equipment recommended</li>
        </ul>
    </div>

    <div class="section">
        <h2>Technical Details</h2>
        <h3>Test Parameters Identified:</h3>
        <ul>
        {% for category, params in test_parameters.items() %}
            <li><strong>{{ category.title() }}:</strong> {{ params|join(', ') }}</li>
        {% endfor %}
        </ul>
    </div>

    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc; font-size: 12px; color: #666;">
        <p>This report was automatically generated by the Engineering Test Report Generator.</p>
        <p>Report ID: {{ analysis_results.analysis_timestamp }}</p>
    </footer>
</body>
</html>
        '''

        from jinja2 import Template
        template = Template(html_template)

        html_content = template.render(
            config=self.config,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            quality=self.analysis_results['quality_assessment'],
            statistical_analysis=self.analysis_results['statistical_analysis'],
            pass_fail_analysis=self.analysis_results['pass_fail_analysis'],
            test_parameters=self.analysis_results['test_parameters'],
            analysis_results=self.analysis_results
        )

        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_content)
            print(f"HTML report saved to: {output_path}")

        return html_content

    def generate_pdf_report(self, output_path: str) -> None:
        """Generate PDF report using ReportLab"""

        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors

        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph(self.config['report_title'], title_style))
        story.append(Spacer(1, 0.5*inch))

        # Report info
        info_data = [
            ['Company:', self.config['company_name']],
            ['Test Standard:', self.config['test_standard']], 
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Records:', str(self.analysis_results['quality_assessment']['total_records'])]
        ]

        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))

        story.append(info_table)
        story.append(Spacer(1, 0.3*inch))

        # Statistical Analysis Table
        story.append(Paragraph('Statistical Analysis Summary', styles['Heading2']))

        stats_data = [['Parameter', 'Mean', 'Std Dev', 'Min', 'Max', 'Outliers']]
        for param, stats in self.analysis_results['statistical_analysis'].items():
            stats_data.append([
                param,
                f"{stats['mean']:.3f}",
                f"{stats['std']:.3f}", 
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                str(stats['outliers_count'])
            ])

        stats_table = Table(stats_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch, 0.8*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(stats_table)
        story.append(Spacer(1, 0.3*inch))

        # Pass/Fail Analysis
        if self.analysis_results['pass_fail_analysis']:
            story.append(Paragraph('Pass/Fail Analysis', styles['Heading2']))

            for column, analysis in self.analysis_results['pass_fail_analysis'].items():
                pass_rate = analysis['pass_rate']
                fail_rate = analysis['fail_rate']

                pf_data = [
                    ['Test Type', column],
                    ['Total Tests', str(analysis['total_tests'])],
                    ['Passed Tests', f"{analysis['passed_tests']} ({pass_rate:.1f}%)"],
                    ['Failed Tests', f"{analysis['failed_tests']} ({fail_rate:.1f}%)"]
                ]

                pf_table = Table(pf_data, colWidths=[2*inch, 3*inch])
                pf_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                ]))

                story.append(pf_table)
                story.append(Spacer(1, 0.2*inch))

        # Build PDF
        doc.build(story)
        print(f"PDF report saved to: {output_path}")

    def generate_excel_report(self, output_path: str) -> None:
        """Generate comprehensive Excel report with multiple sheets"""

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Raw data sheet
            if self.test_data is not None:
                self.test_data.to_excel(writer, sheet_name='Raw_Data', index=False)

            # Statistical analysis sheet
            if self.analysis_results and 'statistical_analysis' in self.analysis_results:
                stats_df = pd.DataFrame(self.analysis_results['statistical_analysis']).T
                stats_df.to_excel(writer, sheet_name='Statistical_Analysis')

            # Basic statistics sheet
            if self.analysis_results and 'basic_statistics' in self.analysis_results:
                self.analysis_results['basic_statistics'].to_excel(writer, sheet_name='Basic_Statistics')

            # Summary sheet
            summary_data = {
                'Metric': ['Total Records', 'Data Completeness', 'Quality Rating'],
                'Value': [
                    self.analysis_results['quality_assessment']['total_records'],
                    f"{self.analysis_results['quality_assessment']['data_completeness']:.1%}",
                    self.analysis_results['quality_assessment']['quality_rating']
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        print(f"Excel report saved to: {output_path}")

# Utility function for generating sample test data
def generate_sample_test_data(num_samples: int = 100) -> pd.DataFrame:
    """Generate realistic sample test data for demonstration"""

    np.random.seed(42)

    # Simulate tensile test data
    data = {
        'specimen_id': [f'SPEC_{i+1:03d}' for i in range(num_samples)],
        'test_date': pd.date_range('2023-01-01', periods=num_samples, freq='D'),
        'material_type': np.random.choice(['Steel_A36', 'Al_6061', 'Ti_6Al4V'], num_samples, p=[0.5, 0.3, 0.2]),
        'tensile_strength_mpa': np.random.normal(400, 50, num_samples),
        'yield_strength_mpa': np.random.normal(250, 30, num_samples),
        'elongation_percent': np.random.normal(20, 5, num_samples),
        'elastic_modulus_gpa': np.random.normal(200, 20, num_samples),
        'test_temperature_c': np.random.normal(23, 2, num_samples),
        'crosshead_speed_mm_min': np.random.choice([1.0, 2.0, 5.0], num_samples),
        'specimen_width_mm': np.random.normal(12.5, 0.1, num_samples),
        'specimen_thickness_mm': np.random.normal(3.0, 0.05, num_samples)
    }

    df = pd.DataFrame(data)

    # Add pass/fail criteria based on material standards
    def determine_pass_fail(row):
        material = row['material_type']
        tensile = row['tensile_strength_mpa']
        elongation = row['elongation_percent']

        if material == 'Steel_A36':
            return 'PASS' if tensile >= 380 and elongation >= 15 else 'FAIL'
        elif material == 'Al_6061':
            return 'PASS' if tensile >= 290 and elongation >= 8 else 'FAIL'
        else:  # Ti_6Al4V
            return 'PASS' if tensile >= 900 and elongation >= 6 else 'FAIL'

    df['test_result'] = df.apply(determine_pass_fail, axis=1)

    # Add some realistic measurement noise and outliers
    df['tensile_strength_mpa'] = np.maximum(0, df['tensile_strength_mpa'] + np.random.normal(0, 5, num_samples))

    # Add a few outliers
    outlier_indices = np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False)
    df.loc[outlier_indices, 'tensile_strength_mpa'] *= np.random.uniform(1.2, 1.5, len(outlier_indices))

    return df
