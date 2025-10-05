import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional
from ultralytics import YOLO
import logging
from pathlib import Path

class DefectDetector:
    """
    Advanced defect detection system using computer vision and deep learning
    """

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = [
            'crack', 'scratch', 'dent', 'corrosion', 'discoloration',
            'missing_component', 'wrong_component', 'dimension_error'
        ]

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load pre-trained defect detection model"""

        try:
            if model_path.endswith('.pt'):
                # YOLO model
                self.model = YOLO(model_path)
                self.logger.info(f"YOLO model loaded from {model_path}")
            else:
                # TensorFlow/Keras model
                self.model = tf.keras.models.load_model(model_path)
                self.logger.info(f"Keras model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            # Initialize with default CNN model
            self.model = self._create_default_model()

    def _create_default_model(self) -> tf.keras.Model:
        """Create a default CNN model for defect detection"""

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(len(self.class_names), activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""

        # Resize to model input size
        image_resized = cv2.resize(image, (224, 224))

        # Normalize pixel values
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)

        return image_batch

    def detect_defects(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect defects in the given image

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary containing detection results
        """

        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")

        # Preprocess image
        processed_image = self.preprocess_image(image)

        try:
            if isinstance(self.model, YOLO):
                # YOLO detection
                results = self.model(image, conf=self.confidence_threshold)
                detections = self._parse_yolo_results(results[0])
            else:
                # CNN classification
                predictions = self.model.predict(processed_image)
                detections = self._parse_cnn_results(predictions[0], image)

            return {
                'defects_found': len(detections['defects']) > 0,
                'defect_count': len(detections['defects']),
                'defects': detections['defects'],
                'overall_quality': detections['overall_quality'],
                'confidence_scores': detections['confidence_scores'],
                'processing_time': detections.get('processing_time', 0)
            }

        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            return {'error': str(e)}

    def _parse_yolo_results(self, results) -> Dict[str, Any]:
        """Parse YOLO detection results"""

        defects = []
        confidence_scores = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = box
                    defect_info = {
                        'type': self.class_names[int(cls)],
                        'confidence': float(conf),
                        'bounding_box': {
                            'x1': int(x1), 'y1': int(y1),
                            'x2': int(x2), 'y2': int(y2)
                        },
                        'area': (x2 - x1) * (y2 - y1),
                        'severity': self._assess_defect_severity(self.class_names[int(cls)], conf)
                    }
                    defects.append(defect_info)
                    confidence_scores.append(float(conf))

        # Calculate overall quality score
        overall_quality = self._calculate_overall_quality(defects)

        return {
            'defects': defects,
            'overall_quality': overall_quality,
            'confidence_scores': confidence_scores
        }

    def _parse_cnn_results(self, predictions: np.ndarray, image: np.ndarray) -> Dict[str, Any]:
        """Parse CNN classification results"""

        defects = []
        confidence_scores = predictions.tolist()

        for i, (class_name, score) in enumerate(zip(self.class_names, predictions)):
            if score >= self.confidence_threshold:
                # For CNN, we don't have bounding boxes, so use whole image
                h, w = image.shape[:2]
                defect_info = {
                    'type': class_name,
                    'confidence': float(score),
                    'bounding_box': {
                        'x1': 0, 'y1': 0, 'x2': w, 'y2': h
                    },
                    'area': w * h,
                    'severity': self._assess_defect_severity(class_name, score)
                }
                defects.append(defect_info)

        overall_quality = self._calculate_overall_quality(defects)

        return {
            'defects': defects,
            'overall_quality': overall_quality,
            'confidence_scores': confidence_scores
        }

    def _assess_defect_severity(self, defect_type: str, confidence: float) -> str:
        """Assess defect severity based on type and confidence"""

        # Critical defects that affect functionality
        critical_defects = ['crack', 'missing_component', 'wrong_component']

        # Major defects that affect appearance/performance
        major_defects = ['dent', 'corrosion', 'dimension_error']

        # Minor defects that are cosmetic
        minor_defects = ['scratch', 'discoloration']

        if defect_type in critical_defects:
            if confidence > 0.8:
                return 'CRITICAL'
            else:
                return 'MAJOR'
        elif defect_type in major_defects:
            if confidence > 0.7:
                return 'MAJOR'
            else:
                return 'MINOR'
        else:
            return 'MINOR'

    def _calculate_overall_quality(self, defects: List[Dict]) -> Dict[str, Any]:
        """Calculate overall quality assessment"""

        if not defects:
            return {
                'score': 100,
                'grade': 'A',
                'status': 'PASS'
            }

        # Calculate quality score based on defect severity
        severity_weights = {'CRITICAL': 40, 'MAJOR': 20, 'MINOR': 5}
        total_penalty = sum(severity_weights.get(d['severity'], 0) for d in defects)

        quality_score = max(0, 100 - total_penalty)

        # Determine grade and status
        if quality_score >= 95:
            grade, status = 'A', 'PASS'
        elif quality_score >= 85:
            grade, status = 'B', 'PASS'
        elif quality_score >= 75:
            grade, status = 'C', 'CONDITIONAL'
        else:
            grade, status = 'D', 'FAIL'

        return {
            'score': quality_score,
            'grade': grade,
            'status': status
        }

    def detect_defects_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch

        Args:
            images: List of images as numpy arrays

        Returns:
            List of detection results for each image
        """

        results = []
        for i, image in enumerate(images):
            try:
                result = self.detect_defects(image)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process image {i}: {str(e)}")
                results.append({'image_index': i, 'error': str(e)})

        return results

    def visualize_detections(self, image: np.ndarray, detection_results: Dict[str, Any]) -> np.ndarray:
        """
        Visualize detection results on the image

        Args:
            image: Original image
            detection_results: Results from detect_defects()

        Returns:
            Image with visualized detections
        """

        viz_image = image.copy()

        if 'defects' not in detection_results:
            return viz_image

        # Define colors for different severities
        severity_colors = {
            'CRITICAL': (0, 0, 255),    # Red
            'MAJOR': (0, 165, 255),     # Orange
            'MINOR': (0, 255, 255)      # Yellow
        }

        for defect in detection_results['defects']:
            bbox = defect['bounding_box']
            severity = defect['severity']
            defect_type = defect['type']
            confidence = defect['confidence']

            color = severity_colors.get(severity, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(
                viz_image,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                color, 2
            )

            # Add label
            label = f"{defect_type}: {confidence:.2f} ({severity})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Background for text
            cv2.rectangle(
                viz_image,
                (bbox['x1'], bbox['y1'] - label_size[1] - 10),
                (bbox['x1'] + label_size[0], bbox['y1']),
                color, -1
            )

            # Text
            cv2.putText(
                viz_image, label,
                (bbox['x1'], bbox['y1'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2
            )

        # Add overall quality score
        quality = detection_results.get('overall_quality', {})
        quality_text = f"Quality: {quality.get('score', 0):.1f} ({quality.get('grade', 'N/A')})"

        cv2.putText(
            viz_image, quality_text,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (255, 255, 255), 2
        )

        return viz_image

    def generate_inspection_report(self, detection_results: Dict[str, Any], 
                                 part_id: str = None) -> Dict[str, Any]:
        """
        Generate detailed inspection report

        Args:
            detection_results: Results from detect_defects()
            part_id: Optional part identifier

        Returns:
            Comprehensive inspection report
        """

        from datetime import datetime

        report = {
            'inspection_id': f"INSP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'part_id': part_id or 'UNKNOWN',
            'timestamp': datetime.now().isoformat(),
            'inspection_summary': {
                'total_defects': detection_results.get('defect_count', 0),
                'defects_found': detection_results.get('defects_found', False),
                'overall_quality': detection_results.get('overall_quality', {}),
                'processing_time_ms': detection_results.get('processing_time', 0)
            },
            'detailed_findings': [],
            'recommendations': [],
            'next_actions': []
        }

        # Process individual defects
        for i, defect in enumerate(detection_results.get('defects', []), 1):
            finding = {
                'defect_id': i,
                'type': defect['type'],
                'severity': defect['severity'],
                'confidence': defect['confidence'],
                'location': defect['bounding_box'],
                'area_affected': defect['area'],
                'description': self._get_defect_description(defect['type'])
            }
            report['detailed_findings'].append(finding)

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(detection_results)

        # Determine next actions
        quality_status = detection_results.get('overall_quality', {}).get('status', 'UNKNOWN')
        if quality_status == 'FAIL':
            report['next_actions'] = ['Reject part', 'Investigate root cause', 'Review process parameters']
        elif quality_status == 'CONDITIONAL':
            report['next_actions'] = ['Secondary inspection required', 'Engineering review', 'Customer approval needed']
        else:
            report['next_actions'] = ['Accept part', 'Continue production', 'Regular monitoring']

        return report

    def _get_defect_description(self, defect_type: str) -> str:
        """Get detailed description for defect type"""

        descriptions = {
            'crack': 'Linear discontinuity in material that may propagate under stress',
            'scratch': 'Surface mark or groove that affects appearance',
            'dent': 'Local deformation causing surface irregularity',
            'corrosion': 'Material degradation due to chemical reaction',
            'discoloration': 'Change in surface color from specified appearance',
            'missing_component': 'Required component is not present in assembly',
            'wrong_component': 'Incorrect component installed in assembly',
            'dimension_error': 'Measured dimension outside specified tolerance'
        }

        return descriptions.get(defect_type, 'Unknown defect type')

    def _generate_recommendations(self, detection_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on detection results"""

        recommendations = []
        defects = detection_results.get('defects', [])

        if not defects:
            recommendations.append('Part meets quality standards - continue production')
            return recommendations

        # Group defects by type
        defect_types = {}
        for defect in defects:
            defect_type = defect['type']
            if defect_type not in defect_types:
                defect_types[defect_type] = []
            defect_types[defect_type].append(defect)

        # Generate specific recommendations
        if 'crack' in defect_types:
            recommendations.append('Investigate material stress and fatigue factors')

        if 'scratch' in defect_types:
            recommendations.append('Review handling procedures and protective measures')

        if 'dimension_error' in defect_types:
            recommendations.append('Calibrate measurement equipment and review machining parameters')

        if 'corrosion' in defect_types:
            recommendations.append('Review environmental controls and surface treatments')

        return recommendations

# Utility functions for testing and demonstration
def create_sample_defect_images(num_images: int = 10) -> List[np.ndarray]:
    """Create sample images with simulated defects for testing"""

    images = []

    for i in range(num_images):
        # Create base image (simulated part)
        image = np.ones((224, 224, 3), dtype=np.uint8) * 200  # Light gray background

        # Add some structure (simulated part features)
        cv2.rectangle(image, (50, 50), (174, 174), (150, 150, 150), -1)
        cv2.circle(image, (112, 112), 30, (100, 100, 100), -1)

        # Randomly add defects
        if np.random.random() > 0.5:  # 50% chance of defect
            defect_type = np.random.choice(['crack', 'scratch', 'dent'])

            if defect_type == 'crack':
                # Add crack-like line
                cv2.line(image, (80, 80), (144, 144), (50, 50, 50), 2)
            elif defect_type == 'scratch':
                # Add scratch marks
                for j in range(3):
                    start_x = np.random.randint(60, 120)
                    start_y = np.random.randint(60, 120)
                    end_x = start_x + np.random.randint(10, 40)
                    end_y = start_y + np.random.randint(-10, 10)
                    cv2.line(image, (start_x, start_y), (end_x, end_y), (80, 80, 80), 1)
            elif defect_type == 'dent':
                # Add circular dent
                center = (np.random.randint(70, 154), np.random.randint(70, 154))
                cv2.circle(image, center, 15, (120, 120, 120), -1)

        images.append(image)

    return images
