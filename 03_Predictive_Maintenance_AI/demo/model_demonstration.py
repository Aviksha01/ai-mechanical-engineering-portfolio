# Sample data generation and model demonstration

import pandas as pd
import numpy as np
from models.failure_prediction_model import FailurePredictionModel, MaintenanceScheduleOptimizer, generate_sample_sensor_data

def main():
    print("=== Predictive Maintenance Model Demonstration ===")

    # Generate sample data
    print("\n1. Generating sample sensor data...")
    data = generate_sample_sensor_data(1000)
    print(f"Generated {len(data)} samples with {data.shape[1]} features")

    # Initialize and train model
    print("\n2. Training failure prediction model...")
    model = FailurePredictionModel(model_type='ensemble')

    X, y = model.prepare_data(data, 'failure_within_days')
    results = model.train_model(X, y)

    print(f"Model Accuracy: {results['accuracy']:.3f}")
    print(f"AUC Score: {results['auc_score']:.3f}")

    # Make predictions on sample equipment
    print("\n3. Making predictions for sample equipment...")

    # Simulate current sensor readings for different equipment
    equipment_data = {
        'Equipment_A': np.array([70, 3.2, 2.8, 2.1, 140, 42, 2500, 6000, 45]),
        'Equipment_B': np.array([62, 2.1, 1.9, 1.5, 155, 47, 2100, 3000, 15]),
        'Equipment_C': np.array([85, 4.5, 4.1, 3.8, 120, 35, 3200, 7500, 90])
    }

    predictions = {}
    for equipment_id, sensor_data in equipment_data.items():
        prediction = model.predict_failure_probability(sensor_data)
        predictions[equipment_id] = prediction

        print(f"{equipment_id}: {prediction['risk_level']} risk ({prediction['failure_probability']:.3f} probability)")

    # Optimize maintenance schedule
    print("\n4. Optimizing maintenance schedule...")
    optimizer = MaintenanceScheduleOptimizer()
    schedule = optimizer.optimize_schedule(predictions)

    print("Recommended Maintenance Schedule:")
    for item in schedule:
        print(f"- {item['equipment_id']}: {item['maintenance_type']} maintenance in {item['days_until_maintenance']} days")
        print(f"  Expected savings: ${item['expected_cost_savings']:.0f}")

    # Save model
    print("\n5. Saving trained model...")
    model.save_model('models/predictive_maintenance_model.pkl')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
