from pathlib import Path
import joblib

MODEL = Path(__file__).parent.parent / 'models' / 'bmi_model.joblib'


def main():
    if not MODEL.exists():
        print('BMI model not found:', MODEL)
        return

    model = joblib.load(MODEL)
    # Example: height 1.75 m, weight 70 kg, age 30
    sample = [[1.75, 70, 30]]
    pred = model.predict(sample)
    print('Predicted BodyFat:', pred[0])


if __name__ == '__main__':
    main()
