import PropTypes from 'prop-types';
import './EmotionDisplay.css';

function EmotionDisplay({ emotions }) {
  if (!emotions || Object.keys(emotions).length === 0) {
    return (
      <div className="emotion-display empty">
        <p>No se detectaron emociones</p>
      </div>
    );
  }

  // Normalizar valores: si vienen como porcentajes (>1), convertir a [0,1]
  const values = Object.values(emotions);
  const maxVal = Math.max(...values);
  const normalized = Object.fromEntries(
    Object.entries(emotions).map(([k, v]) => [k, maxVal > 1 ? (Number(v) / 100) : Number(v)])
  );

  // Obtener la emoción dominante
  const dominant = Object.entries(normalized).reduce((a, b) => 
    a[1] > b[1] ? a : b
  );

  // Ordenar emociones por valor
  const sortedEmotions = Object.entries(normalized)
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className="emotion-display">
      <div className="dominant-emotion">
        <h3>{dominant[0]}</h3>
        <p className="confidence">{(dominant[1] * 100).toFixed(1)}%</p>
      </div>

      <div className="emotion-bars">
        {sortedEmotions.map(([emotion, value]) => (
          <div key={emotion} className="emotion-bar">
            <div className="emotion-label">
              <span className="emotion-name">{emotion}</span>
            </div>
            <div className="bar-container">
              <div 
                className="bar-fill" 
                style={{ width: `${value * 100}%` }}
              />
            </div>
            <span className="value">{(value * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

EmotionDisplay.propTypes = {
  emotions: PropTypes.object
};

export default EmotionDisplay;
