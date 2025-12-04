import PropTypes from 'prop-types';
import './GestureDisplay.css';

function GestureDisplay({ gesture }) {
  if (!gesture || gesture.error) {
    return (
      <div className="gesture-display empty">
        <p>{gesture?.message || 'No se detectó gesto'}</p>
      </div>
    );
  }

  const { 
    gesture: gestureName, 
    confidence, 
    emoji, 
    display_name, 
    top_3 
  } = gesture;

  // Confianza en porcentaje
  const confidencePercent = (confidence * 100).toFixed(1);
  
  // Determinar color según confianza
  const getConfidenceColor = (conf) => {
    if (conf > 0.8) return 'high';
    if (conf > 0.5) return 'medium';
    return 'low';
  };

  return (
    <div className="gesture-display">
      <div className="gesture-main">
        <div className="gesture-emoji">{emoji}</div>
        <div className="gesture-info">
          <h3 className="gesture-name">{display_name}</h3>
          <div className={`gesture-confidence ${getConfidenceColor(confidence)}`}>
            <div className="confidence-bar">
              <div 
                className="confidence-fill" 
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
            <span className="confidence-text">{confidencePercent}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

GestureDisplay.propTypes = {
  gesture: PropTypes.shape({
    gesture: PropTypes.string,
    confidence: PropTypes.number,
    emoji: PropTypes.string,
    display_name: PropTypes.string,
    error: PropTypes.string,
    message: PropTypes.string,
    top_3: PropTypes.arrayOf(PropTypes.shape({
      gesture: PropTypes.string,
      confidence: PropTypes.number,
      emoji: PropTypes.string,
      display_name: PropTypes.string
    }))
  })
};

export default GestureDisplay;
