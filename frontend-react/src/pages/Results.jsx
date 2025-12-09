import { useLocation, Link } from 'react-router-dom';
import EmotionDisplay from '../components/EmotionDisplay';
import './Results.css';

function Results() {
  const location = useLocation();
  
  // Recibir logs separados por sistema
  const { 
    faceRecognitionLogs = [], 
    voiceCommandLogs = [],
    emotions = null 
  } = location.state || {};

  // Estadísticas para Reconocimiento Facial
  const getFaceStats = () => {
    return {
      total: faceRecognitionLogs.length,
      success: faceRecognitionLogs.filter(l => l.type === 'success').length,
      error: faceRecognitionLogs.filter(l => l.type === 'error').length,
      info: faceRecognitionLogs.filter(l => l.type === 'info').length,
      warning: faceRecognitionLogs.filter(l => l.type === 'warning').length
    };
  };

  // Estadísticas para Comandos de Voz
  const getVoiceStats = () => {
    return {
      total: voiceCommandLogs.length,
      success: voiceCommandLogs.filter(l => l.type === 'success').length,
      error: voiceCommandLogs.filter(l => l.type === 'error').length,
      info: voiceCommandLogs.filter(l => l.type === 'info').length,
      warning: voiceCommandLogs.filter(l => l.type === 'warning').length
    };
  };

  const faceStats = getFaceStats();
  const voiceStats = getVoiceStats();

  return (
    <div className="results-page">
      <div className="results-header">
        <h1>Resultados de la Sesión</h1>
        <p>Análisis detallado de la captura realizada</p>
      </div>

      <div className="stats-grid">
        {/* Estadísticas de Reconocimiento Facial */}
        <div className="stat-card facial-section">
          <div className="stat-icon">�</div>
          <div className="stat-content">
            <h3>Reconocimiento Facial</h3>
            <div className="stat-details">
              <p className="stat-value">Total: {faceStats.total}</p>
              <p className="stat-mini success">{faceStats.success}</p>
              <p className="stat-mini error">{faceStats.error}</p>
            </div>
          </div>
        </div>

        {/* Estadísticas de Comandos de Voz */}
        <div className="stat-card voice-section">
          <div className="stat-icon"></div>
          <div className="stat-content">
            <h3>Comandos de Voz</h3>
            <div className="stat-details">
              <p className="stat-value">Total: {voiceStats.total}</p>
              <p className="stat-mini success">{voiceStats.success}</p>
              <p className="stat-mini error">{voiceStats.error}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="results-content">
        <div className="emotions-result">
          <h2>Última Emoción Detectada</h2>
          <EmotionDisplay emotions={emotions} />
        </div>

        {/* Registros Separados por Sistema */}
        <div className="logs-container-dual">
          {/* Log de Reconocimiento Facial */}
          <div className="logs-result facial-logs">
            <h2>Registro de Reconocimiento Facial</h2>
            <div className="logs-list">
              {faceRecognitionLogs.length === 0 ? (
                <div className="empty-state">
                  <p>No hay actividad de reconocimiento facial</p>
                </div>
              ) : (
                faceRecognitionLogs.map(log => (
                  <div key={log.id} className={`log-item ${log.type}`}>
                    <span className="log-timestamp">{log.timestamp}</span>
                    <span className="log-type">{log.type.toUpperCase()}</span>
                    <span className="log-message">{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Log de Comandos de Voz */}
          <div className="logs-result voice-logs">
            <h2>Registro de Comandos de Voz</h2>
            <div className="logs-list">
              {voiceCommandLogs.length === 0 ? (
                <div className="empty-state">
                  <p>No hay actividad de comandos de voz</p>
                </div>
              ) : (
                voiceCommandLogs.map(log => (
                  <div key={log.id} className={`log-item ${log.type}`}>
                    <span className="log-timestamp">{log.timestamp}</span>
                    <span className="log-type">{log.type.toUpperCase()}</span>
                    <span className="log-message">{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="actions">
        <Link to="/capture" className="btn btn-primary">
          Nueva Captura
        </Link>
        <Link to="/" className="btn btn-secondary">
          Volver al Inicio
        </Link>
      </div>
    </div>
  );
}

export default Results;
