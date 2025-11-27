import { Link } from "react-router-dom";
import "./Home.css";

import DogClassifier from "../components/DogClassifier";

function Home() {
  return (
    <div className="home-container">
      <div className="hero">
        <div className="hero-content">
          <h1>JarvisTEC</h1>
          <p className="subtitle">
            Asistente inteligente con reconocimiento facial y procesamiento de
            voz impulsado por Machine Learning e Inteligencia Artificial
          </p>
          <div className="button-group">
            <Link to="/capture" className="btn btn-primary">
              Iniciar Captura
            </Link>
            <Link to="/results" className="btn btn-secondary">
              Ver Resultados
            </Link>
            <a
              href="#classifier"
              className="btn btn-secondary"
              style={{ marginLeft: "10px" }}
            >
              🐶 Identificar Perro
            </a>
          </div>
        </div>

        <div className="hero-image">
          <img
            src="https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800&q=80"
            alt="AI Robot Assistant"
          />
        </div>
      </div>

      <div className="features">
        <div className="feature-card">
          <h2>📸 Reconocimiento Facial</h2>
          <p>
            Detecta emociones en tiempo real usando tecnología avanzada de
            análisis facial con IA
          </p>
        </div>

        <div className="feature-card">
          <h2>🎤 Procesamiento de Voz</h2>
          <p>
            Transcripción de audio y procesamiento de comandos mediante
            algoritmos de reconocimiento de voz
          </p>
        </div>

        <div className="feature-card">
          <h2>🧠 Modelos ML</h2>
          <p>
            Predicciones precisas basadas en 10 modelos de machine learning
            entrenados con datos reales
          </p>
        </div>
      </div>

      {/* Nueva sección para el clasificador de perros */}
      <div className="classifier-section" id="classifier">
        <DogClassifier />
      </div>

      <div className="cta-section">
        <h2>Experimenta el Futuro de la IA</h2>
        <p className="subtitle">
          Interactúa con JarvisTEC usando comandos de voz y descubre el poder
          del machine learning
        </p>
      </div>

      <div className="info-section">
        <h3>¿Cómo funciona?</h3>
        <ol>
          <li>Inicia la captura de cámara y micrófono desde la interfaz</li>
          <li>
            El sistema analiza las emociones faciales en tiempo real usando IA
          </li>
          <li>
            Da comandos de voz para interactuar con el asistente inteligente
          </li>
          <li>Visualiza los resultados y predicciones de los modelos ML</li>
        </ol>
      </div>
    </div>
  );
}

export default Home;
