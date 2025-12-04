import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import CameraCapture from '../components/CameraCapture';
import AudioRecorder from '../components/AudioRecorder';
import EmotionDisplay from '../components/EmotionDisplay';
import GestureDisplay from '../components/GestureDisplay';
import { apiClient } from '../services/api-client';
import './Capture.css';

function Capture() {
  // Estados separados para cada sistema
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isAudioActive, setIsAudioActive] = useState(false);
  const [currentEmotions, setCurrentEmotions] = useState(null);
  const [currentGesture, setCurrentGesture] = useState(null);
  
  // REGISTROS SEPARADOS PARA CADA SISTEMA
  const [faceRecognitionLogs, setFaceRecognitionLogs] = useState([]); // 📸 Logs de reconocimiento facial
  const [voiceCommandLogs, setVoiceCommandLogs] = useState([]); // 🎤 Logs de comandos de voz
  
  const [snapshotPreview, setSnapshotPreview] = useState(null);
  // Metadatos
  const [movieGenres, setMovieGenres] = useState([]);
  const [movieYears, setMovieYears] = useState([]);
  const [airOrigins, setAirOrigins] = useState([]);
  const [airDestinations, setAirDestinations] = useState([]);
  const [airCarriers, setAirCarriers] = useState([]);
  // Prompt para completar parámetros faltantes
  const [pendingCommand, setPendingCommand] = useState(null);
  const [promptValues, setPromptValues] = useState({});
  const [isListeningForCommand, setIsListeningForCommand] = useState(false);
  const [lastResult, setLastResult] = useState(null);
  const navigate = useNavigate();

  // Cargar metadatos al montar
  useEffect(() => {
    const loadMeta = async () => {
      try {
        const movies = await apiClient.getMovieMeta();
        setMovieGenres(movies.genres || []);
        setMovieYears(movies.years || []);
        addVoiceLog('🎬 Metadatos de películas cargados', 'info');
      } catch (e) {
        addVoiceLog('⚠️ No se pudieron cargar géneros/años de películas', 'warning');
      }
      try {
        const air = await apiClient.getAirlineMeta();
        setAirOrigins(air.origins || []);
        setAirDestinations(air.destinations || []);
        setAirCarriers(air.carriers || []);
        addVoiceLog('🛫 Metadatos de aerolíneas cargados', 'info');
      } catch (e) {
        addVoiceLog('⚠️ No se pudieron cargar orígenes/destinos/aerolíneas', 'warning');
      }
    };
    loadMeta();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Función para agregar log al sistema de RECONOCIMIENTO FACIAL
  const addFaceLog = (message, type = 'info') => {
    const newLog = {
      id: Date.now(),
      message,
      type,
      timestamp: new Date().toLocaleTimeString(),
      system: 'face'
    };
    setFaceRecognitionLogs(prev => [newLog, ...prev].slice(0, 20));
  };

  // Función para agregar log al sistema de COMANDOS DE VOZ
  const addVoiceLog = (message, type = 'info') => {
    const newLog = {
      id: Date.now() + 1, // +1 para evitar IDs duplicados
      message,
      type,
      timestamp: new Date().toLocaleTimeString(),
      system: 'voice'
    };
    setVoiceCommandLogs(prev => [newLog, ...prev].slice(0, 20));
  };

  // Controles para el sistema de CÁMARA (emociones + gestos)
  const handleStartCamera = () => {
    setIsCameraActive(true);
    addFaceLog('📸 Sistema de reconocimiento facial y gestos INICIADO', 'success');
  };

  const handleStopCamera = () => {
    setIsCameraActive(false);
    setCurrentEmotions(null);
    setCurrentGesture(null);
    addFaceLog('📸 Sistema de reconocimiento facial y gestos DETENIDO', 'info');
  };

  // Controles para el sistema de AUDIO (comandos de voz)
  const handleStartAudio = () => {
    setIsAudioActive(true);
    addVoiceLog('🎤 Sistema de comandos de voz INICIADO', 'success');
  };

  const handleStopAudio = () => {
    setIsAudioActive(false);
    addVoiceLog('🎤 Sistema de comandos de voz DETENIDO', 'info');
  };

  const handleSnapshot = (imageBlob) => {
    const url = URL.createObjectURL(imageBlob);
    setSnapshotPreview(url);
    addFaceLog('📷 Foto capturada correctamente', 'success');
    
    // Limpiar URL después de 30 segundos
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  };

  const handleEmotionDetected = async (imageBlob) => {
    try {
      // Procesar emociones y gestos en paralelo
      const [faceData, gestureData] = await Promise.all([
        apiClient.recognizeFace(imageBlob),
        apiClient.classifyGesture(imageBlob)
      ]);
      
      // === PROCESAR EMOCIONES ===
      const details = faceData?.details || [];
      const faceCount = faceData?.face_count ?? details.length;

      if (!details.length || faceCount === 0) {
        setCurrentEmotions(null);
        addFaceLog(faceData?.message || 'No se detectó rostro en la imagen', 'warning');
      } else {
        // Elegir rostro dominante: el que coincide con dominant_emotion o el primero
        let selected = details.find(d => d.top_emotion === faceData?.dominant_emotion) || details[0];
        let scores = selected?.scores || faceData?.attributes?.emotion || null;

        if (!scores || typeof scores !== 'object') {
          addFaceLog('No se pudieron obtener las emociones del rostro', 'warning');
          setCurrentEmotions(null);
        } else {
          // Normalizar a [0,1] si vienen en porcentaje
          const values = Object.values(scores);
          const maxVal = Math.max(...values);
          const normalized = Object.fromEntries(
            Object.entries(scores).map(([k, v]) => [k, maxVal > 1 ? (Number(v) / 100) : Number(v)])
          );

          // Ordenar para log: top primero, luego secundarias
          const ordered = Object.entries(normalized).sort((a, b) => b[1] - a[1]);
          const top = ordered[0];
          const secondary = ordered.slice(1).map(([k, v]) => `${k} ${(v * 100).toFixed(1)}%`).join(', ');

          setCurrentEmotions(normalized);
          addFaceLog(`😊 Emoción: ${top[0]} ${(top[1] * 100).toFixed(1)}% | Otras: ${secondary}`, 'success');
        }
      }

      // === PROCESAR GESTOS ===
      if (gestureData?.error) {
        setCurrentGesture(null);
        addFaceLog(`⚠️ Gesto: ${gestureData.message || 'No disponible'}`, 'warning');
      } else {
        setCurrentGesture(gestureData);
        const gestureInfo = `${gestureData.emoji} ${gestureData.display_name}`;
        const conf = (gestureData.confidence * 100).toFixed(1);
        addFaceLog(`👋 Gesto detectado: ${gestureInfo} (${conf}%)`, 'success');
      }
      
    } catch (error) {
      console.error('Error procesando imagen:', error);
      addFaceLog(`❌ Error: ${error.message}`, 'error');
    }
  };

  const handleTranscription = async (audioOrText) => {
    // Prevent processing if a form is already open
    if (pendingCommand) {
      return;
    }

    try {
      let text = "";
      if (typeof audioOrText === 'string') {
        text = audioOrText;
      } else {
        // addVoiceLog("🎤 Procesando audio...", "info"); // Optional: too noisy?
        const res = await apiClient.transcribeAudio(audioOrText);
        text = res.transcription || res;
      }
      
      if (!text) {
        console.log("🎤 Audio procesado pero sin texto detectado.");
        return;
      }
      
      const lower = text.toLowerCase();
      addVoiceLog(`🎤 Transcripción: "${text}"`, 'info');

      // 1. Global Commands
      if (lower.includes("exit")) {
        handleStopAudio();
        addVoiceLog("🛑 Comando EXIT detectado", "warning");
        return;
      }
      if (lower.includes("reset")) {
        cancelPrompt();
        setLastResult(null);
        addVoiceLog("🔄 Comando RESET detectado", "info");
        return;
      }

      // 2. Wake Word - REMOVED (Always listening)
      /*
      if (!isListeningForCommand) {
        if (lower.includes("travistec")) {
          setIsListeningForCommand(true);
          addVoiceLog("🤖 ¡Hola! Escuchando comando...", "success");
        }
        return;
      }
      */

      // 3. Command Parsing (Always active)
      if (true) {
        let task = null;
        
        if (lower.includes("airline")) task = "airline";
        else if (lower.includes("avocado") || lower.includes("aguacate")) task = "avocado";
        else if (lower.includes("london") && lower.includes("crime")) task = "london";
        else if (lower.includes("chicago") && lower.includes("crime")) task = "chicago";
        else if (lower.includes("movie") || lower.includes("pelicula")) task = "movie";
        else if (lower.includes("bitcoin")) task = "bitcoin";
        else if (lower.includes("car") || lower.includes("coche")) task = "car_price";
        else if (lower.includes("bmi") || lower.includes("masa")) task = "bmi";
        else if (lower.includes("sp500")) task = "sp500";
        else if (lower.includes("cirrhosis")) task = "cirrhosis";

        if (task) {
          // setIsListeningForCommand(false); // No longer needed
          addVoiceLog(`✅ Comando detectado: ${task}`, "success");
          
          // Trigger the form logic
          const commandObj = { task, params: { needs: ['all'] } }; 
          handleCommand(commandObj);
        }
      }

    } catch (error) {
      console.error('❌ Error transcribiendo:', error);
      addVoiceLog(`❌ Error transcribiendo: ${error.message}`, 'error');
    }
  };

  const handleCommand = async (command) => {
    // Prevent overwriting an active form with repeated/noise commands
    if (pendingCommand) {
      console.log('⚠️ Ignorando nuevo comando porque hay uno pendiente:', command.task);
      return;
    }

    try {
      console.log('🎯 handleCommand llamado con:', command);
      addVoiceLog(`🎯 Comando parseado: ${command.task || 'desconocido'}`, 'info');
      // Si el parser indicó que faltan parámetros, abrir prompt
      const needs = command?.params?.needs || [];
      if (needs.length) {
        setPendingCommand(command);
        const init = {};
        if (command.task === 'movie') {
          init.genre = command.params.genre || (movieGenres[0] || '');
          init.year = command.params.year || (movieYears[movieYears.length - 1] || '');
        } else if (command.task === 'airline') {
          init.origin = command.params.origin || (airOrigins[0] || '');
          init.dest = command.params.dest || (airDestinations[0] || '');
          init.carrier = command.params.carrier || (airCarriers[0] || '');
          init.month = command.params.month || new Date().getMonth() + 1;
          init.day = command.params.day || new Date().getDate();
          init.distance = command.params.distance || 500;
        } else if (command.task === 'london' || command.task === 'chicago') {
          init.month = command.params.month || new Date().getMonth() + 1;
          init.date = command.params.date || new Date().getDate();
          init.day = command.params.day || 'viernes';
        }
        setPromptValues(init);
        addVoiceLog('ℹ️ Falta información. Completa el formulario y confirma.', 'info');
        return;
      }

      const response = await apiClient.processCommand(command);
      console.log('✅ Respuesta del servidor:', response);
      addVoiceLog(`✅ ${response}`, 'success');
      setLastResult(response);
    } catch (error) {
      console.error('❌ Error procesando comando:', error);
      addVoiceLog(`❌ Error: ${error.message}`, 'error');
    }
  };

  const cancelPrompt = () => {
    setPendingCommand(null);
    setPromptValues({});
  };

  const confirmPrompt = async () => {
    if (!pendingCommand) return;
    const pc = JSON.parse(JSON.stringify(pendingCommand));
    if (pc.task === 'movie') {
      if (promptValues.genre) pc.params.genre = promptValues.genre;
      if (promptValues.year) pc.params.year = parseInt(promptValues.year, 10);
    } else if (pc.task === 'airline') {
      if (promptValues.origin) pc.params.origin = promptValues.origin;
      if (promptValues.dest) pc.params.dest = promptValues.dest;
      if (promptValues.carrier) pc.params.carrier = promptValues.carrier;
      if (promptValues.month) pc.params.month = parseInt(promptValues.month, 10);
      if (promptValues.day) pc.params.day = parseInt(promptValues.day, 10);
      if (promptValues.distance) pc.params.distance = parseInt(promptValues.distance, 10);
    } else if (pc.task === 'london' || pc.task === 'chicago') {
      if (promptValues.month) pc.params.month = parseInt(promptValues.month, 10);
      if (promptValues.date) pc.params.date = parseInt(promptValues.date, 10);
      if (promptValues.day) pc.params.day = promptValues.day;
    } else if (pc.task === 'avocado' || pc.task === 'bitcoin' || pc.task === 'sp500') {
      if (promptValues.days) pc.params.days = parseInt(promptValues.days, 10);
    } else if (pc.task === 'car_price') {
      if (promptValues.year) pc.params.year = parseInt(promptValues.year, 10);
      if (promptValues.km) pc.params.km = parseInt(promptValues.km, 10);
    } else if (pc.task === 'bmi') {
      if (promptValues.height) pc.params.height = parseFloat(promptValues.height);
      if (promptValues.weight) pc.params.weight = parseFloat(promptValues.weight);
      if (promptValues.age) pc.params.age = parseInt(promptValues.age, 10);
    } else if (pc.task === 'cirrhosis') {
      if (promptValues.age_years) pc.params.age = parseFloat(promptValues.age_years) * 365;
      if (promptValues.bilirubin) pc.params.bilirubin = parseFloat(promptValues.bilirubin);
    }

    delete pc.params.needs;

    try {
      addVoiceLog('⏩ Enviando comando con parámetros completados…', 'info');
      const response = await apiClient.processCommand(pc);
      addVoiceLog(`✅ ${response}`, 'success');
      setLastResult(response);
    } catch (e) {
      addVoiceLog(`❌ Error: ${e.message}`, 'error');
    } finally {
      setPendingCommand(null);
      setPromptValues({});
    }
  };

  const goToResults = () => {
    navigate('/results', { 
      state: { 
        faceRecognitionLogs, 
        voiceCommandLogs,
        emotions: currentEmotions 
      } 
    });
  };

  // 🔧 FUNCIÓN DE PRUEBA - Llama directamente al API sin voz
  const testCommand = async (commandName) => {
    const testCommands = {
      bitcoin: { task: 'bitcoin', text: 'bitcoin 7', params: { days: 7 } },
      movie: { task: 'movie', text: 'película', params: {} },
      car: { task: 'car', text: 'coche 2020 50000', params: { year: 2020, km: 50000 } },
      bmi: { task: 'bmi', text: 'imc 1.75 75 30', params: { height: 1.75, weight: 75, age: 30 } },
      london: { task: 'london', text: 'londres viernes', params: { day: 'viernes' } }
    };

    const command = testCommands[commandName];
    addVoiceLog(`🧪 Enviando comando de prueba: ${commandName}`, 'info');
    
    try {
      console.log('🧪 TEST - Enviando:', command);
      const response = await apiClient.processCommand(command);
      console.log('🧪 TEST - Respuesta:', response);
      addVoiceLog(`✅ ${response}`, 'success');
    } catch (error) {
      console.error('🧪 TEST - Error:', error);
      addVoiceLog(`❌ Error en prueba: ${error.message}`, 'error');
    }
  };

  return (
    <div className="capture-page">
      <div className="capture-header">
        <h1>JarvisTEC - Sistemas Inteligentes</h1>
        <p>Controla cada sistema de forma independiente</p>
      </div>

      {/* SECCIÓN 1: RECONOCIMIENTO FACIAL Y EMOCIONES */}
      <div className="system-section camera-section">
        <div className="section-header">
          <h2>📸 Sistema de Reconocimiento Facial</h2>
          <p>Detecta emociones en tiempo real usando la cámara</p>
        </div>

        <div className="controls">
          {!isCameraActive ? (
            <button onClick={handleStartCamera} className="btn btn-start">
              ▶️ Activar Cámara
            </button>
          ) : (
            <button onClick={handleStopCamera} className="btn btn-stop">
              ⏹️ Detener Cámara
            </button>
          )}
        </div>

        <div className="system-content">
          <div className="camera-container">
            <CameraCapture 
              isActive={isCameraActive}
              onSnapshot={handleSnapshot}
              onEmotionDetected={handleEmotionDetected}
            />
            
            {snapshotPreview && (
              <div className="snapshot-preview">
                <h3>Última captura</h3>
                <img src={snapshotPreview} alt="Snapshot preview" />
              </div>
            )}
          </div>

          <div className="analysis-results">
            <div className="emotions-container">
              <h3>😊 Emociones Detectadas</h3>
              <EmotionDisplay emotions={currentEmotions} />
            </div>
            
            <div className="gesture-container">
              <h3>👋 Gesto Reconocido</h3>
              <GestureDisplay gesture={currentGesture} />
            </div>
          </div>
        </div>

        {/* LOG DE RECONOCIMIENTO FACIAL - DEBAJO DE LA CÁMARA */}
        <div className="logs-section-inline facial-logs-inline">
          <h3>📋 Registro de Actividad</h3>
          <div className="logs-container">
            {faceRecognitionLogs.length === 0 ? (
              <p className="no-logs">No hay actividad de reconocimiento facial. Activa la cámara para comenzar.</p>
            ) : (
              faceRecognitionLogs.map(log => (
                <div key={log.id} className={`log-entry ${log.type}`}>
                  <span className="timestamp">[{log.timestamp}]</span>
                  <span className="log-icon">📸</span>
                  <span className="message">{log.message}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* SEPARADOR */}
      <div className="system-separator"></div>

      {/* SECCIÓN 2: COMANDOS DE VOZ */}
      <div className="system-section audio-section">
        <div className="section-header">
          <h2>🎤 Sistema de Comandos de Voz</h2>
          <p>Habla comandos para obtener predicciones y recomendaciones</p>
        </div>

        <div className="controls">
          {!isAudioActive ? (
            <button onClick={handleStartAudio} className="btn btn-start">
              ▶️ Activar Micrófono
            </button>
          ) : (
            <button onClick={handleStopAudio} className="btn btn-stop">
              ⏹️ Detener Micrófono
            </button>
          )}
        </div>

        <div className="system-content">
          <div className="audio-container">
            <AudioRecorder 
              isActive={isAudioActive}
              onTranscription={handleTranscription}
              onCommand={handleCommand}
            />
            {pendingCommand && (
              <div className="param-prompt">
                <h4>Completa los datos para “{pendingCommand.task}”</h4>
                {pendingCommand.task === 'movie' && (
                  <div className="prompt-form">
                    <div className="row">
                      <label>Género</label>
                      <select value={promptValues.genre || ''} onChange={e => setPromptValues(v => ({...v, genre: e.target.value}))}>
                        <option value="">-- Selecciona --</option>
                        {movieGenres.map(g => (<option key={g} value={g}>{g}</option>))}
                      </select>
                    </div>
                    <div className="row">
                      <label>Año</label>
                      <select value={promptValues.year || ''} onChange={e => setPromptValues(v => ({...v, year: e.target.value}))}>
                        <option value="">-- Selecciona --</option>
                        {movieYears.map(y => (<option key={y} value={y}>{y}</option>))}
                      </select>
                    </div>
                  </div>
                )}
                {pendingCommand.task === 'airline' && (
                  <div className="prompt-form">
                    <div className="row">
                      <label>Origen</label>
                      <select value={promptValues.origin || ''} onChange={e => setPromptValues(v => ({...v, origin: e.target.value}))}>
                        <option value="">-- Selecciona --</option>
                        {airOrigins.map(c => (<option key={c} value={c}>{c}</option>))}
                      </select>
                    </div>
                    <div className="row">
                      <label>Destino</label>
                      <select value={promptValues.dest || ''} onChange={e => setPromptValues(v => ({...v, dest: e.target.value}))}>
                        <option value="">-- Selecciona --</option>
                        {airDestinations.map(c => (<option key={c} value={c}>{c}</option>))}
                      </select>
                    </div>
                    <div className="row">
                      <label>Aerolínea</label>
                      <select value={promptValues.carrier || ''} onChange={e => setPromptValues(v => ({...v, carrier: e.target.value}))}>
                        <option value="">(opcional)</option>
                        {airCarriers.map(c => (<option key={c} value={c}>{c}</option>))}
                      </select>
                    </div>
                    <div className="row trio">
                      <div>
                        <label>Mes</label>
                        <select value={promptValues.month || ''} onChange={e => setPromptValues(v => ({...v, month: e.target.value}))}>
                          {[...Array(12)].map((_, i) => (<option key={i+1} value={i+1}>{i+1}</option>))}
                        </select>
                      </div>
                      <div>
                        <label>Día</label>
                        <input type="number" min="1" max="31" value={promptValues.day || ''} onChange={e => setPromptValues(v => ({...v, day: e.target.value}))} />
                      </div>
                      <div>
                        <label>Distancia (mi)</label>
                        <input type="number" min="50" max="5000" step="10" value={promptValues.distance || ''} onChange={e => setPromptValues(v => ({...v, distance: e.target.value}))} />
                      </div>
                    </div>
                  </div>
                )}
                {(pendingCommand.task === 'london' || pendingCommand.task === 'chicago') && (
                  <div className="prompt-form">
                    <div className="row trio">
                      <div>
                        <label>Mes</label>
                        <select value={promptValues.month || ''} onChange={e => setPromptValues(v => ({...v, month: e.target.value}))}>
                          {[...Array(12)].map((_, i) => (<option key={i+1} value={i+1}>{i+1}</option>))}
                        </select>
                      </div>
                      <div>
                        <label>Día del mes</label>
                        <input type="number" min="1" max="31" value={promptValues.date || ''} onChange={e => setPromptValues(v => ({...v, date: e.target.value}))} />
                      </div>
                      <div>
                        <label>Día de semana</label>
                        <select value={promptValues.day || 'viernes'} onChange={e => setPromptValues(v => ({...v, day: e.target.value}))}>
                          {['lunes','martes','miercoles','miércoles','jueves','viernes','sabado','sábado','domingo'].map(d => (<option key={d} value={d}>{d}</option>))}
                        </select>
                      </div>
                    </div>
                  </div>
                )}
                {(pendingCommand.task === 'avocado' || pendingCommand.task === 'bitcoin' || pendingCommand.task === 'sp500') && (
                  <div className="prompt-form">
                    <div className="row">
                      <label>Días a predecir</label>
                      <input type="number" min="1" max="365" value={promptValues.days || ''} onChange={e => setPromptValues(v => ({...v, days: e.target.value}))} placeholder="Ej. 7" />
                    </div>
                  </div>
                )}
                {pendingCommand.task === 'car_price' && (
                  <div className="prompt-form">
                    <div className="row">
                      <label>Año</label>
                      <input type="number" min="1990" max="2025" value={promptValues.year || ''} onChange={e => setPromptValues(v => ({...v, year: e.target.value}))} placeholder="Ej. 2015" />
                    </div>
                    <div className="row">
                      <label>Kilometraje</label>
                      <input type="number" min="0" value={promptValues.km || ''} onChange={e => setPromptValues(v => ({...v, km: e.target.value}))} placeholder="Ej. 50000" />
                    </div>
                  </div>
                )}
                {pendingCommand.task === 'bmi' && (
                  <div className="prompt-form">
                    <div className="row trio">
                      <div>
                        <label>Altura (m)</label>
                        <input type="number" step="0.01" value={promptValues.height || ''} onChange={e => setPromptValues(v => ({...v, height: e.target.value}))} placeholder="1.75" />
                      </div>
                      <div>
                        <label>Peso (kg)</label>
                        <input type="number" step="0.1" value={promptValues.weight || ''} onChange={e => setPromptValues(v => ({...v, weight: e.target.value}))} placeholder="70" />
                      </div>
                      <div>
                        <label>Edad</label>
                        <input type="number" value={promptValues.age || ''} onChange={e => setPromptValues(v => ({...v, age: e.target.value}))} placeholder="30" />
                      </div>
                    </div>
                  </div>
                )}
                {pendingCommand.task === 'cirrhosis' && (
                  <div className="prompt-form">
                    <div className="row">
                      <label>Edad (años)</label>
                      <input type="number" value={promptValues.age_years || ''} onChange={e => setPromptValues(v => ({...v, age_years: e.target.value}))} placeholder="50" />
                    </div>
                    <div className="row">
                      <label>Bilirrubina</label>
                      <input type="number" step="0.1" value={promptValues.bilirubin || ''} onChange={e => setPromptValues(v => ({...v, bilirubin: e.target.value}))} placeholder="1.0" />
                    </div>
                  </div>
                )}
                <div className="prompt-actions">
                  <button className="btn" onClick={confirmPrompt}>Aceptar</button>
                  <button className="btn btn-secondary" onClick={cancelPrompt}>Cancelar</button>
                </div>
              </div>
            )}
            {/* Metadatos mínimos visibles para el usuario */}
            <div className="meta-hints">
              <h4>Datos disponibles</h4>
              <div className="meta-row">
                <strong>Géneros:</strong> {movieGenres.slice(0, 8).join(', ')}{movieGenres.length > 8 ? '…' : ''}
              </div>
              <div className="meta-row">
                <strong>Años:</strong> {movieYears.slice(0, 10).join(', ')}{movieYears.length > 10 ? '…' : ''}
              </div>
              <div className="meta-row">
                <strong>Orígenes:</strong> {airOrigins.slice(0, 10).join(', ')}{airOrigins.length > 10 ? '…' : ''}
              </div>
              <div className="meta-row">
                <strong>Destinos:</strong> {airDestinations.slice(0, 10).join(', ')}{airDestinations.length > 10 ? '…' : ''}
              </div>
              <div className="meta-row">
                <strong>Aerolíneas:</strong> {airCarriers.slice(0, 10).join(', ')}{airCarriers.length > 10 ? '…' : ''}
              </div>
            </div>
          </div>

          {/* RESULTADO ACTUAL */}
          {lastResult && (
            <div className="last-result-section" style={{marginBottom: '20px', padding: '15px', background: '#f0f8ff', borderRadius: '8px', border: '1px solid #b0d4ff'}}>
              <h3 style={{marginTop: 0, color: '#0056b3'}}>🎯 Último Resultado</h3>
              <div className="result-card">
                <pre style={{whiteSpace: 'pre-wrap', wordWrap: 'break-word'}}>{JSON.stringify(lastResult, null, 2)}</pre>
              </div>
            </div>
          )}

          {/* LOG DE COMANDOS DE VOZ - AL LADO DEL MICRÓFONO */}
          <div className="logs-section-inline voice-logs-inline">
            <h3>📋 Registro de Actividad</h3>
            <div className="logs-container">
              {voiceCommandLogs.length === 0 ? (
                <p className="no-logs">No hay actividad de comandos de voz. Activa el micrófono para comenzar.</p>
              ) : (
                voiceCommandLogs.map(log => (
                  <div key={log.id} className={`log-entry ${log.type}`}>
                    <span className="timestamp">[{log.timestamp}]</span>
                    <span className="log-icon">🎤</span>
                    <span className="message">{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* BOTÓN DE RESULTADOS */}
      <div className="results-button-container">
        <button onClick={goToResults} className="btn btn-secondary btn-large">
          📊 Ver Estadísticas y Resultados
        </button>
      </div>
    </div>
  );
}

export default Capture;
