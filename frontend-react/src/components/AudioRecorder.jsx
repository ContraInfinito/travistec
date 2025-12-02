import { useState, useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import './AudioRecorder.css';

function AudioRecorder({ onTranscription, onCommand, isActive }) {
  const [isRecording, setIsRecording] = useState(false);
  const [useWebSpeech, setUseWebSpeech] = useState(false);
  const [interimText, setInterimText] = useState('');
  const mediaRecorderRef = useRef(null);
  const recognitionRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    if (isActive) {
      startRecording();
    } else {
      stopRecording();
    }

    return () => {
      stopRecording();
    };
  }, [isActive]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Intentar usar Web Speech API primero
      // const started = startWebSpeechRecognition();
      const started = false; // Force MediaRecorder for custom model
      
      if (started) {
        setUseWebSpeech(true);
        setIsRecording(true);
      } else {
        // Fallback a MediaRecorder
        setUseWebSpeech(false);
        startMediaRecorder(stream);
      }
    } catch (error) {
      console.error('Error al iniciar grabación:', error);
      alert('No se pudo acceder al micrófono. Verifica los permisos.');
    }
  };

  const startWebSpeechRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      console.log('Web Speech API no disponible');
      return false;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = 'es-ES';
      recognition.interimResults = true; // enable live partials
      recognition.continuous = true;

      recognition.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript + ' ';
          } else {
            interimTranscript += transcript + ' ';
          }
        }
        if (interimTranscript) setInterimText(interimTranscript.trim());
        if (finalTranscript) {
          const text = finalTranscript.trim();
          setInterimText('');
          if (onTranscription) onTranscription(text);
          const parsed = parseCommand(text);
          if (parsed && onCommand) onCommand(parsed);
        }
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
      };

      recognition.start();
      recognitionRef.current = recognition;
      setIsRecording(true);
      return true;
    } catch (error) {
      console.error('Error iniciando Web Speech:', error);
      return false;
    }
  };

  const startMediaRecorder = (stream) => {
    const mediaRecorder = new MediaRecorder(stream);
    const audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      
      if (onTranscription) {
        onTranscription(audioBlob);
      }

      // Reiniciar grabación si sigue activo
      if (isActive) {
        audioChunks.length = 0;
        mediaRecorder.start();
        setTimeout(() => {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
          }
        }, 5000);
      }
    };

    mediaRecorder.start();
    mediaRecorderRef.current = mediaRecorder;
    setIsRecording(true);

    // Grabar en intervalos de 5 segundos
    setTimeout(() => {
      if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
    }, 5000);
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
        recognitionRef.current = null;
      } catch (error) {
        console.warn('Error deteniendo reconocimiento:', error);
      }
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    setIsRecording(false);
  };

  const parseCommand = (text) => {
    const t = text.toLowerCase();
    
    // Más flexible - no requiere exactamente "travistec"
    if (!t.includes('travis') && t.length < 3) {
      return null;
    }

    // Limpiar texto
    let payloadText = t.replace('travis tec', '').replace('travistec', '').replace('travis', '').trim();

    // Extraer números y palabras clave
    const numbers = payloadText.match(/\d+(\.\d+)?/g) || [];
    const params = {};
    let task = 'unknown';

    // 1) Bitcoin - buscar en cualquier parte del texto (predicción a CORTO PLAZO)
    if (payloadText.match(/\b(bitcoin|btc|bit coin)\b/i)) {
      task = 'bitcoin';
      if (numbers.length > 0) {
        params.days = parseInt(numbers[0]); // Cambiado a días
      }
    }

    // 2) Película - extrae género y año si están presentes
    else if (payloadText.match(/\b(pel[ií]culas?|movies?|recomienda|recomendaci[óo]n|film|films)\b/i)) {
      task = 'movie';
      // posibles géneros comunes (ES e inglés)
      const genreMap = {
        'accion': 'Action','acción': 'Action','action': 'Action',
        'comedia': 'Comedy','comedy': 'Comedy',
        'drama': 'Drama',
        'romance': 'Romance','romantic': 'Romance',
        'terror': 'Horror','horror': 'Horror',
        'thriller': 'Thriller','suspenso': 'Thriller',
        'animacion': 'Animation','animación': 'Animation','animation': 'Animation',
        'aventura': 'Adventure','adventure': 'Adventure',
        'crimen': 'Crime','crime': 'Crime',
        'documental': 'Documentary','documentary': 'Documentary',
        'familia': 'Family','family': 'Family',
        'fantasia': 'Fantasy','fantasía': 'Fantasy','fantasy': 'Fantasy',
        'misterio': 'Mystery','mystery': 'Mystery',
        'musical': 'Music','music': 'Music',
        'ciencia ficcion': 'Science Fiction','ciencia ficción': 'Science Fiction','scifi': 'Science Fiction','science fiction': 'Science Fiction',
        'guerra': 'War','war': 'War',
        'historia': 'History','history': 'History',
        'western': 'Western'
      };
      const yearMatch = payloadText.match(/(19\d{2}|20\d{2})/);
      if (yearMatch) {
        const yearNum = parseInt(yearMatch[1], 10);
        if (yearNum >= 1900 && yearNum <= 2035) params.year = yearNum;
      }
      // buscar palabra de género
      const gKeys = Object.keys(genreMap);
      const found = gKeys.find(g => payloadText.includes(g));
      if (found) params.genre = genreMap[found];
      // si faltan parámetros, solicitarlos
      const needs = [];
      if (!params.genre) needs.push('genre');
      if (!params.year) needs.push('year');
      if (needs.length) params.needs = needs;
    }

    // 3) Automóvil - más variaciones (acepta plurales y "teccar")
    else if (payloadText.match(/\b(autom[óo]viles?|autos?|coches?|carros?|cars?|veh[íi]culos?|teccar|tec\s*car)\b/i)) {
      task = 'car';
      if (numbers.length >= 2) {
        params.year = parseInt(numbers[0], 10);
        params.km = parseInt(numbers[1], 10);
      }
    }

    // 4) SP500 - muy flexible (acepta 500, 507, 50, etc.) - predicción a CORTO PLAZO
    else if (payloadText.match(/\b(sp\s*[45]\d{2}|sp\s*50\d?|s\s*[&y]?\s*p\s*[45]\d{2}|s\s*[&y]?\s*p\s*50\d?|standard|sandp|s\s*and\s*p)\b/i)) {
      task = 'sp500';
      if (numbers.length > 0) {
        params.days = parseInt(numbers[0]); // Cambiado a días
      }
    }

    // 5) Masa corporal - más variaciones
    else if (payloadText.match(/\b(masa\s*corporal|imc|bmi|grasa|peso|altura)\b/i)) {
      task = 'bmi';
      if (numbers.length >= 2) {
        params.height = parseFloat(numbers[0]);
        params.weight = parseFloat(numbers[1]);
      }
      if (numbers.length >= 3) {
        params.age = parseInt(numbers[2], 10);
      }
    }

    // 6) Aguacate - más flexible (acepta plurales) - predicción a CORTO PLAZO
    else if (payloadText.match(/\b(aguacates?|avocados?|paltas?)\b/i)) {
      task = 'avocado';
      if (numbers.length > 0) {
        params.days = parseInt(numbers[0]); // Cambiado a días
      }
    }

    // 7) Londres - acepta mes (nombre) y día del mes
    else if (payloadText.match(/\b(londres|london)\b/i)) {
      task = 'london';
      const days = ['lunes', 'martes', 'miercoles', 'miércoles', 'jueves', 'viernes', 'sabado', 'sábado', 'domingo'];
      const foundDay = days.find(d => payloadText.includes(d));
      if (foundDay) params.day = foundDay;
      const months = {
        'enero':1,'febrero':2,'marzo':3,'abril':4,'mayo':5,'junio':6,
        'julio':7,'agosto':8,'septiembre':9,'octubre':10,'noviembre':11,'diciembre':12
      };
      const mName = Object.keys(months).find(m => payloadText.includes(m));
      if (mName) params.month = months[mName];
      // día del mes (número después del mes)
      const domMatch = payloadText.match(new RegExp(`${mName ?? ''}[^\d]*(\d{1,2})`));
      if (domMatch) params.date = parseInt(domMatch[1], 10);
      const needs = [];
      if (!params.month) needs.push('month');
      if (!params.date) needs.push('day');
      if (needs.length) params.needs = needs;
    }

    // 8) Chicago - acepta mes (nombre) y día del mes
    else if (payloadText.match(/\b(chicago)\b/i)) {
      task = 'chicago';
      const days = ['lunes', 'martes', 'miercoles', 'miércoles', 'jueves', 'viernes', 'sabado', 'sábado', 'domingo'];
      const foundDay = days.find(d => payloadText.includes(d));
      if (foundDay) params.day = foundDay;
      const months = {
        'enero':1,'febrero':2,'marzo':3,'abril':4,'mayo':5,'junio':6,
        'julio':7,'agosto':8,'septiembre':9,'octubre':10,'noviembre':11,'diciembre':12
      };
      const mName = Object.keys(months).find(m => payloadText.includes(m));
      if (mName) params.month = months[mName];
      const domMatch = payloadText.match(new RegExp(`${mName ?? ''}[^\d]*(\d{1,2})`));
      if (domMatch) params.date = parseInt(domMatch[1], 10);
      const needs = [];
      if (!params.month) needs.push('month');
      if (!params.date) needs.push('day');
      if (needs.length) params.needs = needs;
    }

    // 9) Cirrosis - más flexible (acepta plurales)
    else if (payloadText.match(/\b(cirrosis|cirrhosis|h[íi]gados?)\b/i)) {
      task = 'cirrhosis';
      // Extraer edad y bilirrubina si se dan parámetros numéricos
      if (numbers.length >= 1) {
        params.age = parseFloat(numbers[0]) * 365; // Convertir años a días
      }
      if (numbers.length >= 2) {
        params.bilirubin = parseFloat(numbers[1]);
      }
    }

    // 10) Avión - acepta origen, destino (IATA 3 letras) y aerolínea
    else if (payloadText.match(/\b(aviones?|vuelos?|flights?|aerol[íi]neas?)\b/i)) {
      task = 'airline';
      // Extraer mes, día y distancia
      if (numbers.length >= 1) {
        params.month = parseInt(numbers[0], 10);  // Primer número = mes
      }
      if (numbers.length >= 2) {
        params.day = parseInt(numbers[1], 10);  // Segundo número = día
      }
      if (numbers.length >= 3) {
        params.distance = parseInt(numbers[2], 10);  // Tercer número = distancia en millas
      }
      
      // Extraer mes por nombre
      const months = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
      };
      const foundMonth = Object.keys(months).find(m => payloadText.includes(m));
      if (foundMonth) {
        params.month = months[foundMonth];
      }

      // Origen/destino IATA: buscar 3 letras después de palabras clave
      const iataMatchOrigin = payloadText.match(/(origen|desde)\s+([a-z]{3})/i);
      const iataMatchDest = payloadText.match(/(destino|hacia|a)\s+([a-z]{3})/i);
      if (iataMatchOrigin) params.origin = iataMatchOrigin[2].toUpperCase();
      if (iataMatchDest) params.dest = iataMatchDest[2].toUpperCase();

      // Aerolínea por nombre -> código aproximado
      const carrierMap = {
        'delta': 'DL', 'american': 'AA', 'americana': 'AA', 'united': 'UA',
        'jetblue': 'B6', 'spirit': 'NK', 'southwest': 'WN', 'alaska': 'AS'
      };
      const carrKey = Object.keys(carrierMap).find(k => payloadText.includes(k));
      if (carrKey) params.carrier = carrierMap[carrKey];

      const needs = [];
      if (!params.origin) needs.push('origin');
      if (!params.dest) needs.push('dest');
      if (needs.length) params.needs = needs;
    }

    return {
      text: payloadText,
      task: task,
      params: params
    };
  };

  return (
    <div className="audio-recorder">
      <div className={`recording-indicator ${isRecording ? 'active' : ''}`}>
        <div className="pulse"></div>
        <span>
          {isRecording 
            ? `🎤 Grabando ${useWebSpeech ? '(Web Speech)' : '(MediaRecorder)'}`
            : '🎤 Micrófono inactivo'
          }
        </span>
      </div>
      {/* Live interim transcription */}
      {useWebSpeech && isRecording && (
        <div className="interim-text">
          <strong>Escuchando…</strong>
          <div className="bubble">{interimText || 'Empieza a hablar'}</div>
        </div>
      )}

      {/* Available voice commands */}
      <div className="commands-help">
        <h4>Comandos disponibles</h4>
        <ul>
          <li>“Travis TEC bitcoin 7 días”</li>
          <li>“Travis TEC película acción 2012”</li>
          <li>“Travis TEC coche 2020 50000”</li>
          <li>“Travis TEC IMC 1.75 75 30”</li>
          <li>“Travis TEC Londres agosto 23 viernes”</li>
          <li>“Travis TEC Chicago agosto 23 martes”</li>
          <li>“Travis TEC aguacate 5 días”</li>
          <li>“Travis TEC avión junio 10 300 origen MIA destino JFK aerolínea Delta”</li>
        </ul>
        <p>Tip: Di “Travis TEC …” para activar el comando.</p>
      </div>
    </div>
  );
}

AudioRecorder.propTypes = {
  onTranscription: PropTypes.func,
  onCommand: PropTypes.func,
  isActive: PropTypes.bool.isRequired
};

export default AudioRecorder;
