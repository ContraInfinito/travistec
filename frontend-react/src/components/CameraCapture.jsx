import { useRef, useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import './CameraCapture.css';

function CameraCapture({ onSnapshot, onEmotionDetected, isActive }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  // Removed auto-capture interval to analyze only on manual snapshot
  const [captureInterval, setCaptureInterval] = useState(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (isActive) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [isActive]);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        // Wait for metadata/canplay to ensure dimensions are set
        const v = videoRef.current;
        const onReady = () => setIsReady(true);
        v.addEventListener('loadedmetadata', onReady, { once: true });
        v.addEventListener('canplay', onReady, { once: true });
        try { v.play(); } catch {}
      }
      
      setStream(mediaStream);

      // Auto-capture disabled: analysis happens only when user clicks snapshot
      // If needed later, re-enable with setInterval and captureFrame()
    } catch (error) {
      console.error('Error al iniciar cámara:', error);
      alert('No se pudo acceder a la cámara. Verifica los permisos.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsReady(false);

    // No periodic capture to clear
  };

  // captureFrame was used for periodic captures; now rely solely on manual snapshot

  const takeManualSnapshot = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    // If video not ready yet, avoid capturing a black frame
    if (!isReady || (video.videoWidth || 0) < 2 || (video.videoHeight || 0) < 2) {
      console.warn('Snapshot attempted before camera ready');
      return;
    }

    const canvas = canvasRef.current;
    
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (blob) {
        if (onSnapshot) {
          onSnapshot(blob);
        }
        if (onEmotionDetected) {
          onEmotionDetected(blob);
        }
      }
    }, 'image/jpeg', 0.9);
  };

  return (
    <div className="camera-capture">
      <div className="video-container">
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline
          className={isActive ? 'active' : ''}
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        
        {!isActive && (
          <div className="video-overlay">
            <p>Cámara inactiva</p>
          </div>
        )}
      </div>
      
      {isActive && (
        <button 
          onClick={takeManualSnapshot}
          className="snapshot-btn"
          disabled={!isReady}
        >
          {isReady ? 'Tomar Foto' : 'Preparando cámara…'}
        </button>
      )}
    </div>
  );
}

CameraCapture.propTypes = {
  onSnapshot: PropTypes.func,
  onEmotionDetected: PropTypes.func,
  isActive: PropTypes.bool.isRequired
};

export default CameraCapture;
