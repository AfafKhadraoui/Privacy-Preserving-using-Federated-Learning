import { useEffect, useRef, useState } from 'react';
import axios from 'axios';

const CLIENT_API_URL = 'http://localhost:5001';

export default function Register() {
  const [name, setName] = useState('');
  const [registrationConfig, setRegistrationConfig] = useState(null);
  const [captures, setCaptures] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [useCamera, setUseCamera] = useState(false);

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    let mounted = true;
    axios.get(`${CLIENT_API_URL}/client/registration-config`)
      .then((res) => {
        if (!mounted) return;
        const instructions = res.data.instructions || [];
        setRegistrationConfig({
          numImages: res.data.num_registration_images,
          instructions,
        });
        setCaptures(Array.from({ length: res.data.num_registration_images }, () => null));
      })
      .catch(() => {
        if (mounted) {
          setStatus({ type: 'error', msg: 'Could not load registration settings from the local API.' });
        }
      });

    return () => {
      mounted = false;
      stopCamera();
    };
  }, []);

  useEffect(() => {
    if (useCamera && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [currentStep, useCamera]);

  const numImages = registrationConfig?.numImages || 0;
  const currentCapture = captures[currentStep];
  const completedCount = captures.filter(Boolean).length;
  const allCaptured = numImages > 0 && completedCount === numImages;

  const startCamera = async () => {
    setStatus(null);
    setUseCamera(true);
    if (!streamRef.current) {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
    }
    setTimeout(() => {
      if (videoRef.current) videoRef.current.srcObject = streamRef.current;
    }, 100);
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setUseCamera(false);
  };

  const setStepImage = (file, preview) => {
    setCaptures((prev) => {
      const next = [...prev];
      if (next[currentStep]?.preview) {
        URL.revokeObjectURL(next[currentStep].preview);
      }
      next[currentStep] = { file, preview };
      return next;
    });
  };

  const capturePhoto = () => {
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
    canvas.toBlob((blob) => {
      const file = new File([blob], `registration-${currentStep + 1}.jpg`, { type: 'image/jpeg' });
      setStepImage(file, URL.createObjectURL(blob));
      setUseCamera(false);
    }, 'image/jpeg');
  };

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setStepImage(file, URL.createObjectURL(file));
    e.target.value = '';
  };

  const retakeCurrentImage = () => {
    const shouldResumeCamera = Boolean(streamRef.current);
    setCaptures((prev) => {
      const next = [...prev];
      if (next[currentStep]?.preview) {
        URL.revokeObjectURL(next[currentStep].preview);
      }
      next[currentStep] = null;
      return next;
    });
    if (shouldResumeCamera) {
      setUseCamera(true);
    }
    setStatus(null);
  };

  const goToStep = (stepIndex) => {
    setCurrentStep(Math.max(0, Math.min(numImages - 1, stepIndex)));
    if (streamRef.current) {
      setUseCamera(true);
    }
  };

  const handleSubmit = async () => {
    if (!name.trim()) {
      return setStatus({ type: 'error', msg: 'Please enter your name.' });
    }
    if (!allCaptured) {
      return setStatus({
        type: 'error',
        msg: `Please upload or capture all ${numImages} images before registering, not just ${completedCount}.`,
      });
    }

    setLoading(true);
    setStatus(null);

    const form = new FormData();
    form.append('name', name.trim());
    captures.forEach((capture, index) => {
      form.append('images', capture.file, `registration-${index + 1}.jpg`);
    });

    try {
      const res = await axios.post(`${CLIENT_API_URL}/client/register`, form);
      setStatus({ type: 'success', msg: res.data.message });
    } catch (err) {
      setStatus({ type: 'error', msg: err.response?.data?.detail || err.response?.data?.message || 'Server error.' });
    }
    stopCamera();
    setLoading(false);
  };

  const instruction = registrationConfig?.instructions[currentStep] || 'Capture image';

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h2 style={styles.title}>🔐 Face Registration</h2>
        <p style={styles.sub}>Your photos stay on this device. Only embeddings participate in FL.</p>

        <input
          placeholder="Full Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={styles.input}
        />

        {registrationConfig && (
          <>
            <div style={styles.progressRow}>
              <span style={styles.progressText}>Step {currentStep + 1}/{numImages}</span>
              <span style={styles.progressText}>{completedCount}/{numImages} captured</span>
            </div>
            <div style={styles.progressTrack}>
              <div style={{ ...styles.progressFill, width: `${(completedCount / numImages) * 100}%` }} />
            </div>

            <div style={styles.stepPanel}>
              <p style={styles.instruction}>{instruction}</p>

              <div style={{ display: 'flex', gap: 8 }}>
                <label style={{ ...styles.uploadBtn, flex: 1 }}>
                  📁 Upload
                  <input type="file" accept="image/*" onChange={handleFile} hidden />
                </label>
                <button
                  style={{ ...styles.uploadBtn, flex: 1, border: 'none' }}
                  onClick={useCamera ? stopCamera : startCamera}
                  type="button"
                >
                  {useCamera ? '✖ Cancel' : '📷 Camera'}
                </button>
              </div>

              {useCamera && (
                <div style={styles.captureArea}>
                  <video ref={videoRef} autoPlay playsInline style={styles.preview} />
                  <button onClick={capturePhoto} style={styles.btn} type="button">📸 Capture</button>
                </div>
              )}

              {currentCapture && !useCamera && (
                <div style={styles.captureArea}>
                  <img src={currentCapture.preview} alt={instruction} style={styles.preview} />
                  <button onClick={retakeCurrentImage} style={styles.secondaryBtn} type="button">🔁 Retake</button>
                </div>
              )}
            </div>

            <div style={styles.navRow}>
              <button
                onClick={() => goToStep(currentStep - 1)}
                style={styles.secondaryBtn}
                disabled={currentStep === 0 || loading}
                type="button"
              >
                ← Back
              </button>
              <button
                onClick={() => goToStep(currentStep + 1)}
                style={styles.secondaryBtn}
                disabled={!currentCapture || currentStep === numImages - 1 || loading}
                type="button"
              >
                Next →
              </button>
            </div>
          </>
        )}

        <button onClick={handleSubmit} style={styles.btn} disabled={loading}>
          {loading ? '⏳ Processing...' : '✅ Register Face'}
        </button>

        {status && (
          <div style={{
            ...styles.alert,
            background: status.type === 'success' ? '#d4edda' : '#f8d7da',
            color: status.type === 'success' ? '#155724' : '#721c24',
          }}>
            {status.msg}
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  page: { minHeight: '90vh', display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#f0f2f5' },
  card: { background: '#fff', padding: 32, borderRadius: 8, width: 390, boxShadow: '0 4px 20px rgba(0,0,0,0.1)', display: 'flex', flexDirection: 'column', gap: 14 },
  title: { margin: 0, color: '#1a1a2e', textAlign: 'center' },
  sub: { color: '#666', fontSize: 13, textAlign: 'center', margin: 0 },
  input: { padding: '10px 14px', borderRadius: 8, border: '1px solid #ccc', fontSize: 15 },
  progressRow: { display: 'flex', justifyContent: 'space-between', gap: 12 },
  progressText: { color: '#555', fontSize: 13, fontWeight: 600 },
  progressTrack: { height: 8, background: '#e8e8f5', borderRadius: 8, overflow: 'hidden' },
  progressFill: { height: '100%', background: '#1a1a2e', transition: 'width 180ms ease' },
  stepPanel: { display: 'flex', flexDirection: 'column', gap: 12 },
  instruction: { margin: 0, color: '#1a1a2e', fontSize: 18, fontWeight: 700, textAlign: 'center' },
  uploadBtn: { background: '#eef', padding: '10px 14px', borderRadius: 8, cursor: 'pointer', textAlign: 'center', fontWeight: 600, color: '#333' },
  captureArea: { display: 'flex', flexDirection: 'column', gap: 10 },
  preview: { width: '100%', borderRadius: 8, maxHeight: 220, objectFit: 'cover', background: '#111' },
  navRow: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 },
  btn: { background: '#1a1a2e', color: '#fff', padding: 12, borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 15, fontWeight: 600 },
  secondaryBtn: { background: '#f4f4f8', color: '#1a1a2e', padding: 11, borderRadius: 8, border: '1px solid #d9d9e8', cursor: 'pointer', fontSize: 14, fontWeight: 600 },
  alert: { padding: '10px 14px', borderRadius: 8, fontWeight: 500 },
};
