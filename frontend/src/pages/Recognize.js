import { useState, useRef } from 'react';
import axios from 'axios';

export default function Recognize() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [status, setStatus]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [useCamera, setUseCamera] = useState(false);
  
  const videoRef  = useRef(null);
  const streamRef = useRef(null);

  const startCamera = async () => {
    setUseCamera(true);
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    streamRef.current = stream;
    setTimeout(() => { if (videoRef.current) videoRef.current.srcObject = stream; }, 100);
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    setUseCamera(false);
  };

  const capturePhoto = () => {
    const canvas = document.createElement('canvas');
    canvas.width  = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
    canvas.toBlob(blob => {
      const f = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
      setFile(f);
      setPreview(URL.createObjectURL(blob));
      stopCamera();
    }, 'image/jpeg');
  };

  const handleFile = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleSubmit = async () => {
    if (!file) return setStatus({ type: 'error', msg: 'Please provide a face image.' });
    setLoading(true); setStatus(null);
    const form = new FormData();
    form.append('image', file);

    try {
      const res = await axios.post('http://localhost:5001/client/recognize', form);
      if (res.data.status === 'recognized') {
        setStatus({ type: 'success', msg: `Recognized: ${res.data.identity} (Confidence: ${(res.data.confidence * 100).toFixed(1)}%)` });
      } else if (res.data.status === 'unknown') {
        setStatus({ type: 'error', msg: `Unknown user. (Best match: ${(res.data.confidence * 100).toFixed(1)}%)` });
      } else {
        setStatus({ type: 'error', msg: res.data.message || 'Unknown status.' });
      }
    } catch (err) {
      setStatus({ type: 'error', msg: err.response?.data?.detail || err.response?.data?.message || 'Server error.' });
    }
    setLoading(false);
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h2 style={styles.title}>👁️ Face Recognition</h2>
        <p style={styles.sub}>Scan your face to see if the global model recognizes you.</p>

        <div style={{ display: 'flex', gap: 8 }}>
          <label style={{ ...styles.uploadBtn, flex: 1 }}>
            📁 Upload
            <input type="file" accept="image/*" onChange={handleFile} hidden />
          </label>
          <button style={{ ...styles.uploadBtn, flex: 1, border: 'none', cursor: 'pointer' }}
            onClick={useCamera ? stopCamera : startCamera}>
            {useCamera ? '✖ Cancel' : '📷 Camera'}
          </button>
        </div>

        {useCamera && (
          <div>
            <video ref={videoRef} autoPlay playsInline style={styles.preview} />
            <button onClick={capturePhoto} style={styles.btn}>📸 Capture</button>
          </div>
        )}

        {preview && !useCamera &&
          <img src={preview} alt="preview" style={styles.preview} />}

        <button onClick={handleSubmit} style={styles.btn} disabled={loading}>
          {loading ? 'Processing...' : 'Recognize Face'}
        </button>

        {status && (
          <div style={{ ...styles.alert,
            background: status.type === 'success' ? '#d4edda' : '#f8d7da',
            color:      status.type === 'success' ? '#155724' : '#721c24' }}>
            {status.type === 'success' ? '✅' : '❌'} {status.msg}
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  page:      { minHeight: '90vh', display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#f0f2f5' },
  card:      { background: '#fff', padding: 36, borderRadius: 16, width: 360, boxShadow: '0 4px 20px rgba(0,0,0,0.1)', display: 'flex', flexDirection: 'column', gap: 14 },
  title:     { margin: 0, color: '#1a1a2e', textAlign: 'center' },
  sub:       { color: '#666', fontSize: 13, textAlign: 'center', margin: 0 },
  input:     { padding: '10px 14px', borderRadius: 8, border: '1px solid #ccc', fontSize: 15 },
  uploadBtn: { background: '#eef', padding: '10px 14px', borderRadius: 8, cursor: 'pointer', textAlign: 'center', fontWeight: 600, color: '#333' },
  preview:   { width: '100%', borderRadius: 10, maxHeight: 220, objectFit: 'cover' },
  btn:       { background: '#1a1a2e', color: '#fff', padding: 12, borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 15, fontWeight: 600 },
  alert:     { padding: '10px 14px', borderRadius: 8, fontWeight: 500 },
};
