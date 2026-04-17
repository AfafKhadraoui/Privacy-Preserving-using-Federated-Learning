import { useState, useRef } from 'react';
import axios from 'axios';

export default function Register() {
  const [name, setName]       = useState('');
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [status, setStatus]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [useCamera, setUseCamera] = useState(false); // toggle camera mode
  const videoRef  = useRef(null);
  const streamRef = useRef(null); // keep stream ref to stop it later

  // Start webcam stream
  const startCamera = async () => {
    setUseCamera(true);
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    streamRef.current = stream;
    // Attach stream to video element after it mounts
    setTimeout(() => { if (videoRef.current) videoRef.current.srcObject = stream; }, 100);
  };

  // Stop camera and switch back to upload mode
  const stopCamera = () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    setUseCamera(false);
  };

  // Capture frame from video, convert to Blob, set as file
  const capturePhoto = () => {
    const canvas = document.createElement('canvas');
    canvas.width  = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
    canvas.toBlob(blob => {
      const f = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
      setFile(f);
      setPreview(URL.createObjectURL(blob));
      stopCamera(); // close camera after capture
    }, 'image/jpeg');
  };

  const handleFile = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleSubmit = async () => {
    if (!name || !file) return setStatus({ type: 'error', msg: 'Fill all fields.' });
    setLoading(true); setStatus(null);
    const form = new FormData();
    form.append('name', name);
    form.append('image', file);
    try {
      const res = await axios.post('http://localhost:5000/api/register', form);
      setStatus({ type: 'success', msg: res.data.message });
    } catch (err) {
      setStatus({ type: 'error', msg: err.response?.data?.message || 'Server error.' });
    }
    setLoading(false);
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h2 style={styles.title}>🔐 Face Registration</h2>
        <p style={styles.sub}>Your photo stays on this device — privacy preserved.</p>

        <input placeholder="Full Name" value={name}
          onChange={e => setName(e.target.value)} style={styles.input} />

        {/* Toggle between upload and camera */}
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

        {/* Live camera feed */}
        {useCamera && (
          <div>
            <video ref={videoRef} autoPlay playsInline style={styles.preview} />
            <button onClick={capturePhoto} style={styles.btn}>📸 Capture</button>
          </div>
        )}

        {/* Preview of uploaded/captured photo */}
        {preview && !useCamera &&
          <img src={preview} alt="preview" style={styles.preview} />}

        <button onClick={handleSubmit} style={styles.btn} disabled={loading}>
          {loading ? 'Processing...' : 'Register Face'}
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