import { DrawingUtils, FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import { useEffect, useRef, useState } from "react";

// ── TYPES & INTERFACES ─────────────────────────────────────────
interface landmark { x: number; y: number; z: number; visibility?: number; }
interface FallFeatures { torsoLegAngle: number | null; kneeAnkleDistance: number; headFloorDistance: number; headAngle: number; noseAnkleDistance: number; aspectRatio: number; }
interface FrameVote { torsoLegAngle: boolean; kneeAnkleDistance: boolean; headFloorDistance: boolean; headAngle: boolean; noseAnkleDistance: boolean; aspectRatio: boolean; }

// ── CONFIGURATION ──────────────────────────────────────────────
const MAX_BUFFER = 40;            // The rolling window for confidence calculation
const FALL_THRESHOLD = 0.6;       // 60% confidence triggers the fall state
const RECOVERY_THRESHOLD = 0.35;  // Below 35% confidence triggers recovery timer
const WS_URL = "ws://10.0.1.25:8080"; 

// Weighted voting: Prioritizes critical metrics like torso angle and height
const FEATURE_WEIGHTS: Record<keyof FrameVote, number> = { 
  torsoLegAngle: 3, 
  headFloorDistance: 2, 
  noseAnkleDistance: 2, 
  aspectRatio: 2, 
  kneeAnkleDistance: 1, 
  headAngle: 1 
};
const MAX_SCORE_PER_FRAME = Object.values(FEATURE_WEIGHTS).reduce((a, b) => a + b, 0);

// ── MATHEMATICAL UTILITIES ─────────────────────────────────────
const calculateTorsoLegAngle = (s: landmark, h: landmark, k: landmark) => {
  const ax = h.x - s.x, ay = h.y - s.y, bx = k.x - h.x, by = k.y - h.y;
  const dot = ax * bx + ay * by, magA = Math.sqrt(ax*ax + ay*ay), magB = Math.sqrt(bx*bx + by*by);
  return magA === 0 || magB === 0 ? null : (Math.acos(Math.min(1, Math.max(-1, dot / (magA * magB)))) * 180) / Math.PI;
};

const buildFrameVote = (f: FallFeatures): FrameVote => ({
  torsoLegAngle: ((f.torsoLegAngle ?? 180) >= 70 && (f.torsoLegAngle ?? 180) <= 110) || (f.torsoLegAngle ?? 180) < 30,
  kneeAnkleDistance: f.kneeAnkleDistance < 0.25,
  headFloorDistance: f.headFloorDistance < 0.35,
  headAngle: f.headAngle < 45 || f.headAngle > 135,
  noseAnkleDistance: f.noseAnkleDistance < 1.3,
  aspectRatio: f.aspectRatio > 1.2, 
});

const PoseEngine = () => {
  // ── REACT STATE ──────────────────────────────────────────────
  const [isPredicting, setIsPredicting] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [fallDetected, setFallDetected] = useState(false);
  const [isFalseAlarm, setIsFalseAlarm] = useState(false);
  const [hardwareAlert, setHardwareAlert] = useState(false);
  const [wsStatus, setWsStatus] = useState<"connecting" | "connected" | "disconnected">("disconnected");
  
  // Monitoring Timers
  const [recoveryCounter, setRecoveryCounter] = useState(0);
  const [fallDurationCounter, setFallDurationCounter] = useState(0);

  // ── REFS (PERFORMANCE & PERSISTENCE) ─────────────────────────
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  
  // Logic Refs (Updating these doesn't trigger a re-render)
  const frameBufferRef = useRef<FrameVote[]>([]); // This is your buffer!
  const stopSignalRef = useRef(false);            // The Kill Switch
  const recoveryTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const twilioTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── 1. TWILIO TIMER (12s) ────────────────────────────────────
  useEffect(() => {
    if (fallDetected && !isFalseAlarm) {
      if (!twilioTimerRef.current) {
        twilioTimerRef.current = setInterval(() => {
          setFallDurationCounter(prev => {
            if (prev >= 11) { // On the 12th second
              console.log("SENDING TWILIO ALERT VIA WEBSOCKET...");
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: "SEND_TWILIO_SMS" }));
              }
              clearInterval(twilioTimerRef.current!);
              return 12;
            }
            return prev + 1;
          });
        }, 1000);
      }
    } else {
      if (twilioTimerRef.current) clearInterval(twilioTimerRef.current);
      twilioTimerRef.current = null;
      setFallDurationCounter(0);
    }
  }, [fallDetected, isFalseAlarm]);

  // ── 2. RECOVERY / FALSE ALARM TIMER (10s) ────────────────────
  useEffect(() => {
    if (fallDetected && confidence < RECOVERY_THRESHOLD) {
      if (!recoveryTimerRef.current) {
        recoveryTimerRef.current = setInterval(() => {
          setRecoveryCounter(p => {
            if (p >= 9) { // After 10s of movement/standing
              setIsFalseAlarm(true);
              setFallDetected(false);
              setHardwareAlert(false);
              return 0;
            }
            return p + 1;
          });
        }, 1000);
      }
    } else {
      if (recoveryTimerRef.current) clearInterval(recoveryTimerRef.current);
      recoveryTimerRef.current = null;
      setRecoveryCounter(0);
      // Reset false alarm if we detect a new high-confidence fall
      if (confidence > FALL_THRESHOLD) setIsFalseAlarm(false);
    }
  }, [confidence, fallDetected]);

  // ── 3. WEBSOCKET SETUP ───────────────────────────────────────
  useEffect(() => {
    const connect = () => {
      setWsStatus("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => setWsStatus("connected");
      ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (d.type === "FALL_DETECTED") {
          setHardwareAlert(true);
          setIsFalseAlarm(false);
          handleStart(); // Automatically boot up vision engine if sensor hits
        }
      };
      ws.onclose = () => { setWsStatus("disconnected"); setTimeout(connect, 3000); };
    };
    connect();
    return () => wsRef.current?.close();
  }, []);

  // ── 4. CONTROL HANDLERS (EXPLICIT BUTTON LOGIC) ──────────────
  const handleStart = async () => {
    stopSignalRef.current = false; // Release kill switch
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadeddata = () => {
        videoRef.current!.play();
        setIsPredicting(true);
        predictFall();
      };
    }
  };

  const handleStop = () => {
    stopSignalRef.current = true; // Activate kill switch
    setIsPredicting(false);
    setConfidence(0);
    setFallDetected(false);
    setIsFalseAlarm(false);
    setHardwareAlert(false);
    
    // Stop the camera hardware tracks
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    
    // Clear canvas visual leftovers
    const ctx = canvasRef.current?.getContext("2d");
    ctx?.clearRect(0, 0, canvasRef.current?.width || 0, canvasRef.current?.height || 0);
  };

  // ── 5. THE AI LOOP ───────────────────────────────────────────
  const predictFall = () => {
    // Kill Switch Check: Exit immediately if system is disabled
    if (stopSignalRef.current || !poseLandmarkerRef.current || !videoRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");

    if (canvas && video.videoWidth > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const results = poseLandmarkerRef.current.detectForVideo(video, performance.now());

    // Visualization: Draw the skeleton
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const drawingUtils = new DrawingUtils(ctx);
      if (results.landmarks) {
        for (const landmarks of results.landmarks) {
          drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 3 });
          drawingUtils.drawLandmarks(landmarks, { radius: 2, color: '#FF0000' });
        }
      }
    }

    if (results.worldLandmarks?.length > 0) {
      const wl = results.worldLandmarks[0];
      const avg = (a: number, b: number) => ({ x: (wl[a].x + wl[b].x)/2, y: (wl[a].y + wl[b].y)/2, z: (wl[a].z + wl[b].z)/2 });

      // Build the features for this frame
      const f: FallFeatures = {
        torsoLegAngle: calculateTorsoLegAngle(avg(11,12), avg(23,24), avg(25,26)),
        kneeAnkleDistance: Math.abs(avg(25,26).y - avg(27,28).y),
        headFloorDistance: Math.abs(wl[0].y - avg(29,30).y),
        headAngle: (Math.atan2(wl[0].y - avg(11,12).y, wl[0].x - avg(11,12).x) * 180) / Math.PI,
        noseAnkleDistance: Math.sqrt(Math.pow(wl[0].x-avg(27,28).x,2)+Math.pow(wl[0].y-avg(27,28).y,2)),
        aspectRatio: Math.abs(wl[27].x - wl[12].x) / Math.abs(wl[0].y - avg(29,30).y),
      };

      // BUFFER MANAGEMENT: Store the results of this frame
      frameBufferRef.current.push(buildFrameVote(f));
      if (frameBufferRef.current.length > MAX_BUFFER) frameBufferRef.current.shift();

      // Calculation: Weighted confidence score over the buffer
      const bufferScore = frameBufferRef.current.reduce((t, v) => t + (Object.keys(FEATURE_WEIGHTS) as Array<keyof FrameVote>).reduce((st, k) => st + (v[k] ? FEATURE_WEIGHTS[k] : 0), 0), 0);
      let curConf = bufferScore / (MAX_SCORE_PER_FRAME * frameBufferRef.current.length);

      // Anti-Kneeling Guard: Slash confidence if the user is upright
      if (f.aspectRatio < 0.85) curConf *= 0.3;

      setConfidence(curConf);
      if (curConf >= FALL_THRESHOLD) {
        setFallDetected(true);
        setIsFalseAlarm(false);
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "VISION_CONFIRMED" }));
        }
      }
    }
    
    // Only schedule the next frame if the stop signal is FALSE
    if (!stopSignalRef.current) {
      window.requestAnimationFrame(predictFall);
    }
  };

  // Initialize MediaPipe once
  useEffect(() => {
    FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm").then(v => {
      PoseLandmarker.createFromOptions(v, { baseOptions: { modelAssetPath: "/pose_landmarker_full.task", delegate: "GPU" }, runningMode: "VIDEO" })
      .then(p => poseLandmarkerRef.current = p);
    });
  }, []);

  return (
    <div className="flex flex-col items-center gap-4 p-4 bg-black min-h-screen text-white font-mono">
      <div className="w-full max-w-4xl border-2 border-blue-900 rounded-xl p-6 bg-gray-900 shadow-2xl">
        
        {/* Status Headers */}
        <div className="flex justify-between text-xs mb-4 uppercase tracking-widest">
          <div className="flex gap-4">
            <span>Fall Confidence: <span className={confidence > 0.5 ? 'text-red-400' : 'text-blue-400'}>{Math.round(confidence * 100)}%</span></span>
            {fallDetected && <span>Down Time: {fallDurationCounter}s</span>}
          </div>
          <span className={wsStatus === 'connected' ? 'text-green-400' : 'text-red-500'}>Status: {wsStatus}</span>
        </div>

        {/* Dynamic Warning Banners */}
        {hardwareAlert && !fallDetected && !isFalseAlarm && (
          <div className="bg-yellow-500 text-black text-center p-2 font-bold mb-2 animate-pulse rounded">
            ⚡ SENSOR TRIGGER: ANALYZING CAMERA...
          </div>
        )}
        {fallDetected && (
          <div className="bg-red-600 text-white text-center p-2 font-bold text-lg mb-2 rounded shadow-lg border-2 border-red-400">
            🚨 EMERGENCY: FALL DETECTED ({fallDurationCounter}s)
          </div>
        )}
        {isFalseAlarm && (
          <div className="bg-blue-600 text-white text-center p-2 font-bold text-lg mb-2 rounded border-2 border-blue-400 ">
            ✅ FALSE ALARM: RECOVERY CONFIRMED
          </div>
        )}
        {recoveryCounter > 0 && (
          <div className="bg-green-700 text-center p-1 text-xs mb-2 italic rounded">
            Stabilization detected. Auto-clearing alert in {10 - recoveryCounter}s...
          </div>
        )}

        {/* The Viewport */}
        <div className="relative aspect-video bg-black rounded-lg overflow-hidden border-4 border-gray-800 shadow-inner">
          <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover opacity-60" autoPlay muted playsInline />
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
        </div>

        {/* Controls */}
        <button 
          onClick={() => isPredicting ? handleStop() : handleStart()} 
          className={`mt-6 w-full p-4 rounded-lg font-bold transition-all transform active:scale-95 ${isPredicting ? 'bg-red-900 hover:bg-red-800' : 'bg-blue-700 hover:bg-blue-600'}`}
        >
          {isPredicting ? "DEACTIVATE MONITORING" : "INITIALIZE POSE ENGINE"}
        </button>
      </div>
    </div>
  );
};

export default PoseEngine;