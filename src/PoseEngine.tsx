import { DrawingUtils, FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import { useEffect, useRef, useState } from "react";

// ── TYPES & INTERFACES ─────────────────────────────────────────
interface landmark { x: number; y: number; z: number; visibility?: number; }
interface FallFeatures {
  torsoLegAngle: number | null;
  kneeAnkleDistance: number;
  headFloorDistance: number;
  headAngle: number;
  noseAnkleDistance: number;
  aspectRatio: number;
}
interface FrameVote {
  torsoLegAngle: boolean;
  kneeAnkleDistance: boolean;
  headFloorDistance: boolean;
  headAngle: boolean;
  noseAnkleDistance: boolean;
  aspectRatio: boolean;
}
// ── I-SYNC INTEGRATION ─────────────────────────────────────────
// When a fall is confirmed, this sends an alert to the I-Sync server.
// The I-Sync server then broadcasts to the caregiver dashboard,
// triggers the patient's 10-second countdown, and sends an emergency SMS.
// !! Replace ISYNC_PATIENT_ID with the patient's ID from the I-Sync profile screen !!
const ISYNC_SERVER     = "https://i-sync-ai.onrender.com";
const ISYNC_PATIENT_ID = "PAT-HEWR25L";

const sendIsyncFallAlert = () => {
  fetch(`${ISYNC_SERVER}/fall`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patientId: ISYNC_PATIENT_ID }),
  })
    .then(() => console.log("[I-Sync] Fall alert sent successfully."))
    .catch(err => console.error("[I-Sync] Fall alert failed:", err));
};
// ──────────────────────────────────────────────────────────────

// ── CONFIGURATION ──────────────────────────────────────────────
const MIN_BUFFER = 8;
const MAX_BUFFER = 40;
const FALL_THRESHOLD = 0.6;
const RECOVERY_THRESHOLD = 0.35;
const WS_URL = import.meta.env.VITE_WS_URL;

// ── I-SYNC BRIDGE ──────────────────────────────────────────────
// When loaded inside the I-Sync WebView, window.__isInsideISync is true.
// Use ReactNativeWebView.postMessage instead of WebSocket in that case.
const isInsideISync = (): boolean =>
  typeof window !== "undefined" && !!(window as any).ReactNativeWebView;

// const sendToISync = (type: string, confidence = 0, feature = "") => {
//   if (!isInsideISync()) return;
//   (window as any).ReactNativeWebView?.postMessage(
//     JSON.stringify({ type, confidence, feature })
//   );
// };

// ── DYNAMIC BUFFER ─────────────────────────────────────────────
const getDynamicBufferSize = (confidence: number): number => {
  const t = Math.min(1, confidence / FALL_THRESHOLD);
  const size = MIN_BUFFER + (MAX_BUFFER - MIN_BUFFER) * (t * t);
  return Math.round(size);
};

// ── WEIGHTED VOTING ────────────────────────────────────────────
const FEATURE_WEIGHTS: Record<keyof FrameVote, number> = {
  torsoLegAngle: 3,
  headFloorDistance: 2,
  noseAnkleDistance: 2,
  aspectRatio: 2,
  kneeAnkleDistance: 1,
  headAngle: 1,
};
const MAX_SCORE_PER_FRAME = Object.values(FEATURE_WEIGHTS).reduce((a, b) => a + b, 0);

// ── FEATURE CALCULATIONS ───────────────────────────────────────
const calculateTorsoLegAngle = (s: landmark, h: landmark, k: landmark): number | null => {
  const ax = h.x - s.x, ay = h.y - s.y;
  const bx = k.x - h.x, by = k.y - h.y;
  const dot = ax * bx + ay * by;
  const magA = Math.sqrt(ax * ax + ay * ay);
  const magB = Math.sqrt(bx * bx + by * by);
  if (magA === 0 || magB === 0) return null;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (magA * magB)))) * 180) / Math.PI;
};

const buildFrameVote = (f: FallFeatures): FrameVote => ({
  torsoLegAngle:
    ((f.torsoLegAngle ?? 180) >= 70 && (f.torsoLegAngle ?? 180) <= 110) ||
    (f.torsoLegAngle ?? 180) < 30,
  kneeAnkleDistance: f.kneeAnkleDistance < 0.25,
  headFloorDistance: f.headFloorDistance < 0.35,
  headAngle: f.headAngle < 45 || f.headAngle > 135,
  noseAnkleDistance: f.noseAnkleDistance < 1.3,
  aspectRatio: f.aspectRatio > 1.2,
});

const scoreFrame = (vote: FrameVote): number =>
  (Object.keys(FEATURE_WEIGHTS) as Array<keyof FrameVote>).reduce(
    (total, key) => total + (vote[key] ? FEATURE_WEIGHTS[key] : 0),
    0,
  );

const evaluateBuffer = (buffer: FrameVote[]): { isFall: boolean; confidence: number } => {
  if (buffer.length === 0) return { isFall: false, confidence: 0 };
  const bufferScore = buffer.reduce((total, frame) => total + scoreFrame(frame), 0);
  const confidence = bufferScore / (MAX_SCORE_PER_FRAME * buffer.length);
  return { isFall: confidence >= FALL_THRESHOLD, confidence };
};

// ── COMPONENT ──────────────────────────────────────────────────
const PoseEngine = () => {
  // ── STATE ────────────────────────────────────────────────────
  const [isPredicting, setIsPredicting] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [fallDetected, setFallDetected] = useState(false);
  const [isFalseAlarm, setIsFalseAlarm] = useState(false);
  const [hardwareAlert, setHardwareAlert] = useState(false);
  const [wsStatus, setWsStatus] = useState<"connecting" | "connected" | "disconnected">("disconnected");
  const [recoveryCounter, setRecoveryCounter] = useState(0);
  const [fallDurationCounter, setFallDurationCounter] = useState(0);

  // ── REFS ─────────────────────────────────────────────────────
  const modelReadyRef = useRef(false);
const isPredictingRef = useRef(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const frameBufferRef = useRef<FrameVote[]>([]);
  const stopSignalRef = useRef(false);
  const confidenceRef = useRef(0);
  const targetSizeRef = useRef(MIN_BUFFER);
  const recoveryTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const twilioTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Prevents the I-Sync alert firing on every frame once a fall is detected.
  // Resets when the fall clears so the next real fall sends a fresh alert.
  const isyncAlertSentRef = useRef(false);

  // const iSyncHandledRef = useRef(false); // tracks if I-Sync already handled this fall

  // ── 1. AUTO-START + READY HANDSHAKE (I-Sync only) ─────────────
  // useEffect(() => {
  //   if (!isInsideISync()) return;
  //   sendToISync("POSE_ENGINE_READY");
  //   handleStart();
  // }, []);
useEffect(() => {
  isPredictingRef.current = isPredicting;
}, [isPredicting]);
  // ── 2. LISTEN FOR MESSAGES FROM I-SYNC ───────────────────────
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      try {
        const msg = typeof e.data === "string" ? JSON.parse(e.data) : e.data;

        if (msg.type === "APP_CONFIRMED") {
          // I-Sync countdown expired or user tapped Confirm — fall already handled by app
          // iSyncHandledRef.current = true;
          setFallDetected(false);
          frameBufferRef.current = [];
        }

        if (msg.type === "APP_FALSE_ALARM") {
          // I-Sync user tapped "I'm OK"
          // iSyncHandledRef.current = true;
          setFallDetected(false);
          setIsFalseAlarm(true);
          frameBufferRef.current = [];
        }

        if (msg.type === "FALL_CANCELLED") {
          // I-Sync cancelled mid-countdown (e.g. patient pressed cancel before timer ended)
          // iSyncHandledRef.current = true;
          setFallDetected(false);
          setIsFalseAlarm(false);
          frameBufferRef.current = [];
        }
      } catch {}
    };

    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, []);
  

  // ── 3. TWILIO ALERT TIMER (fires after 12s — standalone mode only) ─
  useEffect(() => {
    // if (isInsideISync()) return; // I-Sync handles its own SMS
    if (fallDetected && !isFalseAlarm) {
      if (!twilioTimerRef.current) {
        twilioTimerRef.current = setInterval(() => {
          setFallDurationCounter(prev => {
            if (prev >= 11) {
              console.log("Sending Twilio SMS via WebSocket...");
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: "SEND_TWILIO_SMS" }));
              }
              clearInterval(twilioTimerRef.current!);
              twilioTimerRef.current = null;
              return 12;
            }
            return prev + 1;
          });
        }, 1000);
      }
    } else {
      if (twilioTimerRef.current) {
        clearInterval(twilioTimerRef.current);
        twilioTimerRef.current = null;
      }
      setFallDurationCounter(0);
    }
  }, [fallDetected, isFalseAlarm]);

  // ── 4. RECOVERY TIMER (clears fall after 10s of low confidence) ─
  useEffect(() => {
    if (fallDetected && confidence < RECOVERY_THRESHOLD) {
      if (!recoveryTimerRef.current) {
        recoveryTimerRef.current = setInterval(() => {
          setRecoveryCounter(p => {
            if (p >= 9) {
              setIsFalseAlarm(true);
              setFallDetected(false);
              setHardwareAlert(false);
              frameBufferRef.current = [];
              isyncAlertSentRef.current = false; // reset so next fall sends a fresh alert
              if (recoveryTimerRef.current) {
                clearInterval(recoveryTimerRef.current);
                recoveryTimerRef.current = null;
              }
              return 0;
            }
            return p + 1;
          });
        }, 1000);
      }
    } else {
      if (recoveryTimerRef.current) {
        clearInterval(recoveryTimerRef.current);
        recoveryTimerRef.current = null;
      }
      setRecoveryCounter(0);
      if (confidence > FALL_THRESHOLD) setIsFalseAlarm(false);
    }
  }, [confidence, fallDetected]);

  // ── 5. WEBSOCKET (standalone mode only — skipped inside I-Sync) ─
  useEffect(() => {
    // if (isInsideISync()) return; // no WebSocket needed inside I-Sync
    const connect = () => {
      setWsStatus("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket connected.");
        setWsStatus("connected");
      };

      ws.onmessage = (e) => {
  try {
    const d = JSON.parse(e.data);
    if (d.type === "FALL_DETECTED") {
      console.log("Hardware fall alert received — starting vision check.");
      setHardwareAlert(true);
      setIsFalseAlarm(false);
      if (!isPredictingRef.current) handleStart(); // use ref, not state
    }
  } catch (err) {
    console.error("WebSocket message parse error:", err);
  }
};

      ws.onerror = (e) => console.error("WebSocket error:", e);

      ws.onclose = () => {
        setWsStatus("disconnected");
        setTimeout(connect, 3000);
      };
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  // ── 6. MEDIAPIPE INITIALISATION ───────────────────────────────
 useEffect(() => {
  if (poseLandmarkerRef.current) return;
  FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  ).then(vision =>
    PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "/pose_landmarker_full.task",
        delegate: "CPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    })
  ).then(p => {
    poseLandmarkerRef.current = p;
    modelReadyRef.current = true;
    console.log("PoseLandmarker ready.");
    // If camera was already started before model finished loading, start loop now
    if (isPredictingRef.current) {
      requestAnimationFrame(() => predictFall());
    }
  });
}, []);

  // ── 7. CAMERA CONTROLS ────────────────────────────────────────
 const handleStart = async () => {
   console.log("handleStart called", {
    stopSignal: stopSignalRef.current,
    isPredicting: isPredictingRef.current,
    modelReady: modelReadyRef.current,
  });
  if (!stopSignalRef.current && isPredictingRef.current) return;
  stopSignalRef.current = false;
  frameBufferRef.current = [];
  confidenceRef.current = 0;
  targetSizeRef.current = MIN_BUFFER;
  isyncAlertSentRef.current = false;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadeddata = () => {
        videoRef.current!.play().then(() => {
          setIsPredicting(true);
          isPredictingRef.current = true;
          // Only start loop if model is already ready
          // If not ready, the MediaPipe init above will start it
          if (modelReadyRef.current) {
            requestAnimationFrame(() => predictFall());
          }
        });
      };
    }
  } catch (err) {
    console.error("Camera access error:", err);
  }
};

  const handleStop = () => {
    stopSignalRef.current = true;
    setIsPredicting(false);
    isPredictingRef.current = false;
    setConfidence(0);
    setFallDetected(false);
    setIsFalseAlarm(false);
    setHardwareAlert(false);
    frameBufferRef.current = [];
    confidenceRef.current = 0;
    targetSizeRef.current = MIN_BUFFER;
    isyncAlertSentRef.current = false; // reset on stop
    // iSyncHandledRef.current = false;

    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }

    const ctx = canvasRef.current?.getContext("2d");
    ctx?.clearRect(0, 0, canvasRef.current?.width || 0, canvasRef.current?.height || 0);
  };

  // ── 8. THE DETECTION LOOP ─────────────────────────────────────
  const predictFall = () => {
    if (stopSignalRef.current || !poseLandmarkerRef.current || !videoRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");

    if (canvas && video.videoWidth > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const results = poseLandmarkerRef.current.detectForVideo(video, performance.now());

    // Draw skeleton overlay
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const drawingUtils = new DrawingUtils(ctx);
      if (results.landmarks) {
        for (const landmarks of results.landmarks) {
          drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: "#00FF00",
            lineWidth: 3,
          });
          drawingUtils.drawLandmarks(landmarks, { radius: 2, color: "#FF0000" });
        }
      }
    }

    if (results.worldLandmarks?.length > 0) {
      const wl = results.worldLandmarks[0];

      const avg = (a: number, b: number): landmark => ({
        x: (wl[a].x + wl[b].x) / 2,
        y: (wl[a].y + wl[b].y) / 2,
        z: (wl[a].z + wl[b].z) / 2,
      });

      const avgShoulder = avg(11, 12);
      const avgHip      = avg(23, 24);
      const avgKnee     = avg(25, 26);
      const avgAnkle    = avg(27, 28);
      const avgHeel     = avg(29, 30);
      const nose        = wl[0];

      const f: FallFeatures = {
        torsoLegAngle:     calculateTorsoLegAngle(avgShoulder, avgHip, avgKnee),
        kneeAnkleDistance: Math.abs(avgKnee.y - avgAnkle.y),
        headFloorDistance: Math.abs(nose.y - avgHeel.y),
        headAngle:         (Math.atan2(nose.y - avgShoulder.y, nose.x - avgShoulder.x) * 180) / Math.PI,
        noseAnkleDistance: Math.sqrt(
          Math.pow(nose.x - avgAnkle.x, 2) +
          Math.pow(nose.y - avgAnkle.y, 2) +
          Math.pow(nose.z - avgAnkle.z, 2)
        ),
        aspectRatio: Math.abs(wl[27].x - wl[12].x) / Math.abs(nose.y - avgHeel.y) || 0,
      };

      const targetSize = getDynamicBufferSize(confidenceRef.current);
      targetSizeRef.current = targetSize;
      frameBufferRef.current.push(buildFrameVote(f));
      while (frameBufferRef.current.length > targetSize) {
        frameBufferRef.current.shift();
      }

      const { confidence: curConf, isFall } = evaluateBuffer(frameBufferRef.current);
      const finalConf = f.aspectRatio < 0.85 ? curConf * 0.3 : curConf;
      confidenceRef.current = finalConf;
      setConfidence(finalConf);

      // if (isFall && finalConf >= FALL_THRESHOLD && !iSyncHandledRef.current) {
      //   setFallDetected(true);
      //   setIsFalseAlarm(false);
      
      if (isFall && finalConf >= FALL_THRESHOLD) {
  setFallDetected(true);
  setIsFalseAlarm(false);

  // Find the highest-weighted feature that voted true in the last frame
  const lastFrame = frameBufferRef.current.at(-1);
  const topFeature = lastFrame
    ? (Object.keys(FEATURE_WEIGHTS) as Array<keyof FrameVote>)
        .filter(k => lastFrame[k])
        .sort((a, b) => FEATURE_WEIGHTS[b] - FEATURE_WEIGHTS[a])[0] ?? ""
    : "";

  if (!isyncAlertSentRef.current) {
    isyncAlertSentRef.current = true;

    // Send to React Native app via WebView bridge
    if ((window as any).ReactNativeWebView) {
      (window as any).ReactNativeWebView.postMessage(
        JSON.stringify({ type: "FALL_DETECTED", confidence: finalConf, feature: topFeature })
      );
    } else {
      // Standalone fallback — WebSocket + I-Sync server
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "VISION_CONFIRMED" }));
      }
      sendIsyncFallAlert();
    }
  }
}

    }

    if (!stopSignalRef.current) {
      window.requestAnimationFrame(predictFall);
    }
  };

  // ── RENDER ────────────────────────────────────────────────────
  const wsColor = wsStatus === "connected"
    ? "text-green-400"
    : wsStatus === "connecting"
    ? "text-yellow-400"
    : "text-red-500";

  return (
    <div className="flex flex-col items-center gap-4 p-4 bg-black min-h-screen text-white font-mono">
      <div className="w-full max-w-4xl border-2 border-blue-900 rounded-xl p-6 bg-gray-900 shadow-2xl">

        {/* Status bar */}
        <div className="flex justify-between text-xs mb-4 uppercase tracking-widest">
          <div className="flex gap-4">
            <span>
              Fall Confidence:{" "}
              <span className={confidence > 0.5 ? "text-red-400" : "text-blue-400"}>
                {Math.round(confidence * 100)}%
              </span>
            </span>
            <span>
              Buffer: {frameBufferRef.current.length}/{targetSizeRef.current}
            </span>
            {fallDetected && <span>Down Time: {fallDurationCounter}s</span>}
          </div>
          {<span className={wsColor}>● {wsStatus}</span>}
          {/* {!isInsideISync() && <span className={wsColor}>● {wsStatus}</span>} */}
        </div>

        {/* Alert banners */}
        {hardwareAlert && !fallDetected && !isFalseAlarm && (
          <div className="bg-yellow-500 text-black text-center p-2 font-bold mb-2 animate-pulse rounded">
            ⚡ SENSOR TRIGGER — ANALYZING CAMERA...
          </div>
        )}
        {fallDetected && (
          <div className="bg-red-600 text-white text-center p-2 font-bold text-lg mb-2 rounded shadow-lg border-2 border-red-400">
            🚨 EMERGENCY: FALL DETECTED ({fallDurationCounter}s)
            {/* {!isInsideISync() && fallDurationCounter < 12 && (
              <div className="text-sm font-normal mt-1">
                Alert sending in {12 - fallDurationCounter}s — press stop if false alarm
              </div>
            )} */}
            {fallDurationCounter < 12 && (
              <div className="text-sm font-normal mt-1">
                Alert sending in {12 - fallDurationCounter}s — press stop if false alarm
              </div>
            )}
            {/* {isInsideISync() && (
              <div className="text-sm font-normal mt-1">
                Waiting for I-Sync confirmation...
              </div>
            )} */}
          </div>
        )}
        {isFalseAlarm && (
          <div className="bg-blue-600 text-white text-center p-2 font-bold text-lg mb-2 rounded border-2 border-blue-400">
            ✅ FALSE ALARM — RECOVERY CONFIRMED
          </div>
        )}
        {recoveryCounter > 0 && fallDetected && (
          <div className="bg-green-700 text-center p-1 text-xs mb-2 italic rounded">
            Stabilization detected. Auto-clearing in {10 - recoveryCounter}s...
          </div>
        )}

        {/* Video viewport */}
        <div className="relative aspect-video bg-black rounded-lg overflow-hidden border-4 border-gray-800 shadow-inner">
          <video
            ref={videoRef}
            className="absolute inset-0 w-full h-full object-cover opacity-60"
            autoPlay
            muted
            playsInline
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
          />
        </div>

        {/* Control button — hidden inside I-Sync (camera auto-starts) */}
        {/* {!isInsideISync() && (
          <button
            onClick={() => isPredicting ? handleStop() : handleStart()}
            className={`mt-6 w-full p-4 rounded-lg font-bold transition-all transform active:scale-95 ${
              isPredicting
                ? "bg-red-900 hover:bg-red-800"
                : "bg-blue-700 hover:bg-blue-600"
            }`}
          >
            {isPredicting ? "DEACTIVATE MONITORING" : "INITIALIZE POSE ENGINE"}
          </button>
        )} */}
        
          <button
            onClick={() => isPredicting ? handleStop() : handleStart()}
            className={`mt-6 w-full p-4 rounded-lg font-bold transition-all transform active:scale-95 ${
              isPredicting
                ? "bg-red-900 hover:bg-red-800"
                : "bg-blue-700 hover:bg-blue-600"
            }`}
          >
            {isPredicting ? "DEACTIVATE MONITORING" : "INITIALIZE POSE ENGINE"}
          </button>
       

      </div>
    </div>
  );
};

export default PoseEngine;