import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import { useEffect, useRef, useState } from "react";

// ── Types ─────────────────────────────────────────────────────
interface landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

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

// ── Constants ─────────────────────────────────────────────────
const MIN_BUFFER = 8;
const MAX_BUFFER = 40;
const FALL_THRESHOLD = 0.6;
const WS_URL = "ws://10.0.1.25:8080"; // ← replace with your PC's IP

const getDynamicBufferSize = (confidence: number): number => {
  const t = Math.min(1, confidence / FALL_THRESHOLD);
  const size = MIN_BUFFER + (MAX_BUFFER - MIN_BUFFER) * (t * t);
  return Math.round(size);
};

const FEATURE_WEIGHTS: Record<keyof FrameVote, number> = {
  torsoLegAngle: 3,
  headFloorDistance: 2,
  noseAnkleDistance: 2,
  aspectRatio: 2,
  kneeAnkleDistance: 1,
  headAngle: 1,
};

const MAX_SCORE_PER_FRAME = Object.values(FEATURE_WEIGHTS).reduce(
  (a, b) => a + b,
  0,
);

// ── Feature calculations ──────────────────────────────────────
const calculateTorsoLegAngle = (
  shoulder: landmark,
  hip: landmark,
  knee: landmark,
): number | null => {
  const ax = hip.x - shoulder.x;
  const ay = hip.y - shoulder.y;
  const bx = knee.x - hip.x;
  const by = knee.y - hip.y;
  const dot = ax * bx + ay * by;
  const magA = Math.sqrt(ax * ax + ay * ay);
  const magB = Math.sqrt(bx * bx + by * by);
  if (magA === 0 || magB === 0) return null;
  const cosAngle = Math.min(1, Math.max(-1, dot / (magA * magB)));
  return (Math.acos(cosAngle) * 180) / Math.PI;
};

const calculateKneeAnkleDistance = (
  avgKnee: landmark,
  avgAnkle: landmark,
): number => Math.abs(avgKnee.y - avgAnkle.y);

const calculateHeadFloorDistance = (
  nose: landmark,
  avgHeel: landmark,
): number => Math.abs(nose.y - avgHeel.y);

const calculateHeadAngle = (nose: landmark, avgShoulder: landmark): number =>
  (Math.atan2(nose.y - avgShoulder.y, nose.x - avgShoulder.x) * 180) / Math.PI;

const calculateNoseAnkleDistance = (
  nose: landmark,
  avgAnkle: landmark,
): number => {
  const dx = nose.x - avgAnkle.x;
  const dy = nose.y - avgAnkle.y;
  const dz = nose.z - avgAnkle.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

const bodyAspectRatio = (
  leftAnkle: landmark,
  rightShoulder: landmark,
  nose: landmark,
  avgHeel: landmark,
): number => {
  const horizontalSpread = Math.abs(leftAnkle.x - rightShoulder.x);
  const verticalHeight = Math.abs(nose.y - avgHeel.y);
  if (verticalHeight === 0) return 0;
  return horizontalSpread / verticalHeight;
};

const buildFrameVote = (features: FallFeatures): FrameVote => ({
  torsoLegAngle:
    ((features.torsoLegAngle ?? 180) >= 70 &&
      (features.torsoLegAngle ?? 180) <= 110) ||
    (features.torsoLegAngle ?? 180) < 30,
  kneeAnkleDistance: (features.kneeAnkleDistance ?? 1) < 0.25,
  headFloorDistance: (features.headFloorDistance ?? 1) < 0.4,
  headAngle: features.headAngle < 45 || features.headAngle > 135,
  noseAnkleDistance: features.noseAnkleDistance < 1.5,
  aspectRatio: features.aspectRatio > 1.0,
});

const scoreFrame = (vote: FrameVote): number =>
  (Object.keys(FEATURE_WEIGHTS) as Array<keyof FrameVote>).reduce(
    (total, key) => total + (vote[key] ? FEATURE_WEIGHTS[key] : 0),
    0,
  );

const evaluateBuffer = (
  buffer: FrameVote[],
): { isFall: boolean; confidence: number } => {
  if (buffer.length === 0) return { isFall: false, confidence: 0 };
  const bufferScore = buffer.reduce(
    (total, frame) => total + scoreFrame(frame),
    0,
  );
  const confidence = bufferScore / (MAX_SCORE_PER_FRAME * buffer.length);
  return { isFall: confidence >= FALL_THRESHOLD, confidence };
};

// ── Component ─────────────────────────────────────────────────
const PoseEngine = () => {
  const [isPredicting, setIsPredicting] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [fallDetected, setFallDetected] = useState(false);
  const [hardwareAlert, setHardwareAlert] = useState(false); // ← new: tracks ESP32 alert
  const [wsStatus, setWsStatus] = useState<"connecting" | "connected" | "disconnected">("disconnected");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const isTracking = useRef(false);
  const frameBufferRef = useRef<FrameVote[]>([]);
  const targetSizeRef = useRef(MIN_BUFFER);

  const clearCanvas = (ctx: CanvasRenderingContext2D) => {
    ctx.clearRect(0, 0, canvasRef.current!.width, canvasRef.current!.height);
  };

  // ── Start camera & prediction ────────────────────────────
  const startPrediction = async () => {
    if (isTracking.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadeddata = () => {
          videoRef.current!.play();
          isTracking.current = true;
          setIsPredicting(true);
          predictFall();
        };
      }
    } catch (error) {
      console.error("Error accessing webcam:", error);
    }
  };

  // ── Stop camera & prediction ─────────────────────────────
  const stopPrediction = () => {
    isTracking.current = false;
    frameBufferRef.current = [];
    setFallDetected(false);
    setHardwareAlert(false);
    setConfidence(0);
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    const canvasCtx = canvasRef.current?.getContext("2d");
    if (canvasCtx) clearCanvas(canvasCtx);
    targetSizeRef.current = MIN_BUFFER;
    setIsPredicting(false);
  };

  const handleButtonPress = async () => {
    if (isTracking.current) {
      stopPrediction();
    } else {
      await startPrediction();
    }
  };

  // ── WebSocket connection to relay server ─────────────────
  // ── WebSocket connection to relay server ─────────────────
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout>;

    const connect = () => {
      console.log("Attempting to connect to WebSocket...");
      setWsStatus("connecting");
      
      ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log("✅ WebSocket connected to relay server");
        setWsStatus("connected");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "FALL_DETECTED") {
            console.log("🚨 Hardware fall alert received! Starting vision check...");
            setHardwareAlert(true);
            
            // Auto-start camera if not already running
            if (!isTracking.current) {
              startPrediction();
            }
          }
        } catch (e) {
          console.error("❌ WebSocket message parse error:", e);
        }
      };

      ws.onerror = (e) => {
        console.error("⚠️ WebSocket error occurred", e);
        setWsStatus("disconnected");
      };

      ws.onclose = (e) => {
        // e.code 1000 is a "normal closure" (like when the component unmounts)
        if (e.code !== 1000) {
          console.log("🔌 WebSocket closed unexpectedly. Retrying in 3s...");
          setWsStatus("disconnected");
          reconnectTimeout = setTimeout(connect, 3000);
        } else {
          console.log("🔌 WebSocket closed normally.");
        }
      };
    };

    connect();

    // ── Cleanup function: Runs when the component unmounts ──
    return () => {
      if (ws) {
        ws.close(1000, "Component unmounted");
      }
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
    };
  }, []);
  // useEffect(() => {
  //   setWsStatus("connecting");
  //   const ws = new WebSocket(WS_URL);

  //   ws.onopen = () => {
  //     console.log("WebSocket connected to relay server");
  //     setWsStatus("connected");
  //   };

  //   ws.onmessage = (event) => {
  //     try {
  //       const data = JSON.parse(event.data);
  //       if (data.type === "FALL_DETECTED") {
  //         console.log("Hardware fall alert received — starting vision check...");
  //         setHardwareAlert(true);
  //         // Auto-start camera if not already running
  //         if (!isTracking.current) {
  //           startPrediction();
  //         }
  //       }
  //     } catch (e) {
  //       console.error("WebSocket message parse error:", e);
  //     }
  //   };

  //   ws.onerror = (e) => {
  //     console.error("WebSocket error:", e);
  //     setWsStatus("disconnected");
  //   };

  //   ws.onclose = () => {
  //     console.log("WebSocket disconnected");
  //     setWsStatus("disconnected");
  //   };

  //   return () => ws.close();
  // }, []);

  // ── Pose landmarker initialisation ───────────────────────
  useEffect(() => {
    if (poseLandmarkerRef.current) return;
    const createPoseLandmarker = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
      );
      poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath: "/pose_landmarker_full.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
        },
      );
    };
    createPoseLandmarker();
  }, []);

  // ── Frame prediction loop ────────────────────────────────
  const predictFall = () => {
    if (!isTracking.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const poseLandmarker = poseLandmarkerRef.current;
    if (!video || !canvas || !poseLandmarker) return;

    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const startTimeMs = performance.now();
    const poseLandmarkerResults = poseLandmarker.detectForVideo(
      video,
      startTimeMs,
    );

    const canvasCtx = canvas.getContext("2d");
    if (canvasCtx) {
      clearCanvas(canvasCtx);
      const drawingUtils = new DrawingUtils(canvasCtx);
      if (poseLandmarkerResults.landmarks) {
        for (const landmarks of poseLandmarkerResults.landmarks) {
          drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS);
          drawingUtils.drawLandmarks(landmarks, {
            radius: (data) => DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1),
          });
        }
      }
    }

    if (poseLandmarkerResults.worldLandmarks?.length > 0) {
      const wl = poseLandmarkerResults.worldLandmarks[0];

      const nose          = wl[0];
      const leftShoulder  = wl[11];
      const rightShoulder = wl[12];
      const leftHip       = wl[23];
      const rightHip      = wl[24];
      const leftKnee      = wl[25];
      const rightKnee     = wl[26];
      const leftAnkle     = wl[27];
      const rightAnkle    = wl[28];
      const leftHeel      = wl[29];
      const rightHeel     = wl[30];

      const avg = (a: landmark, b: landmark) => ({
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
        z: (a.z + b.z) / 2,
      });

      const avgShoulder = avg(leftShoulder, rightShoulder);
      const avgHip      = avg(leftHip, rightHip);
      const avgKnee     = avg(leftKnee, rightKnee);
      const avgAnkle    = avg(leftAnkle, rightAnkle);
      const avgHeel     = avg(leftHeel, rightHeel);

      const extractedFeatures: FallFeatures = {
        torsoLegAngle:     calculateTorsoLegAngle(avgShoulder, avgHip, avgKnee),
        kneeAnkleDistance: calculateKneeAnkleDistance(avgKnee, avgAnkle),
        headFloorDistance: calculateHeadFloorDistance(nose, avgHeel),
        headAngle:         calculateHeadAngle(nose, avgShoulder),
        noseAnkleDistance: calculateNoseAnkleDistance(nose, avgAnkle),
        aspectRatio:       bodyAspectRatio(leftAnkle, rightShoulder, nose, avgHeel),
      };

      const vote = buildFrameVote(extractedFeatures);
      frameBufferRef.current.push(vote);

      const targetSize = getDynamicBufferSize(confidence);
      targetSizeRef.current = targetSize;
      while (frameBufferRef.current.length > targetSize) {
        frameBufferRef.current.shift();
      }

      const { isFall, confidence: currentConfidence } = evaluateBuffer(
        frameBufferRef.current,
      );

      setFallDetected(isFall);
      setConfidence(currentConfidence);

      // Clear hardware alert once vision confirms or denies
      if (hardwareAlert && currentConfidence > 0.1) {
        setHardwareAlert(false);
      }
    }

    window.requestAnimationFrame(predictFall);
  };

  // ── Status indicator colour ──────────────────────────────
  const wsStatusColor = {
    connected: "text-green-400",
    connecting: "text-yellow-400",
    disconnected: "text-red-400",
  }[wsStatus];

  // ── Render ───────────────────────────────────────────────
  return (
    <div className="">
      <div className="max-w-7xl w-full flex justify-center items-center flex-col gap-2 p-3">

        {/* Header / status bar */}
        <div className="w-full">
          <h2 className="text-center text-blue-400 font-mono">
            Pose Engine Ready ...
          </h2>

          <div className="text-xs text-white font-mono bg-gray-900 px-3 py-1 flex items-center justify-between flex-wrap gap-1">
            <span>
              Confidence: {Math.round(confidence * 100)}% | Buffer:{" "}
              {frameBufferRef.current.length}/{targetSizeRef.current}
            </span>
            <span className={wsStatusColor}>
              ● Server: {wsStatus}
            </span>
          </div>

          {/* Hardware alert banner */}
          {hardwareAlert && !fallDetected && (
            <div className="bg-yellow-500 text-black text-center py-2 font-bold text-sm">
              ⚡ Hardware fall detected — verifying with camera...
            </div>
          )}

          {/* Vision confirmed fall banner */}
          {fallDetected && (
            <div className="bg-red-600 text-white text-center py-2 font-bold text-lg">
              ⚠️ FALL CONFIRMED — Hardware + Vision —{" "}
              {Math.round(confidence * 100)}% confidence
            </div>
          )}
        </div>

        {/* Video + canvas */}
        <div className="relative max-w-full w-full h-[90vh] bg-black border-2 border-blue-100 rounded-2xl">
          <video
            className="absolute top-0 left-0 w-full h-full object-contain"
            ref={videoRef}
            autoPlay
            playsInline
          />
          <canvas
            className="absolute top-0 left-0 w-full h-full object-contain"
            ref={canvasRef}
          />
        </div>

        {/* Button */}
        <div>
          <button
            className="bg-blue-950 border-blue-300 rounded-xl p-4"
            onClick={handleButtonPress}
          >
            <p className="text-blue-100">
              {!isPredicting ? "Enable Prediction" : "Disable Prediction"}
            </p>
          </button>
        </div>

      </div>
    </div>
  );
};

export default PoseEngine;