import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import { useEffect, useRef, useState } from "react";

// Types
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
    aspectRatio:  boolean;
 }

 const BUFFER_SIZE = 20;
//  Weights which per each feature based on their importance. for typing, object where key is type FrameVote and value is of type number
 const FEATURE_WEIGHTS: Record<keyof FrameVote, number> = {
  torsoLegAngle:     3,
  headFloorDistance: 2,
  noseAnkleDistance: 2,
  aspectRatio:       2,
  kneeAnkleDistance: 1,
  headAngle:         1,
};

const MAX_SCORE_PER_FRAME = Object.values(FEATURE_WEIGHTS).reduce((a, b) => a + b, 0); // = 11; reduce acts as a sum where zero is a and the values in the array are b
const MAX_BUFFER_SCORE = MAX_SCORE_PER_FRAME * BUFFER_SIZE; // = 220
const FALL_THRESHOLD = 0.6;
// Feature extraction formulas:
// Clamp cosine to [-1, 1] to prevent floating point errors, then convert to degrees. can return null because magA or magB can be 0 and so return null
//Angle between torso (shoulder→hip) and leg (hip→knee) vectors — fall range: 70°–110°

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

// Vertical distance between knee and ankle — small value means legs are flat/collapsed
const calculateKneeAnkleDistance = (
  avgKnee: landmark,
  avgAnkle: landmark,
): number => {
  return Math.abs(avgKnee.y - avgAnkle.y);
};

// Vertical distance between nose and heel — small value means head is near the ground
const calculateHeadFloorDistance = (nose: landmark, avgHeel: landmark): number => {
    return Math.abs(nose.y - avgHeel.y);
};
// not totally reliable: check the nose and shoulder y landmarks on vertical and horizontal orientation
// const calculateUpperBodyAlignment = (
//   nose: landmark,
//   avgShoulder: landmark,
// ) => {
//     return avgShoulder.y <= nose.y;
// };

// Tilt angle of head relative to shoulders — changes dramatically when fallen
const calculateHeadAngle = (nose: landmark, avgShoulder: landmark): number => {
    // angle = arctan((nose.y - shoulder.y) / (nose.x - shoulder.x)) × (180 / π)
    return (Math.atan2(nose.y - avgShoulder.y, nose.x - avgShoulder.x) * 180) / Math.PI;
};

// 3D distance from nose to ankle — collapses when body goes from vertical to horizontal
const calculateNoseAnkleDistance = (nose: landmark, avgAnkle: landmark): number => {
   const dx = nose.x - avgAnkle.x;
   const dy = nose.y - avgAnkle.y;
   const dz = nose.z - avgAnkle.z;

    return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

// Ratio of body width to height — large value means body is wider than tall, indicating a fall
const bodyAspectRatio = (
  leftAnkle: landmark,
  rightShoulder: landmark,
  nose: landmark,
  avgHeel: landmark
): number => {
  const horizontalSpread = Math.abs(leftAnkle.x - rightShoulder.x);
  const verticalHeight = Math.abs(nose.y - avgHeel.y);
  if (verticalHeight === 0) return 0;
  return horizontalSpread / verticalHeight;
};

const buildFrameVote = (features: FallFeatures): FrameVote => ({
    torsoLegAngle: (features.torsoLegAngle ?? 180) >= 70 && (features.torsoLegAngle ?? 180) <= 110,
    kneeAnkleDistance: (features.kneeAnkleDistance ?? 1) < 0.05,
    headFloorDistance: (features.headFloorDistance ?? 1) < 0.15,
    headAngle: features.headAngle < 45 || features.headAngle > 135,
    noseAnkleDistance: features.noseAnkleDistance < 0.3,
    aspectRatio: (features.aspectRatio) > 1.0
});

// scoreFrame calculates the vote per frame depending on each feature
const scoreFrame = (vote: FrameVote): number =>
  (Object.keys(FEATURE_WEIGHTS) as Array<keyof FrameVote>).reduce(
    (total, key) => total + (vote[key] ? FEATURE_WEIGHTS[key] : 0),
    0
  );

//   buffer.length is the total number of frames.
// evealuateBuffer determines the total number of votes across n number of frames
  const evaluateBuffer = (
  buffer: FrameVote[]
): { isFall: boolean; confidence: number } => {
  if (buffer.length === 0) return { isFall: false, confidence: 0 };
  const bufferScore = buffer.reduce((total, frame) => total + scoreFrame(frame), 0);
  const confidence = bufferScore / MAX_BUFFER_SCORE;
  return { isFall: confidence >= FALL_THRESHOLD, confidence };
};

const PoseEngine = () => {
  const [isPredicting, setIsPredicting] = useState(false); //isPredicting is the state that sets
  const [confidence, setConfidence] = useState(0);
  const [fallDetected, setFallDetected] = useState(false);

  // We use useref for the video and canvas elements because they change in each frame and we don't want it to trigger a rerender
  const videoRef = useRef<HTMLVideoElement | null>(null); //holds reference to actual video DOM element
  const canvasRef = useRef<HTMLCanvasElement | null>(null); //holds reference to actual canvas DOM element
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const isTracking = useRef(false);

//   const prevFrameRef = useRef<PrevFrameData | null>(null);
  const frameBufferRef = useRef<FrameVote[]>([]); //array that stores the last 20 frame votes
  //   Clears the canvas/skeleton
  const clearCanvas = (ctx: CanvasRenderingContext2D) => {
    ctx.clearRect(0, 0, canvasRef.current!.width, canvasRef.current!.height);
  };

  const predictFall = () => {
    console.log("predictFall called, isTracking:", isTracking.current);
    if (!isTracking.current) return;
    // we can change the values of videoRef and canvasRef in this function because it would be changed on an event
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const poseLandmarker = poseLandmarkerRef.current;

    if (!video || !canvas || !poseLandmarker) {
      console.log("missing ref:", { video, canvas, poseLandmarker });
      return;
    }
    // const canvasCtx = canvas.getContext("2d");

    // check if canvas width and video width are unequal, if so, change
    console.log("Video width: ", video.width);
    console.log("Videowidth: ", video.videoWidth);
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const startTimeMs = performance.now();
    console.log("about to detect, videoWidth:", video.videoWidth); // ← add this
    const poseLandmarkerResults = poseLandmarker.detectForVideo(
      video,
      startTimeMs,
    );
    console.log("results:", poseLandmarkerResults);
    const canvasCtx = canvas.getContext("2d");
    if (canvasCtx) {
      // canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      clearCanvas(canvasCtx);
      const drawingUtils = new DrawingUtils(canvasCtx);
      if (poseLandmarkerResults.landmarks) {
        // landmarks are for canvas: relative based on camera
        for (const landmarks of poseLandmarkerResults.landmarks) {
          drawingUtils.drawConnectors(
            landmarks,
            PoseLandmarker.POSE_CONNECTIONS,
          );
          //    drawingUtils.drawLandmarks(landmarks);
          drawingUtils.drawLandmarks(landmarks, {
            radius: (data) => DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1),
          });
        }
      }
    }

    // use world landmarks for fall detection: real-world 3d coordinates in meters. uses hips as center. MAKE SURE HIPS ARE SEEN
    //is their array empty?
    if (poseLandmarkerResults.worldLandmarks?.length > 0) {
      const wl = poseLandmarkerResults.worldLandmarks[0];

      // parts used in detection
      const nose = wl[0];
      const leftShoulder = wl[11];
      const rightShoulder = wl[12];
      const leftHip = wl[23];
      const rightHip = wl[24];
      const leftKnee = wl[25];
      const rightKnee = wl[26];
      const leftAnkle = wl[27];
      const rightAnkle = wl[28];
      const leftHeel = wl[29];
      const rightHeel = wl[30];

      // calculate average: implicit return
      const avg = (a: landmark, b: landmark) => ({
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
        z: (a.z + b.z) / 2,
      });

      const avgShoulder = avg(leftShoulder, rightShoulder);
      const avgHip = avg(leftHip, rightHip);
      const avgKnee = avg(leftKnee, rightKnee);
      const avgAnkle = avg(leftAnkle, rightAnkle);
      const avgHeel = avg(leftHeel, rightHeel);

      const extractedFeatures: FallFeatures = {
        torsoLegAngle: calculateTorsoLegAngle(avgShoulder, avgHip, avgKnee),
        kneeAnkleDistance: calculateKneeAnkleDistance(avgKnee, avgAnkle),
        headFloorDistance: calculateHeadFloorDistance(nose, avgHeel),
        headAngle: calculateHeadAngle(nose, avgShoulder),
        noseAnkleDistance: calculateNoseAnkleDistance(nose, avgAnkle),
        aspectRatio: bodyAspectRatio(leftAnkle, rightShoulder, nose, avgHeel)

      };
      

      // Step 1 — convert features to a frame vote
const vote = buildFrameVote(extractedFeatures);

// Step 2 — push vote into buffer, drop oldest if full
frameBufferRef.current.push(vote);
if (frameBufferRef.current.length > 20) {
  frameBufferRef.current.shift();
}

// Step 3 — evaluate the buffer (you'll write this next)
const { isFall, confidence: currentConfidence } = evaluateBuffer(frameBufferRef.current);


// Debug: log feature values and whether each threshold fired
console.table({
  torsoLegAngle: { value: extractedFeatures.torsoLegAngle?.toFixed(2),    fired: vote.torsoLegAngle },
  kneeAnkle:     { value: extractedFeatures.kneeAnkleDistance.toFixed(3), fired: vote.kneeAnkleDistance },
  headFloor:     { value: extractedFeatures.headFloorDistance.toFixed(3), fired: vote.headFloorDistance },
  headAngle:     { value: extractedFeatures.headAngle.toFixed(1),         fired: vote.headAngle },
  noseAnkle:     { value: extractedFeatures.noseAnkleDistance.toFixed(3), fired: vote.noseAnkleDistance },
  aspectRatio:   { value: extractedFeatures.aspectRatio.toFixed(3),       fired: vote.aspectRatio },
  confidence:    { value: (currentConfidence * 100).toFixed(1) + "%",     fired: isFall },
});


setFallDetected(isFall);
setConfidence(currentConfidence);
    }

    window.requestAnimationFrame(predictFall);
  };

  const handleButtonPress = async () => {
    if (isTracking.current) {
      isTracking.current = false;
      frameBufferRef.current = [];
setFallDetected(false);
setConfidence(0);
      // To stop camera
    //   if (videoRef.current?.srcObject) {
    //     const stream = videoRef.current.srcObject as MediaStream; //Get the stream. MediaStream is the type bc getUseMedia always returns it
    //     stream.getTracks().forEach((track) => track.stop()); // Get each track on the stream(audio and video) and stop
    //     videoRef.current.srcObject = null; //cleanup
    //   }
    //   const canvasCtx = canvasRef.current?.getContext("2d");
    //   if (canvasCtx) {
    //     clearCanvas(canvasCtx);
    //   }

      setIsPredicting(false);
    } else {
      try {
        // to access the webcam:
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        }); //getUserMedia({video:true}) is what asks user for permission to use webcam
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          console.log("srcObject set");
          videoRef.current.onloadeddata = () => {
            console.log("onloadeddata fired");
            videoRef.current!.play();
            isTracking.current = true;
            setIsPredicting(true);
            predictFall();
          };
        }
      } catch (error) {
        console.error("Error accessing webcam: ", error);
      }
    }
  };
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
          numPoses: 1, //number of bodies it detects: 1 because hypothetically 1. in system, we are only monitoring one person as they have the dashboard. a second isn't in the system, as the wearer and the monitored are linked.
        },
      );
    };
    createPoseLandmarker();
  }, []);
  return (
    <div className="">
      <div className="max-w-7xl w-full flex justify-center items-center flex-col gap-2 p-3">
        {/* Container for heading text */}
        <div>
          <h2 className="text-center text-blue-400 font-mono">
            Pose Engine Ready ...
          </h2>
        
<div className="text-xs text-white font-mono bg-gray-900 px-3 py-1">
  Confidence: {Math.round(confidence * 100)}% | 
  Buffer: {frameBufferRef.current.length}/{BUFFER_SIZE}
    {fallDetected && (
  <div className="bg-red-600 text-white text-center py-2 font-bold text-lg">
    ⚠️ FALL DETECTED — Confidence: {Math.round(confidence * 100)}%
  </div>
  
)}
</div>
        </div>
        {/* Container for video and canvas */}
        <div className="relative max-w-full  w-full h-[90vh] bg-pink-300 border-2 border-blue-100 rounded-2xl">
          <video
            className="absolute top-0 left-0 w-full h-full object-contain"
            ref={videoRef}
            autoPlay
            playsInline
          ></video>
          <canvas
            className="absolute top-0 left-0 w-full h-full object-contain"
            ref={canvasRef}
          ></canvas>
        </div>
        {/* Container containing button */}
        <div>
          <button
            className=" bg-blue-950 border-blue-300 rounded-xl p-4"
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
