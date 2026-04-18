import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import { useEffect, useRef, useState } from "react";

const PoseEngine = () => {
  const [isPredicting, setIsPredicting] = useState(false); //isPredicting is the state that sets

  // We use useref for the video and canvas elements because they change in each frame and we don't want it to trigger a rerender
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const isTracking = useRef(false);

  const handleButtonPress = async () => {
    if (isTracking.current) {
      isTracking.current = false;
      // To stop camera
      const stream = videoRef.current.srcObject as MediaStream; //Get the stream. MediaStream is the type bc getUseMedia always returns it
      stream.getTracks().forEach((track) => track.stop()); // Get each track on the stream(audio and video) and stop
      videoRef.current.srcObject = null; //cleanup
      setIsPredicting(false);
    } else {
      try {
        // to access the webcam:
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        }); //getUserMedia({video:true}) is what asks user for permission to use webcam
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadeddata = () => {
            videoRef.current!.play();
            isTracking.current = true;
            setIsPredicting(true);
          };
        }
      } catch (error) {
        console.error("Error accessing webcam: ", error);
      }
    }
  };
  useEffect(() => {
    const createPoseLandmarker = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
      );
      poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath: "/pose_landmarker_lite.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 2,
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
          <h2 className="text-center text-blue-400">Pose Engine Ready ...</h2>
        </div>
        {/* Container for video and canvas */}
        <div className="max-w-full  w-full h-[90vh] bg-pink-300 border-2 border-blue-100 rounded-2xl">
          <video
            className="w-screen h-full scale-x-[-1]"
            ref={videoRef}
            autoPlay
            playsInline
          ></video>
          <canvas ref={canvasRef}></canvas>
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
