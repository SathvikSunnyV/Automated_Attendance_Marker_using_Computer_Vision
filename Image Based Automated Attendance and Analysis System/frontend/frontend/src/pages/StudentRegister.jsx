import { useRef, useState } from "react";
import Webcam from "react-webcam";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import API from "../api/api";
import toast from "react-hot-toast";

export default function StudentRegister() {

  const webcamRef = useRef(null);
  const navigate = useNavigate();

  const [images, setImages] = useState([]);
  const [uploadedImages, setUploadedImages] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState(0);

  const [name, setName] = useState("");
  const [roll, setRoll] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [otp, setOtp] = useState("");
  const [otpSent, setOtpSent] = useState(false);
  const [otpVerified, setOtpVerified] = useState(false);

  const MAX_IMAGES = 5;

  const captureFrame = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      setImages(prev => [...prev, imageSrc]);
    }
  };

  const startScan = () => {
    setImages([]);
    setProgress(0);
    setScanning(true);

    let count = 0;

    const interval = setInterval(() => {
      captureFrame();
      count++;

      setProgress((count / MAX_IMAGES) * 100);

      if (count >= MAX_IMAGES) {
        clearInterval(interval);
        setScanning(false);
      }
    }, 500);
  };

  const base64ToBlob = async (base64) => {
    const res = await fetch(base64);
    return await res.blob();
  };

  async function sendOtp() {
  try {

    await API.post("/send-otp", null, {
      params: {
        email: email,
        role: "student"
      }
    });

    toast.success("OTP sent to email");
    setOtpSent(true);

  } catch (err) {
    toast.error(err.response?.data?.detail || "OTP failed");
  }
}


async function verifyOtp() {
  try {

    await API.post("/verify-otp", null, {
      params: {
        email: email,
        otp: otp
      }
    });

    toast.success("OTP verified");
    setOtpVerified(true);

  } catch (err) {
    toast.error("Invalid OTP");
  }
}

  async function registerStudent() {
  try {

    const formData = new FormData();

    // ✅ student details
    formData.append("name", name);
    formData.append("roll", roll);
    formData.append("email", email);
    formData.append("password", password);

    // ✅ webcam images
    for (let i = 0; i < images.length; i++) {
      const blob = await base64ToBlob(images[i]);
      formData.append("images", blob, `webcam${i}.jpg`);
    }

    // ✅ uploaded images
    for (let i = 0; i < uploadedImages.length; i++) {
      formData.append("images", uploadedImages[i]);
    }

    // 🔥 SINGLE API CALL
    await API.post("/register-complete", formData, {
      headers: { "Content-Type": "multipart/form-data" }
    });

    toast.success("Registration successful 🎉");
    navigate("/student/login");

  } catch (err) {

    console.error(err.response?.data || err.message);

    if (err.response?.data?.detail === "Face not detected properly") {
      toast.error("❌ Face not clear. Use proper lighting & front face");
    } else {
      toast.error(err.response?.data?.detail || "Registration failed");
    }

  }
}
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-800 relative overflow-hidden text-white">

      {/* 🌌 Aurora Background */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute w-[700px] h-[700px] bg-indigo-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]"></div>
        <div className="absolute w-[600px] h-[600px] bg-purple-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]"></div>
        <div className="absolute w-[500px] h-[500px] bg-cyan-400 opacity-20 blur-[140px] top-[40%] left-[40%]"></div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/5 backdrop-blur-2xl border border-white/10 p-10 rounded-3xl shadow-xl flex flex-col md:flex-row gap-12"
      >

        {/* 🧾 FORM */}
        <div className="flex flex-col gap-4 w-72">

          <h2 className="text-3xl font-bold mb-2">
            Student Registration
          </h2>

          {[{
            placeholder: "Name",
            setter: setName
          },
          {
            placeholder: "Roll Number",
            setter: setRoll
          },
          {
            placeholder: "Email",
            setter: setEmail
          }].map((field, i) => (

            <input
              key={i}
              placeholder={field.placeholder}
              onChange={(e) => field.setter(e.target.value)}
              className="p-3 rounded-lg bg-white/10 border border-white/20 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition"
            />

          ))}

                      {/* Send OTP */}
            {email && !otpSent && (
              <button
                onClick={sendOtp}
                className="bg-indigo-600 p-2 rounded-lg text-sm"
              >
                Send OTP
              </button>
            )}

            {otpSent && !otpVerified && (
  <div className="flex gap-2">

    <input
      placeholder="Enter OTP"
      value={otp}
      onChange={(e) => setOtp(e.target.value)}
      className="p-2 rounded-lg bg-white/10 border border-white/20"
    />

    <button
      onClick={verifyOtp}
      className="bg-green-500 px-3 rounded-lg"
    >
      Verify
    </button>

  </div>
)}

          <input
            type="password"
            placeholder="Password"
            onChange={(e) => setPassword(e.target.value)}
            className="p-3 rounded-lg bg-white/10 border border-white/20 focus:ring-2 focus:ring-indigo-500 transition"
          />

          {/* Upload */}
          <label className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl 
bg-gradient-to-r from-indigo-500/20 to-purple-500/20 
border border-dashed border-white/30 cursor-pointer 
hover:scale-[1.02] hover:border-indigo-400 transition-all">

  <span className="text-sm text-gray-300 text-center">
  {uploadedImages.length > 0
    ? uploadedImages.length === 1
      ? uploadedImages[0].name   // ✅ show single file name
      : `${uploadedImages.length} files selected`
    : "Upload Face Images"}
</span>

  

  <input
    type="file"
    multiple
    accept="image/*"
    className="hidden"
    onChange={(e) => {
      const files = Array.from(e.target.files);

      if (files.length > 5) {
        toast.error("Max 5 images");
        return;
      }

      setUploadedImages(files);
    }}
  />
</label>

          <p className="text-xs text-gray-400">
            Use webcam OR upload images
          </p>

          {/* Register */}
          {otpVerified &&
(images.length === MAX_IMAGES || uploadedImages.length > 0) &&
            name && roll && email && password && (

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={registerStudent}
                className="bg-green-500 hover:bg-green-600 p-3 rounded-lg mt-2 shadow-lg shadow-green-500/30"
              >
                Register Student
              </motion.button>

            )}

        </div>

        {/* 📷 CAMERA */}
        <div className="flex flex-col items-center">

          <div className="relative w-72 h-72 flex items-center justify-center">

            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              className="rounded-full w-full h-full object-cover border-4 border-indigo-500/40 shadow-[0_0_40px_rgba(99,102,241,0.6)]"
            />

            {/* Circular Progress */}
            <svg className="absolute w-full h-full rotate-[-90deg]">
              <circle
                cx="144"
                cy="144"
                r="130"
                stroke="rgba(255,255,255,0.1)"
                strokeWidth="8"
                fill="transparent"
              />
              <motion.circle
                cx="144"
                cy="144"
                r="130"
                stroke="#22c55e"
                strokeWidth="8"
                fill="transparent"
                strokeDasharray={2 * Math.PI * 130}
                strokeDashoffset={
                  2 * Math.PI * 130 * (1 - progress / 100)
                }
                transition={{ duration: 0.3 }}
              />
            </svg>

          </div>

          <p className="mt-4 font-semibold text-gray-300">
            Scanning Face {Math.round(progress)}%
          </p>

          {!scanning && images.length === 0 && (
            <motion.button
              whileHover={{ scale: 1.08 }}
              whileTap={{ scale: 0.95 }}
              onClick={startScan}
              className="mt-4 bg-indigo-600 hover:bg-indigo-500 px-6 py-2 rounded-lg shadow-md shadow-indigo-500/30"
            >
              Start Scan
            </motion.button>
          )}

        </div>

      </motion.div>

    </div>
  );
}