import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Camera, Brain, BarChart3, ScanFace } from "lucide-react";

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-800 text-white relative overflow-hidden">

      {/* 🌌 AURORA GLOW */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute w-[700px] h-[700px] bg-indigo-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]"></div>
        <div className="absolute w-[600px] h-[600px] bg-purple-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]"></div>
        <div className="absolute w-[500px] h-[500px] bg-cyan-400 opacity-20 blur-[140px] top-[40%] left-[40%]"></div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-12 relative z-10">

        {/* 🔥 NAVBAR */}
        <div className="flex justify-between items-center mb-16">

          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-cyan-500 rounded-xl flex items-center justify-center shadow-md">
              <ScanFace size={20} />
            </div>
            <span className="font-semibold text-lg">Face Attendance</span>
          </div>

          

        </div>

        {/* HERO */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-16">

          {/* LEFT */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="max-w-xl"
          >
            <h1 className="text-5xl md:text-6xl font-bold leading-tight bg-gradient-to-r from-white via-gray-300 to-gray-400 bg-clip-text text-transparent">
              AI-Powered Face Attendance System
            </h1>

            <p className="mt-6 text-lg text-gray-300 leading-relaxed">
              Smart attendance powered by{" "}
              <span className="text-indigo-400 font-semibold">
                Artificial Intelligence & Facial Recognition
              </span>.
              Detect students instantly and generate reports in seconds.
            </p>

            {/* 🔥 TRUST LINE */}
            

            {/* BUTTONS */}
            <div className="flex gap-5 mt-10">

              <motion.button
                whileHover={{ scale: 1.08 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => navigate("/role")}
                className="px-8 py-3 rounded-xl 
                bg-gradient-to-r from-indigo-500 to-cyan-500 
                hover:from-indigo-400 hover:to-cyan-400 
                text-white shadow-lg shadow-indigo-500/30 transition-all"
              >
                Get Started
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.08 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => navigate("/role")}
                className="px-8 py-3 rounded-xl border border-white/20 hover:bg-white/10 backdrop-blur-lg transition-all"
              >
                Login
              </motion.button>

            </div>
          </motion.div>

          {/* RIGHT AI SCAN */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7 }}
            className="relative w-80 h-80 flex items-center justify-center"
          >

            <div className="absolute w-full h-full rounded-full border border-white/10"></div>

            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
              className="absolute w-full h-full border-2 border-indigo-400/40 rounded-full"
            />

            <motion.div
              animate={{ y: ["-120px", "120px"] }}
              transition={{ repeat: Infinity, duration: 2 }}
              className="absolute w-full h-[2px] bg-green-400 shadow-[0_0_15px_#4ade80]"
            />

            <div className="w-52 h-52 rounded-full border border-white/20 backdrop-blur-md"></div>

            <motion.div
              animate={{ scale: [1, 1.5, 1] }}
              transition={{ repeat: Infinity, duration: 2 }}
              className="absolute w-4 h-4 bg-green-400 rounded-full"
            />
          </motion.div>
        </div>

        {/* FEATURES */}
        <div className="grid md:grid-cols-3 gap-8 mt-28">

          {[{
            icon: <Camera size={30} />,
            title: "Automatic Face Detection",
            desc: "Detect multiple faces in real-time using advanced computer vision."
          },
          {
            icon: <Brain size={30} />,
            title: "AI Recognition",
            desc: "Deep learning identifies students with high precision."
          },
          {
            icon: <BarChart3 size={30} />,
            title: "Analytics Dashboard",
            desc: "Track attendance trends with smart visual insights."
          }].map((item, i) => (

            <motion.div
              key={i}
              whileHover={{ scale: 1.06, y: -6 }}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.2 }}
              className="bg-white/5 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-lg hover:shadow-indigo-500/20 transition-all"
            >
              <div className="p-3 rounded-xl bg-indigo-500/20 w-fit text-indigo-400">
                {item.icon}
              </div>

              <h3 className="text-xl font-semibold mt-4">
                {item.title}
              </h3>

              <p className="text-gray-400 mt-2 text-sm">
                {item.desc}
              </p>
            </motion.div>

          ))}
        </div>

        {/* FOOTER */}
        <div className="text-center text-gray-500 mt-20 text-sm">
          © 2026 Face Attendance System • AI Powered
        </div>

      </div>
    </div>
  );
}