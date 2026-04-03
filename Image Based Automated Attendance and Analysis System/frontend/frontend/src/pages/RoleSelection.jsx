import { motion } from "framer-motion";
import { GraduationCap, School, Shield } from "lucide-react";
import { useNavigate } from "react-router-dom";


export default function RoleSelection() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-800 relative overflow-hidden text-white">

      {/* 🌌 AURORA BACKGROUND */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute w-[700px] h-[700px] bg-indigo-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]"></div>
        <div className="absolute w-[600px] h-[600px] bg-purple-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]"></div>
        <div className="absolute w-[500px] h-[500px] bg-cyan-400 opacity-20 blur-[140px] top-[40%] left-[40%]"></div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="grid md:grid-cols-3 gap-12 p-10"
      >

        {/* 🎓 STUDENT CARD */}
        <motion.div
          whileHover={{ scale: 1.06, y: -8 }}
          transition={{ type: "spring", stiffness: 200 }}
          className="w-80 p-10 rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 shadow-xl hover:shadow-indigo-500/20"
        >
          <div className="flex flex-col items-center text-center">

            <div className="p-4 rounded-full bg-indigo-500/10 mb-4">
              <GraduationCap size={40} className="text-indigo-400" />
            </div>

            <h2 className="text-2xl font-semibold mt-2">Student</h2>

            <p className="text-gray-400 mt-3 text-sm leading-relaxed">
              Register your face and track attendance analytics with AI insights.
            </p>

            <div className="flex gap-4 mt-8">

              <motion.button
                whileHover={{ scale: 1.08 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => navigate("/student/register")}
                className="px-6 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 transition shadow-md shadow-indigo-500/30"
              >
                Register
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.08 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => navigate("/student/login")}
                className="px-6 py-2 rounded-lg border border-white/20 hover:bg-white/10 backdrop-blur-lg transition"
              >
                Login
              </motion.button>

            </div>
          </div>
        </motion.div>

        {/* 👨‍🏫 TEACHER CARD */}
        <motion.div
          whileHover={{ scale: 1.06, y: -8 }}
          transition={{ type: "spring", stiffness: 200 }}
          className="w-80 p-10 rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 shadow-xl hover:shadow-cyan-500/20"
        >
          <div className="flex flex-col items-center text-center">

            <div className="p-4 rounded-full bg-cyan-500/10 mb-4">
              <School size={40} className="text-cyan-400" />
            </div>

            <h2 className="text-2xl font-semibold mt-2">Teacher</h2>

            <p className="text-gray-400 mt-3 text-sm leading-relaxed">
              Upload classroom photos and manage attendance with AI automation.
            </p>

            <div className="flex gap-4 mt-8">

              <motion.button
                whileHover={{ scale: 1.08 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => navigate("/teacher/register")}
                className="px-6 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 transition shadow-md shadow-cyan-500/30"
              >
                Register
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.08 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => navigate("/teacher/login")}
                className="px-6 py-2 rounded-lg border border-white/20 hover:bg-white/10 backdrop-blur-lg transition"
              >
                Login
              </motion.button>

            </div>
          </div>
        </motion.div>

        {/* 👨‍💼 ADMIN CARD */}
<motion.div
  whileHover={{ scale: 1.06, y: -8 }}
  transition={{ type: "spring", stiffness: 200 }}
  className="w-80 p-10 rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 shadow-xl hover:shadow-amber-500/20"
>
  <div className="flex flex-col items-center text-center">

    <div className="p-4 rounded-full bg-amber-500/10 mb-4">
      <Shield size={40} className="text-amber-400" />
    </div>

    <h2 className="text-2xl font-semibold mt-2">Admin</h2>

    <p className="text-gray-400 mt-3 text-sm leading-relaxed">
      Approve teacher requests and manage system access.
    </p>

    <div className="flex gap-4 mt-8">

      <motion.button
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => navigate("/admin/login")}
        className="px-6 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 transition shadow-md shadow-amber-500/30"
      >
        Login
      </motion.button>

    </div>
  </div>
</motion.div>

      </motion.div>
    </div>
  );
}