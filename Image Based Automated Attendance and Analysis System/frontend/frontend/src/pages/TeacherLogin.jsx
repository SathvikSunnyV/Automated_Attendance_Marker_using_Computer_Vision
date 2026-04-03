import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { useState } from "react";
import API from "../api/api";
import toast from "react-hot-toast";

export default function TeacherLogin() {

  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  async function login() {
  try {

    setLoading(true);

    const res = await API.post("/login-teacher", {
      email: email.trim(),
      password: password
    });

    console.log("LOGIN RESPONSE:", res.data);  // 🔥 DEBUG

    if (res.data.access_token) {

      // ✅ HANDLE BOTH id / teacher_id
      const teacherId = res.data.id || res.data.teacher_id;

      if (!teacherId) {
        toast.error("Login failed: No teacher ID returned");
        return;
      }

      localStorage.setItem("teacher_id", teacherId);
      localStorage.setItem("teacher_name", res.data.name);
      localStorage.setItem("token", res.data.access_token);

      toast.success("Welcome 👋");

      navigate("/teacher/dashboard");

    } else {
      toast.error("Invalid login credentials");
    }

  } catch (err) {
    console.error(err);
    toast.error("Login failed");
  } finally {
    setLoading(false);
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
        initial={{ opacity: 0, y: 40, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="bg-white/5 backdrop-blur-2xl p-10 rounded-3xl border border-white/10 shadow-xl w-80 flex flex-col gap-5"
      >

        <h2 className="text-2xl font-bold text-center">
          Teacher Login
        </h2>

        {/* EMAIL */}
        <input
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="p-3 rounded-lg bg-white/10 border border-white/20 outline-none focus:ring-2 focus:ring-indigo-500 transition"
        />

        {/* PASSWORD */}
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="p-3 rounded-lg bg-white/10 border border-white/20 outline-none focus:ring-2 focus:ring-indigo-500 transition"
        />

        {/* BUTTON */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={login}
          disabled={loading}
          className="bg-indigo-600 hover:bg-indigo-500 transition text-white p-3 rounded-lg mt-2 shadow-lg shadow-indigo-500/30 flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              Logging in...
            </>
          ) : (
            "Login"
          )}
        </motion.button>

        {/* REGISTER LINK */}
        <p className="text-xs text-gray-400 text-center mt-2">
          Don’t have an account?{" "}
          <span
            onClick={() => navigate("/teacher/register")}
            className="text-indigo-400 cursor-pointer hover:underline"
          >
            Register
          </span>
        </p>

      </motion.div>
    </div>
  );
}