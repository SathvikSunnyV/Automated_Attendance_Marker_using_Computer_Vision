import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { motion } from "framer-motion";
import { Shield } from "lucide-react";

export default function AdminLogin() {

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const navigate = useNavigate();

  const handleLogin = async () => {

    try {

      const res = await axios.post(
        "http://localhost:8000/admin/login",
        {
          email,
          password
        }
      );

      localStorage.setItem("admin_token", res.data.access_token);

      navigate("/admin/dashboard");

    } catch (err) {
      alert("Invalid admin credentials");
    }

  };

  return (

    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-800 relative overflow-hidden text-white">

      {/* 🌌 AURORA BACKGROUND */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute w-[700px] h-[700px] bg-indigo-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]" />
        <div className="absolute w-[600px] h-[600px] bg-purple-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]" />
        <div className="absolute w-[500px] h-[500px] bg-cyan-400 opacity-20 blur-[140px] top-[40%] left-[40%]" />
      </div>


      <motion.div
        initial={{ opacity:0, y:40 }}
        animate={{ opacity:1, y:0 }}
        transition={{ duration:0.6 }}
        className="w-96 p-10 rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 shadow-xl"
      >

        <div className="flex flex-col items-center text-center">

          <motion.div
            whileHover={{ scale:1.1 }}
            className="p-4 rounded-full bg-amber-500/10 mb-4"
          >
            <Shield size={40} className="text-amber-400"/>
          </motion.div>

          <h2 className="text-2xl font-semibold mt-2">
            Admin Login
          </h2>

          <p className="text-gray-400 mt-2 text-sm">
            Manage teacher approvals & system access
          </p>

        </div>


        <div className="mt-8 space-y-4">

          <motion.input
            whileFocus={{ scale:1.02 }}
            type="email"
            placeholder="Admin Email"
            className="w-full p-3 rounded-lg bg-white/5 border border-white/10 focus:outline-none focus:border-amber-400"
            value={email}
            onChange={(e)=>setEmail(e.target.value)}
          />


          <motion.input
            whileFocus={{ scale:1.02 }}
            type="password"
            placeholder="Password"
            className="w-full p-3 rounded-lg bg-white/5 border border-white/10 focus:outline-none focus:border-amber-400"
            value={password}
            onChange={(e)=>setPassword(e.target.value)}
          />


          <motion.button
            whileHover={{ scale:1.05 }}
            whileTap={{ scale:0.95 }}
            onClick={handleLogin}
            className="w-full py-3 rounded-lg bg-gradient-to-r from-amber-500 to-yellow-500 font-semibold shadow-lg hover:shadow-amber-500/30 transition"
          >
            Login
          </motion.button>

        </div>

      </motion.div>

    </div>

  );

}