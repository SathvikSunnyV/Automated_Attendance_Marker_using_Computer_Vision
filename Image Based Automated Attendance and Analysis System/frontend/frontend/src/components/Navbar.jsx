import { motion } from "framer-motion";

export default function Navbar() {
  return (
    <motion.div
      initial={{ y: -60 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.4 }}
      className="w-full flex justify-between items-center px-10 py-4 backdrop-blur-lg bg-white/5 border-b border-white/10"
    >
      <h1 className="text-xl font-bold text-white">
        Smart Attendance
      </h1>

      <div className="flex gap-6 text-gray-300">
        <button className="hover:text-white transition">Home</button>
        <button className="hover:text-white transition">Student</button>
        <button className="hover:text-white transition">Teacher</button>
      </div>
    </motion.div>
  );
}