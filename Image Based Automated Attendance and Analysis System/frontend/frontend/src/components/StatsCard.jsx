import { motion } from "framer-motion";

export default function StatsCard({ title, value }) {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      className="p-6 rounded-xl bg-white/5 backdrop-blur-lg border border-white/10 shadow-lg"
    >
      <p className="text-gray-400 text-sm">{title}</p>
      <h2 className="text-3xl font-bold text-white mt-2">{value}</h2>
    </motion.div>
  );
}