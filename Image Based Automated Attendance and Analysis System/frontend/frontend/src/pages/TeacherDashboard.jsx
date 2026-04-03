import { motion } from "framer-motion"
import { useState, useEffect } from "react"
import API from "../api/api"
import TeacherSidebar from "../components/TeacherSidebar"
import toast from "react-hot-toast"

import {
  Users,
  CalendarDays,
  CheckCircle,
  Clock,
  AlertTriangle
} from "lucide-react"

import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from "recharts"

const COLORS = ["#22c55e", "#38bdf8"]

export default function TeacherDashboard() {

  const [stats, setStats] = useState({})
  const [classes, setClasses] = useState([])
  const [insights, setInsights] = useState([])
  const [todayChart, setTodayChart] = useState([])
  const [weeklyChart, setWeeklyChart] = useState([])
  const [low, setLow] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboard()
    const interval = setInterval(loadDashboard, 30000)
    return () => clearInterval(interval)
  }, [])

  async function loadDashboard() {
    try {

      const [
        statsRes,
        classesRes,
        insightsRes,
        chartRes,
        lowRes
      ] = await Promise.all([
        API.get("/teacher/dashboard"),
        API.get("/classes/today"),
        API.get("/teacher/insights"),
        API.get("/teacher/chart-data"),
        API.get("/teacher/low-attendance")
      ])

      setStats(statsRes.data)
      setClasses(classesRes.data)
      setInsights(insightsRes.data.insights)
      setTodayChart(chartRes.data.today)
      setWeeklyChart(chartRes.data.weekly)
      setLow(lowRes.data)

    } catch (err) {
      console.error(err)
      toast.error("Failed to load dashboard")
    } finally {
      setLoading(false)
    }
  }

  function getClassStatus(cls) {
    return cls?.attendance_taken
      ? { text: "Completed", style: "bg-green-500/20 text-green-300" }
      : { text: "Pending", style: "bg-yellow-500/20 text-yellow-300" }
  }

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-800 text-white relative overflow-hidden">

      {/* glow */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute w-[700px] h-[700px] bg-blue-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]"></div>
        <div className="absolute w-[600px] h-[600px] bg-cyan-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]"></div>
      </div>

      <TeacherSidebar />

      <div className="flex-1 p-10">

        <h1 className="text-4xl font-bold mb-10 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
          Teacher Dashboard
        </h1>

        {/* STATS */}
        <div className="grid md:grid-cols-4 gap-6 mb-10">
          <StatCard icon={<Users size={26} />} title="Students" value={stats.students} />
          <StatCard icon={<CalendarDays size={26} />} title="Today Classes" value={stats.classes_today} />
          <StatCard icon={<CheckCircle size={26} />} title="Completed" value={stats.attendance_taken} />
          <StatCard icon={<Clock size={26} />} title="Pending" value={stats.pending} />
        </div>

        {/* CHARTS */}
        <div className="grid md:grid-cols-2 gap-6 mb-10">

          <motion.div className="bg-white/5 backdrop-blur-2xl p-6 rounded-3xl border border-white/10 shadow-lg">
            <h2 className="mb-4 font-semibold">Today Status</h2>

            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={todayChart} dataKey="value" nameKey="name" outerRadius={85}>
                  {todayChart.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </motion.div>

          <motion.div className="bg-white/5 backdrop-blur-2xl p-6 rounded-3xl border border-white/10 shadow-lg">
            <h2 className="mb-4 font-semibold">Weekly Attendance</h2>

            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={weeklyChart}>
                <XAxis dataKey="day" stroke="#aaa" />
                <YAxis stroke="#aaa" />
                <Tooltip />
                <Line type="monotone" dataKey="attendance" stroke="#38bdf8" strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </motion.div>

        </div>

        {/* TODAY CLASSES */}
        <div className="bg-white/5 backdrop-blur-2xl p-6 rounded-3xl border border-white/10 shadow-lg mb-10">

          <h2 className="mb-4 font-semibold">Today's Classes</h2>

          {classes.length === 0 ? (
            <p className="text-white/50">No classes today</p>
          ) : (
            classes.map((c, i) => {
              const status = getClassStatus(c)

              return (
                <motion.div
                  key={c.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex justify-between bg-white/5 p-4 rounded-xl mb-3 hover:bg-white/10 transition"
                >
                  <div>
                    <p>{c.subject}</p>
                    <p className="text-sm text-white/60">Period {c.period}</p>
                  </div>

                  <span className={`px-3 py-1 rounded-full ${status.style}`}>
                    {status.text}
                  </span>
                </motion.div>
              )
            })
          )}

        </div>

        {/* ALERTS */}
        <div className="bg-white/5 backdrop-blur-2xl p-6 rounded-3xl border border-white/10 shadow-lg mb-10">

          <h2 className="mb-4 font-semibold flex items-center gap-2">
            <AlertTriangle className="text-yellow-400" /> Alerts
          </h2>

          {insights.length === 0 ? (
            <p className="text-white/60">No alerts</p>
          ) : (
            insights.map((s, i) => (
              <motion.div
                key={s.roll}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="bg-yellow-500/10 p-4 rounded-xl mb-3 flex justify-between"
              >
                <div>
                  <p>{s.name}</p>
                  <p className="text-sm text-white/60">Roll: {s.roll}</p>
                </div>

                <p>{s.days} days absent consecutively</p>
              </motion.div>
            ))
          )}

        </div>

        {/* LOW */}
        <div className="bg-red-500/10 p-6 rounded-3xl border border-red-500/20">

          <h2 className="text-red-400 font-semibold mb-4">
            ⚠ Low Attendance (&lt; 50%)
          </h2>

          {low.length === 0 ? (
            <p>No students below 50%</p>
          ) : (
            low.map((s, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="flex justify-between mb-2"
              >
                <p>{s.name} ({s.roll})</p>
                <p>{s.percentage}%</p>
              </motion.div>
            ))
          )}

        </div>

      </div>
    </div>
  )
}

function StatCard({ icon, title, value }) {
  return (
    <motion.div
      whileHover={{ scale: 1.05, y: -4 }}
      className="bg-white/5 backdrop-blur-2xl border border-white/10 p-6 rounded-3xl flex justify-between shadow-lg"
    >
      <div>
        <p className="text-white/60 text-sm">{title}</p>
        <h2 className="text-3xl font-bold">{value || 0}</h2>
      </div>

      <div className="bg-blue-500/20 p-3 rounded-xl text-cyan-300">
        {icon}
      </div>
    </motion.div>
  )
}