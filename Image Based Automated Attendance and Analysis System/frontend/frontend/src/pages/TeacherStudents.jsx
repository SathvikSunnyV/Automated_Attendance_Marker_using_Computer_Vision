import { useEffect, useState } from "react"
import API from "../api/api"
import TeacherSidebar from "../components/TeacherSidebar"
import { motion, AnimatePresence } from "framer-motion"
import { Download } from "lucide-react"
import html2canvas from "html2canvas"
import jsPDF from "jspdf"

import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar
} from "recharts"

const COLORS = ["#22c55e", "#ef4444", "#38bdf8"]

export default function TeacherStudents() {

  const [students, setStudents] = useState([])
  const [selected, setSelected] = useState(null)
  const [page, setPage] = useState(1)
  const perPage = 8

  useEffect(() => {
    loadStudents()
  }, [])

  async function loadStudents() {
    const res = await API.get("/students")
    setStudents(res.data || [])
  }

  // ✅ DOWNLOAD FUNCTION (NEW)

  const paginated = students.slice((page - 1) * perPage, page * perPage)

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-800 text-white relative overflow-hidden">

      {/* glow */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute w-[700px] h-[700px] bg-blue-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]"></div>
        <div className="absolute w-[600px] h-[600px] bg-cyan-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]"></div>
      </div>

      <TeacherSidebar />

      <div className="flex-1 p-10">

        <h1 className="text-4xl font-bold mb-8 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
          Student Analytics
        </h1>

        {/* TABLE */}
        <div className="bg-white/5 backdrop-blur-2xl border border-white/10 rounded-3xl overflow-hidden shadow-lg">

          <table className="w-full">

            <thead className="bg-white/10 text-left text-white/70">
              <tr>
                <th className="p-4">Student</th>
                <th>Roll No</th>
                <th>Action</th>
              </tr>
            </thead>

            <tbody>
              {paginated.map((s, i) => (
                <motion.tr
                  key={s.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="border-t border-white/10 hover:bg-white/5 transition"
                >

                  <td className="p-4 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center font-bold shadow-md">
                      {s.name[0]}
                    </div>
                    {s.name}
                  </td>

                  <td className="text-white/70">{s.roll_no}</td>

                  <td className="flex gap-2">

                    {/* VIEW */}
                    <button
                      onClick={() => setSelected(s)}
                      className="px-4 py-1 bg-blue-500/20 rounded-full hover:bg-blue-500/40 transition"
                    >
                      View
                    </button>

                    {/* DOWNLOAD */}
                    <button
  onClick={() => setSelected({ ...s, autoDownload: true })}
  className="flex items-center justify-center 
  w-9 h-9 rounded-lg 
  bg-white/5 border border-white/10 
  hover:bg-white/10 hover:border-white/20 
  transition-all duration-200"
  title="Download Report"
>
  <Download size={16} className="text-white/80 hover:text-white" />
</button>

                  </td>

                </motion.tr>
              ))}
            </tbody>

          </table>

        </div>

        {/* PAGINATION */}
        <div className="flex justify-center items-center gap-6 mt-6">

          <button
            onClick={() => setPage(p => Math.max(p - 1, 1))}
            className="px-4 py-2 bg-white/10 rounded-lg hover:bg-white/20"
          >
            Prev
          </button>

          <span className="text-white/70">
            Page {page}
          </span>

          <button
            onClick={() => setPage(p => p + 1)}
            className="px-4 py-2 bg-white/10 rounded-lg hover:bg-white/20"
          >
            Next
          </button>

        </div>

        <AnimatePresence>
          {selected && (
            <StudentModal student={selected} onClose={() => setSelected(null)} />
          )}
        </AnimatePresence>

      </div>
    </div>
  )
}






function StudentModal({ student, onClose }) {


  const [data, setData] = useState(null)

  useEffect(() => {
    load()
  }, [student])




  async function load() {
    const res = await API.get(`/teacher/student-analysis/${student.id}`)
    setData(res.data)
  }

  useEffect(() => {
  if (data && student.autoDownload) {

    const waitForCharts = setInterval(() => {
      const charts = document.querySelectorAll("canvas, svg")

      if (charts.length > 0) {
        clearInterval(waitForCharts)

        setTimeout(() => {
          downloadFullReport()
        }, 2700) // small safe delay
      }
    }, 200)

    return () => clearInterval(waitForCharts)
  }
}, [data])

  async function downloadFullReport() {

  const element = document.getElementById("report-section")

  // save original styles
  const originalMaxHeight = element.style.maxHeight
  const originalOverflow = element.style.overflow

  element.style.maxHeight = "none"
  element.style.overflow = "visible"

  const canvas = await html2canvas(element, { scale: 2 })
  const imgData = canvas.toDataURL("image/png")

  const pdf = new jsPDF("p", "mm", "a4")

  const imgWidth = 210
  const pageHeight = 297
  const imgHeight = (canvas.height * imgWidth) / canvas.width

  let position = 0

  pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight)

  while (imgHeight + position > pageHeight) {
    position -= pageHeight
    pdf.addPage()
    pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight)
  }

  pdf.save(`${student.name}_full_report.pdf`)

  // ✅ restore styles (IMPORTANT)
  element.style.maxHeight = originalMaxHeight
  element.style.overflow = originalOverflow
}

  if (!data) return null

  const safe = {
    distribution: [],
    subjects: [],
    ledger: [],
    insights: [],
    suggestions: [],
    advanced: {},
    monthly: [],
    ...data
  }

  const percentage = safe.stats?.percentage || 0

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50"
    >

      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        id="report-section"
        className="bg-gradient-to-br from-slate-900 via-blue-950 to-slate-800 w-[1100px] max-h-[90vh] overflow-y-auto rounded-3xl p-6 text-white shadow-2xl"
      >
        

        <button
  onClick={onClose}
  className="absolute top-6 right-6 text-white/70 hover:text-white text-lg"
>
  ✖
</button>

        <div className="flex justify-between items-center mb-6">

  <h1 className="text-3xl font-bold">
    {student.roll_no} Analytics
  </h1>

  <button
  onClick={downloadFullReport}
  className="absolute top-6 right-16 flex items-center justify-center 
  w-9 h-9 rounded-lg bg-white/5 border border-white/10 
  hover:bg-white/10 hover:border-white/20 
  transition-all duration-200 shadow-sm"
  title="Download Report"
>
  <Download size={18} className="text-white/80 hover:text-white transition" />
</button>

</div>

        {/* STATS */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <Stat title="Total" value={safe.stats?.total || 0} />
          <Stat title="Present" value={safe.stats?.present || 0} color="text-green-400" />
          <Stat title="Absent" value={safe.stats?.absent || 0} color="text-red-400" />
          <Stat title="%" value={percentage + "%"} color="text-cyan-400" />
        </div>

        {/* CHARTS */}
        <div className="grid grid-cols-2 gap-6 mb-6">

          <Card title="Monthly Trend">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={safe.monthly}>
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip
  cursor={{ fill: "transparent" }}
  contentStyle={{
    backgroundColor: "#020617",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: "10px"
  }}
/>  
                <Line dataKey="value" stroke="#38bdf8" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={safe.distribution} dataKey="value" innerRadius={60}>
                  {safe.distribution.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
               <Tooltip
  cursor={{ fill: "transparent" }}
  contentStyle={{
    backgroundColor: "#020617",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: "10px"
  }}
/>
              </PieChart>
            </ResponsiveContainer>
          </Card>

        </div>

        {/* SUBJECT + LEDGER */}
        <div className="grid grid-cols-2 gap-6">

          <Card title="Subject Performance">
            <ResponsiveContainer width="100%" height={250}>
            <BarChart data={safe.subjects}>
  <XAxis dataKey="subject" />
  <YAxis />

  <Tooltip
  cursor={{ fill: "transparent" }}
  contentStyle={{
    backgroundColor: "#020617",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: "10px"
  }}
/>

 <Bar
  dataKey="percentage"
  fill="#38bdf8"
  radius={[8,8,0,0]}
  activeBar={{
    fill: "#38bdf8",
    stroke: "none"
  }}
/>
</BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Ledger">
            {safe.ledger.length === 0 ? (
              <p className="text-white/50">No data</p>
            ) : (
              safe.ledger.map((l, i) => (
                <div key={i} className="flex justify-between p-2 border-b border-white/10">
                  <span>{l.subject}</span>
                  <span>{l.attended}/{l.total}</span>
                </div>
              ))
            )}
          </Card>

        </div>

        {/* INSIGHTS */}
        <div className="mt-6 bg-yellow-500/10 p-4 rounded-xl">
          <h3>🧠 Insights</h3>
          {safe.insights.map((i, idx) => (<p key={idx}>• {i}</p>))}
        </div>

        {/* SUGGESTIONS */}
        <div className="mt-4 bg-green-500/10 p-4 rounded-xl">
          <h3>💡 Suggestions</h3>
          {safe.suggestions.map((s, idx) => (<p key={idx}>• {s}</p>))}
        </div>

      </motion.div>
    </motion.div>
  )
}




function Stat({ title, value, color }) {
  return (
    <div className="bg-white/5 backdrop-blur-xl p-4 rounded-xl">
      <p className="text-gray-400">{title}</p>
      <h2 className={`text-2xl font-bold ${color || ""}`}>{value}</h2>
    </div>
  )
}

function Card({ title, children }) {
  return (
    <div className="bg-white/5 backdrop-blur-xl p-4 rounded-xl">
      <h3 className="mb-3 font-semibold">{title}</h3>
      {children}
    </div>
  )
}