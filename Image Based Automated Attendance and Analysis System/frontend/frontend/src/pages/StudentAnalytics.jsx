import { motion, AnimatePresence } from "framer-motion"
import { useEffect, useState } from "react"
import StudentSidebar from "../components/StudentSidebar"
import API from "../api/api"
import toast from "react-hot-toast"

import {
PieChart,
Pie,
Cell,
Tooltip,
ResponsiveContainer,
BarChart,
Bar,
XAxis,
YAxis,
CartesianGrid,
AreaChart,
Area
} from "recharts"

const COLORS = ["#22c55e","#ef4444","#f59e0b","#6366f1"]

function safePct(present,absent){
const total = present + absent
return total>0 ? Math.round((present/total)*100) : 0
}

export default function StudentAnalytics(){

const [distribution,setDistribution] = useState([])
const [summary,setSummary] = useState({})
const [monthly,setMonthly] = useState([])
const [subjects,setSubjects] = useState([])
const [formatted,setFormatted] = useState({})

const [loading,setLoading] = useState(true)
const [error,setError] = useState(null)

useEffect(()=>{

const token = localStorage.getItem("token")

if(!token){
setError("User not authenticated")
setLoading(false)
return
}

fetchAll()

},[])

async function fetchAll(){

setLoading(true)

try{

await Promise.all([
fetchAnalytics(),
fetchMonthly(),
fetchSubjects()
])

}catch(e){

console.error(e)
setError("Failed to load analytics")

}finally{

setLoading(false)

}

}

async function fetchAnalytics(){

const res = await API.get("/student/analytics")
const data = res.data

setDistribution(data.distribution || [])

setSummary({
total_classes: data.total_classes,
present: data.present,
absent: data.absent,
percentage: data.percentage
})

setFormatted(data.formatted || {})

}

async function fetchMonthly(){

const res = await API.get("/student/monthly-attendance")
setMonthly(res.data || [])

}

async function fetchSubjects(){

const res = await API.get("/student/subject-attendance")
setSubjects(res.data || [])

}

const avgPct = summary.percentage ?? 0

/* LOADING */
if(loading){
return(
<div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-800 text-white">
<StudentSidebar/>
<main className="flex-1 flex items-center justify-center">
<div className="flex flex-col items-center gap-4">
<div className="w-10 h-10 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"/>
<p className="text-gray-400 text-sm">Loading analytics...</p>
</div>
</main>
</div>
)
}

/* ERROR */
if(error){
return(
<div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-800 text-white">
<StudentSidebar/>
<main className="flex-1 flex items-center justify-center">
<div className="text-center">
<p className="text-red-400 text-lg font-semibold">{error}</p>
<button
onClick={fetchAll}
className="mt-4 px-6 py-2 bg-indigo-600 rounded-xl hover:bg-indigo-500 shadow"
>
Retry
</button>
</div>
</main>
</div>
)
}

return(

<div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-800 text-white relative overflow-hidden">

{/* background glow */}
<div className="absolute inset-0 -z-10">
<div className="absolute w-[700px] h-[700px] bg-indigo-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]"></div>
<div className="absolute w-[600px] h-[600px] bg-purple-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]"></div>
</div>

<StudentSidebar/>

<main className="flex-1 p-10">

{/* HEADER */}
<motion.div
initial={{opacity:0,y:-20}}
animate={{opacity:1,y:0}}
className="mb-10"
>
<h1 className="text-4xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
Attendance Analytics
</h1>
<p className="text-gray-400 text-sm mt-1">
Performance Insights
</p>
</motion.div>

{/* KPI */}
<div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">

<KpiCard title="Total Classes" value={summary.total_classes} sub="Academic Year"/>
<KpiCard title="Present" value={summary.present} color="text-green-400" sub="Verified"/>
<KpiCard title="Absent" value={summary.absent} color="text-red-400" sub="Missed"/>

<KpiCard
title="Attendance %"
value={`${summary.percentage ?? 0}%`}
color={avgPct < 75 ? "text-red-400" : "text-indigo-400"}
sub={avgPct < 75 ? "Below Threshold" : "Safe Standing"}
/>

</div>

{/* CHARTS */}
<div className="grid lg:grid-cols-3 gap-8 mb-10">

{/* AREA */}
<GlassCard title="Monthly Attendance Trend" className="lg:col-span-2">

<ResponsiveContainer width="100%" height={300}>
<AreaChart data={monthly}>

<defs>
<linearGradient id="colorAttend" x1="0" y1="0" x2="0" y2="1">
<stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
<stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
</linearGradient>
</defs>

<XAxis dataKey="month" stroke="#94a3b8"/>
<YAxis stroke="#94a3b8"/>
<Tooltip content={<CustomTooltip />} cursor={false}/>

<Area
type="monotone"
dataKey="attendance"
stroke="#6366f1"
fill="url(#colorAttend)"
strokeWidth={3}
/>

</AreaChart>
</ResponsiveContainer>

</GlassCard>

{/* PIE */}
<GlassCard title="Attendance Distribution">

<div className="relative flex justify-center items-center">

<ResponsiveContainer width="100%" height={260}>
<PieChart>

<Pie
data={distribution}
innerRadius={75}
outerRadius={95}
dataKey="value"
>

{distribution.map((entry,i)=>(
<Cell key={i} fill={COLORS[i % COLORS.length]} />
))}

</Pie>

<Tooltip content={<PieTooltip data={distribution}/>}/>

</PieChart>
</ResponsiveContainer>

</div>

<div className="mt-4 text-center space-y-1">

<p className="text-sm text-green-400">
Present {formatted.present}
</p>

<p className="text-sm text-red-400">
Absent {formatted.absent}
</p>

<p className="text-xs text-gray-400">
[ Not Updated {formatted.not_updated} ]
</p>

</div>

</GlassCard>

</div>

{/* SUBJECT + LEDGER */}
<div className="grid lg:grid-cols-3 gap-8">

<GlassCard title="Subject Performance (%)" className="lg:col-span-2">

<ResponsiveContainer width="100%" height={350}>
<BarChart
data={subjects.map(s => ({
...s,
percentage: safePct(s.present, s.absent)
}))}
>

<CartesianGrid strokeDasharray="3 3" stroke="#334155"/>
<XAxis dataKey="subject" stroke="#94a3b8"/>
<YAxis domain={[0,100]} stroke="#94a3b8"/>
<Tooltip content={<CustomTooltip />} cursor={false}/>

<Bar dataKey="percentage" radius={[8,8,0,0]}>
{subjects.map((entry, i) => {
const pct = safePct(entry.present, entry.absent)
return (
<Cell key={i} fill={pct < 75 ? "#ef4444" : "#6366f1"} />
)
})}
</Bar>

</BarChart>
</ResponsiveContainer>

</GlassCard>

<GlassCard title="Attendance Ledger">

<div className="space-y-3 max-h-[350px] overflow-y-auto pr-1">

{subjects.length === 0 && (
<p className="text-gray-400 text-sm text-center mt-10">
No attendance data available yet
</p>
)}

<AnimatePresence>

{subjects.map((sub, idx) => {

const pct = safePct(sub.present, sub.absent)
const total = sub.present + sub.absent

return (
<motion.div
key={idx}
initial={{ opacity: 0, y: 10 }}
animate={{ opacity: 1, y: 0 }}
exit={{ opacity: 0 }}
transition={{ delay: idx * 0.05 }}
className="flex justify-between items-center p-4 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition"
>

<div>
<p className="font-semibold">{sub.subject}</p>

<p className={`text-xs ${
pct < 75 ? "text-red-400" : "text-green-400"
}`}>
{pct}%
</p>
</div>

<div className="text-right">
<p className="text-lg font-bold">
{sub.present}/{total}
</p>

<p className="text-xs text-gray-400">
lectures
</p>
</div>

</motion.div>
)

})}

</AnimatePresence>

</div>

</GlassCard>

</div>

</main>

</div>

)

}

/* COMPONENTS */

function KpiCard({title,value,color="text-white",sub}){
return(
<motion.div
whileHover={{scale:1.05,y:-4}}
className="bg-white/5 backdrop-blur-2xl border border-white/10 p-6 rounded-2xl shadow-lg"
>
<p className="text-gray-400 text-sm">{title}</p>
<h2 className={`text-3xl font-bold ${color}`}>
{value ?? 0}
</h2>
<p className="text-xs text-gray-500 mt-1">{sub}</p>
</motion.div>
)
}

function GlassCard({title,children,className=""}){
return(
<motion.div
initial={{opacity:0,y:20}}
animate={{opacity:1,y:0}}
className={`bg-white/5 backdrop-blur-2xl border border-white/10 p-6 rounded-3xl shadow-lg ${className}`}
>
<h3 className="text-lg font-semibold mb-6">
{title}
</h3>
{children}
</motion.div>
)
}

function CustomTooltip({active,payload,label}){
if(!active || !payload?.length) return null
return(
<div className="bg-slate-900 border border-slate-700 p-3 rounded-xl shadow-lg">
<p className="text-gray-400 text-xs mb-1">
{label ?? payload[0]?.name}
</p>
<p className="font-bold text-white">
{payload[0].value}%
</p>
</div>
)
}

function PieTooltip({ active, payload, data }) {

if (!active || !payload || !payload.length) return null

const total = data?.reduce((sum, d) => sum + d.value, 0) || 1
const item = payload[0]
const percentage = ((item.value / total) * 100).toFixed(1)

return (
<div className="bg-slate-900 border border-slate-700 p-3 rounded-xl shadow-lg">
<p className="text-gray-400 text-xs mb-1">{item.name}</p>
<p className="font-bold text-white">
{item.value} ({percentage}%)
</p>
</div>
)
}