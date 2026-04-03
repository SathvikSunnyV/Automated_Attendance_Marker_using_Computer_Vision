import { motion } from "framer-motion"
import { useEffect, useState } from "react"
import { Users, CheckCircle, XCircle, BarChart3, AlertTriangle } from "lucide-react"
import API from "../api/api"
import StudentSidebar from "../components/StudentSidebar"

export default function StudentDashboard(){

  const [summary,setSummary] = useState({
    total_classes:0,
    present:0,
    absent:0,
    percentage:0
  })

  const [recent,setRecent] = useState([])

  useEffect(()=>{

    fetchSummary()
    fetchRecent()

    const ws = new WebSocket("ws://127.0.0.1:8000/ws/attendance")

    ws.onmessage = (event)=>{
      const data = JSON.parse(event.data)

      if(data.type === "attendance_update"){
        fetchSummary()
        fetchRecent()
      }
    }

    return () => ws.close()

  },[])

  async function fetchSummary(){

    try{

      const res = await API.get(`/student/analytics`)

      const data = res.data

      setSummary({
        total_classes: data.total_classes,
        present: data.present,
        absent: data.absent,
        percentage: data.percentage
      })

    }catch(err){
      console.log(err)
    }

  }

  async function fetchRecent(){

    try{
      const res = await API.get(`/student/recent-attendance`)
      setRecent(res.data)
    }catch(err){
      console.log(err)
    }

  }

  const isLow = summary.percentage < 75

  return(

    <div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-800 text-white relative overflow-hidden">

      {/* 🌌 background glow */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute w-[700px] h-[700px] bg-indigo-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]"></div>
        <div className="absolute w-[600px] h-[600px] bg-purple-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]"></div>
      </div>

      <StudentSidebar/>

      <div className="flex-1 p-10">

        <h1 className="text-4xl font-bold mb-8 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
          Student Dashboard
        </h1>

        {/* ALERT */}
        {isLow && (
          <motion.div
            initial={{opacity:0,y:-20}}
            animate={{opacity:1,y:0}}
            className="mb-6 p-5 rounded-2xl bg-red-500/10 border border-red-500/30 flex items-center gap-3 backdrop-blur-lg"
          >
            <AlertTriangle />
            <div>
              <p className="font-semibold">
                Attendance below 75%
              </p>
              <p className="text-sm text-red-300">
                You may not be eligible for exams.
              </p>
            </div>
          </motion.div>
        )}

        {/* KPI */}
        <div className="grid md:grid-cols-4 gap-6 mb-10">

          <Card icon={<Users size={26}/>} title="Total Classes" value={summary.total_classes} color="indigo"/>
          <Card icon={<CheckCircle size={26}/>} title="Present" value={summary.present} color="green"/>
          <Card icon={<XCircle size={26}/>} title="Absent" value={summary.absent} color="red"/>
          <Card icon={<BarChart3 size={26}/>} title="Attendance %" value={`${summary.percentage}%`} color="purple"/>

        </div>

        {/* PROGRESS */}
        <motion.div
          initial={{opacity:0,y:30}}
          animate={{opacity:1,y:0}}
          className="bg-white/5 backdrop-blur-2xl border border-white/10 p-8 rounded-3xl mb-10 shadow-lg"
        >

          <p className="mb-3 font-semibold text-gray-300">
            Attendance Progress
          </p>

          <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">

            <motion.div
              initial={{width:0}}
              animate={{width:`${summary.percentage}%`}}
              transition={{duration:0.6}}
              className={`h-3 rounded-full ${
                isLow
                  ? "bg-red-500"
                  : "bg-gradient-to-r from-indigo-500 to-purple-500"
              }`}
            />

          </div>

        </motion.div>

        {/* TABLE */}
        <motion.div
          initial={{opacity:0,y:30}}
          animate={{opacity:1,y:0}}
          className="bg-white/5 backdrop-blur-2xl border border-white/10 p-8 rounded-3xl shadow-lg"
        >

          <h2 className="text-xl mb-6">
            Recent Attendance
          </h2>

          <div className="overflow-auto">

            <table className="w-full text-sm">

              <thead>
                <tr className="text-white/60 border-b border-white/10">
                  <th className="py-3 text-left">Date</th>
                  <th className="text-left">Subject</th>
                  <th className="text-left">Period</th>
                  <th className="text-left">Status</th>
                </tr>
              </thead>

              <tbody>

                {recent.map((item,i)=>(

                  <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition">

                    <td className="py-3">{item.date}</td>
                    <td>{item.subject}</td>
                    <td>{item.period}</td>

                    <td>
                      <span className={`px-3 py-1 rounded-full text-xs
                        ${item.status==="Present"
                          ? "bg-green-500/20 text-green-300"
                          : item.status==="Absent"
                          ? "bg-red-500/20 text-red-300"
                          : "bg-gray-500/20 text-gray-300"}`}>
                        {item.status}
                      </span>
                    </td>

                  </tr>

                ))}

              </tbody>

            </table>

          </div>

        </motion.div>

      </div>

    </div>

  )

}

function Card({icon,title,value,color}){

  const colors={
    indigo:"bg-indigo-500/20 text-indigo-300",
    green:"bg-green-500/20 text-green-300",
    red:"bg-red-500/20 text-red-300",
    purple:"bg-purple-500/20 text-purple-300"
  }

  return(

    <motion.div
      whileHover={{scale:1.05,y:-4}}
      transition={{type:"spring",stiffness:200}}
      className="bg-white/5 backdrop-blur-xl border border-white/10 p-6 rounded-3xl shadow-md"
    >

      <div className="flex justify-between items-center">

        <div>
          <p className="text-white/60 text-sm">{title}</p>
          <h2 className="text-3xl font-bold">{value}</h2>
        </div>

        <div className={`p-3 rounded-xl ${colors[color]}`}>
          {icon}
        </div>

      </div>

    </motion.div>

  )

}