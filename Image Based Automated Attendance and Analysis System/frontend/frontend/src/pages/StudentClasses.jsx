import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import Calendar from "react-calendar"
import "react-calendar/dist/Calendar.css"
import API from "../api/api"
import StudentSidebar from "../components/StudentSidebar"

export default function StudentClasses(){

  const [date, setDate] = useState(new Date())
  const [attendance, setAttendance] = useState([])

  function fetchAttendance(){

    const formatted = date.getFullYear() + "-" +
      String(date.getMonth()+1).padStart(2,'0') + "-" +
      String(date.getDate()).padStart(2,'0')

    API.get(`/student/day-attendance?date_selected=${formatted}`)
    .then(res=>{
      setAttendance(res.data || [])
    })
    .catch(err=>{
      console.log(err)
    })
  }

  useEffect(()=>{

    fetchAttendance()

    const ws = new WebSocket("ws://127.0.0.1:8000/ws/attendance")

    ws.onmessage = ()=>{
      fetchAttendance()
    }

    const interval = setInterval(fetchAttendance, 5000)

    return () => {
      ws.close()
      clearInterval(interval)
    }

  }, [date])

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
          My Classes
        </h1>

        {/* Refresh */}
        

        <div className="grid md:grid-cols-3 gap-8">

          {/* 📅 Calendar */}
          <motion.div
            initial={{opacity:0,y:30}}
            animate={{opacity:1,y:0}}
            className="bg-white/5 backdrop-blur-2xl border border-white/10 p-6 rounded-3xl shadow-lg"
          >

            <h2 className="mb-4 font-semibold text-gray-300">
              Select Date
            </h2>

            <div className="rounded-xl overflow-hidden">
              <Calendar
                onChange={setDate}
                value={date}
                tileDisabled={({date}) => date.getDay()===0}
              />
            </div>

          </motion.div>

          {/* 📚 Classes */}
          <div className="md:col-span-2">

            <h2 className="mb-6 text-lg text-gray-300">
              {date.toDateString()}
            </h2>

            {/* Sunday */}
            {date.getDay()===0 && (

              <motion.div
                initial={{opacity:0,scale:0.95}}
                animate={{opacity:1,scale:1}}
                className="bg-white/5 backdrop-blur-2xl border border-white/10 p-10 rounded-3xl text-center"
              >

                <h2 className="text-2xl font-bold">
                  No Classes Today
                </h2>

                <p className="text-white/60 mt-2">
                  Enjoy your Sunday 🎉
                </p>

              </motion.div>

            )}

            {/* Weekdays */}
            {date.getDay()!==0 && (

              <div className="grid md:grid-cols-2 gap-6">

                {[1,2,3,4,5,6,7,8].map(period=>{

                  const record = attendance.find(a=>a.period===period)

                  const status = record ? record.status : "Not Updated"
                  const subject = record ? record.subject : "Class"

                  return(

                    <motion.div
                      key={period}
                      whileHover={{scale:1.05,y:-4}}
                      transition={{type:"spring",stiffness:200}}
                      className="bg-white/5 backdrop-blur-2xl border border-white/10 p-6 rounded-3xl shadow-md"
                    >

                      <div className="flex justify-between mb-3">

                        <h3 className="font-semibold">
                          Period {period}
                        </h3>

                        <span className="text-white/50 text-sm">
                          {subject}
                        </span>

                      </div>

                      <span className={`px-3 py-1 rounded-full text-xs font-medium
                        ${status==="Present"
                          ? "bg-green-500/20 text-green-300"
                          : status==="Absent"
                          ? "bg-red-500/20 text-red-300"
                          : "bg-gray-500/20 text-gray-300"}`}>

                        {status}

                      </span>

                    </motion.div>

                  )

                })}

              </div>

            )}

          </div>

        </div>

      </div>

    </div>

  )

}