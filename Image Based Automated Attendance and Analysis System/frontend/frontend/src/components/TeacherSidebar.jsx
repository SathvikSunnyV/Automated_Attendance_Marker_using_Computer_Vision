import { NavLink, useNavigate } from "react-router-dom"
import {
  LayoutDashboard,
  CalendarDays,
  Camera,
  BarChart3,
  LogOut,
  ScanFace
} from "lucide-react"

import { useState } from "react"

export default function TeacherSidebar() {

  const navigate = useNavigate()
  const [open, setOpen] = useState(false)

  const name = localStorage.getItem("teacher_name")

  function logout() {
    localStorage.clear()
    navigate("/")
  }

  const linkStyle = ({ isActive }) =>
    `flex items-center gap-3 px-3 py-3 rounded-xl
    transition-all duration-300
    ${isActive
      ? "bg-gradient-to-r from-blue-500/30 to-cyan-500/20 text-cyan-300"
      : "text-gray-300 hover:bg-white/10 hover:text-white"
    }`

  return (

    <div
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      className={`min-h-screen flex flex-col
      bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900
      backdrop-blur-xl border-r border-white/10
      text-white p-3
      transition-all duration-300 ease-in-out
      ${open ? "w-64" : "w-20"}
      `}
    >

      {/* Logo */}

      <div className="flex items-center gap-3 mb-10">

        <div className="min-w-[40px] h-10 rounded-xl 
        bg-gradient-to-br from-blue-500 to-cyan-500 
        flex items-center justify-center">

          <ScanFace size={20} />

        </div>

        <div
          className={`overflow-hidden whitespace-nowrap
          transition-all duration-300
          ${open ? "opacity-100" : "opacity-0 w-0"}
          `}
        >
          <h2 className="font-bold">Face Attendance</h2>
          <p className="text-xs text-gray-400">Teacher Panel</p>
        </div>

      </div>


      {/* Navigation */}

      <nav className="flex flex-col gap-2">

        <NavLink to="/teacher/dashboard" className={linkStyle}>
          <div className="min-w-[40px]">
            <LayoutDashboard size={20}/>
          </div>

          <span
            className={`transition-all duration-300 whitespace-nowrap
            ${open ? "opacity-100" : "opacity-0 w-0 overflow-hidden"}
            `}
          >
            Dashboard
          </span>

        </NavLink>


        <NavLink to="/teacher/timetable" className={linkStyle}>
          <div className="min-w-[40px]">
            <CalendarDays size={20}/>
          </div>

          <span
            className={`transition-all duration-300 whitespace-nowrap
            ${open ? "opacity-100" : "opacity-0 w-0 overflow-hidden"}
            `}
          >
            Timetable
          </span>

        </NavLink>


        <NavLink to="/teacher/attendance" className={linkStyle}>
          <div className="min-w-[40px]">
            <Camera size={20}/>
          </div>

          <span
            className={`transition-all duration-300 whitespace-nowrap
            ${open ? "opacity-100" : "opacity-0 w-0 overflow-hidden"}
            `}
          >
            Take Attendance
          </span>

        </NavLink>


        <NavLink to="/students" className={linkStyle}>
          <div className="min-w-[40px]">
            <BarChart3 size={20}/>
          </div>

          <span
            className={`transition-all duration-300 whitespace-nowrap
            ${open ? "opacity-100" : "opacity-0 w-0 overflow-hidden"}
            `}
          >
            Student Analytics
          </span>

        </NavLink>

      </nav>


      {/* Profile */}

      <div className="mt-auto pt-6 border-t border-white/10">

        <div className="flex items-center gap-3 mb-4">

          <div className="min-w-[40px] h-10 rounded-xl
          bg-gradient-to-br from-blue-500 to-cyan-500
          flex items-center justify-center">

            {name ? name[0] : "T"}

          </div>

          <div
            className={`transition-all duration-300 whitespace-nowrap
            ${open ? "opacity-100" : "opacity-0 w-0 overflow-hidden"}
            `}
          >
            <p className="text-sm font-semibold">
              {name || "Teacher"}
            </p>

            <p className="text-xs text-gray-400">
              Faculty
            </p>

          </div>

        </div>


        <button
          onClick={logout}
          className="flex items-center gap-3 px-3 py-2 rounded-xl
          hover:bg-red-500/20 text-red-300 transition"
        >

          <div className="min-w-[40px]">
            <LogOut size={20}/>
          </div>

          <span
            className={`transition-all duration-300 whitespace-nowrap
            ${open ? "opacity-100" : "opacity-0 w-0 overflow-hidden"}
            `}
          >
            Logout
          </span>

        </button>

      </div>

    </div>

  )
}