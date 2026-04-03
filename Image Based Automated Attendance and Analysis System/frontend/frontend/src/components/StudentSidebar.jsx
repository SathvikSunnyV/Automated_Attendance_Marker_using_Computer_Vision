import { NavLink, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  CalendarDays,
  BarChart3,
  LogOut,
  ScanFace,
} from "lucide-react";
import { useState } from "react";

export default function StudentSidebar() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  const name = localStorage.getItem("student_name") || "Student";
  const roll = localStorage.getItem("student_roll") || "";

  const logout = () => {
    localStorage.clear();
    navigate("/");
  };

  const linkStyle = ({ isActive }) =>
    `flex items-center gap-3 px-3 py-3 rounded-xl transition-all duration-300 group
     ${
       isActive
         ? "bg-gradient-to-r from-indigo-500/30 to-purple-500/20 text-cyan-300"
         : "text-gray-300 hover:bg-white/10 hover:text-white"
     }`;

  return (
    <div
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      className="w-20 hover:w-64 min-h-screen transition-all duration-300 ease-in-out
                 bg-gradient-to-b from-slate-900 via-indigo-950 to-slate-900
                 backdrop-blur-xl border-r border-white/10 text-white p-4 flex flex-col
                 overflow-hidden"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 mb-10">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center flex-shrink-0">
          <ScanFace size={22} />
        </div>

        <div
          className={`whitespace-nowrap overflow-hidden transition-all duration-300 ${
            open ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-4"
          }`}
        >
          <h2 className="font-bold text-lg">Face Attendance</h2>
          <p className="text-xs text-gray-400">Student Panel</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-2 flex-1">
        <NavLink to="/student/dashboard" end className={linkStyle}>
          <LayoutDashboard size={20} className="flex-shrink-0" />
          <span
            className={`transition-all duration-300 whitespace-nowrap overflow-hidden ${
              open ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-6 w-0"
            }`}
          >
            Dashboard
          </span>
        </NavLink>

        <NavLink to="/student/classes" className={linkStyle}>
          <CalendarDays size={20} className="flex-shrink-0" />
          <span
            className={`transition-all duration-300 whitespace-nowrap overflow-hidden ${
              open ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-6 w-0"
            }`}
          >
            My Classes
          </span>
        </NavLink>

        <NavLink to="/student/analytics" className={linkStyle}>
          <BarChart3 size={20} className="flex-shrink-0" />
          <span
            className={`transition-all duration-300 whitespace-nowrap overflow-hidden ${
              open ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-6 w-0"
            }`}
          >
            Analytics
          </span>
        </NavLink>
      </nav>

      {/* Profile */}
      <div className="mt-auto pt-6 border-t border-white/10">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center flex-shrink-0 text-lg font-semibold">
            {name[0].toUpperCase()}
          </div>

          <div
            className={`transition-all duration-300 whitespace-nowrap overflow-hidden ${
              open ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-4"
            }`}
          >
            <p className="text-sm font-semibold">{name}</p>
            <p className="text-xs text-gray-400">{roll}</p>
          </div>
        </div>

        <button
          onClick={logout}
          className="flex items-center gap-3 w-full px-3 py-3 rounded-xl
                     hover:bg-red-500/20 text-red-300 transition"
        >
          <LogOut size={20} className="flex-shrink-0" />

          <span
            className={`transition-all duration-300 whitespace-nowrap overflow-hidden ${
              open ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-6 w-0"
            }`}
          >
            Logout
          </span>
        </button>
      </div>
    </div>
  );
}