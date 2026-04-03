    import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
    import { Toaster } from "react-hot-toast"
    import ConfirmModal from "./components/ConfirmModal"

    /* Landing */
    import LandingPage from "./pages/LandingPage"
    import RoleSelection from "./pages/RoleSelection"

    /* Student */
    import StudentRegister from "./pages/StudentRegister"
    import StudentLogin from "./pages/StudentLogin"
    import StudentDashboard from "./pages/StudentDashboard"
    import StudentClasses from "./pages/StudentClasses"
    import StudentAnalytics from "./pages/StudentAnalytics"

    /* Teacher */
    import TeacherRegister from "./pages/TeacherRegister"
    import TeacherLogin from "./pages/TeacherLogin"
    import TeacherDashboard from "./pages/TeacherDashboard"
    import Teachertimetable from "./pages/Teachertimetable"
    import TeacherAttendance from "./pages/TeacherAttendance"
    import TeacherStudents from "./pages/TeacherStudents"
    import TeacherSetPassword from "./pages/TeacherSetPassword"

    /* Admin */
import AdminLogin from "./pages/AdminLogin"
import AdminDashboard from "./pages/AdminDashboard"

    export default function App() {

    return (

        <>

        {/* 🔥 TOASTER (GLOBAL ALERT UI) */}
        <Toaster
            position="top-right"
            toastOptions={{
            style: {
                background: "#0f172a",
                color: "#fff",
                border: "1px solid rgba(255,255,255,0.1)"
            }
            }}
        />

        {/* 🔥 CONFIRM MODAL */}
        <ConfirmModal />

        {/* ROUTES */}
        <Router>

            <Routes>

            {/* Landing */}
            <Route path="/" element={<LandingPage />} />
            <Route path="/role" element={<RoleSelection />} />

            {/* Student */}
            <Route path="/student/register" element={<StudentRegister />} />
            <Route path="/student/login" element={<StudentLogin />} />
            <Route path="/student/dashboard" element={<StudentDashboard />} />
            <Route path="/student/classes" element={<StudentClasses />} />
            <Route path="/student/analytics" element={<StudentAnalytics />} />

            {/* Teacher */}
            <Route path="/teacher/register" element={<TeacherRegister />} />
            <Route path="/teacher/login" element={<TeacherLogin />} />
            <Route path="/teacher/set-password" element={<TeacherSetPassword />} />
            <Route path="/teacher/dashboard" element={<TeacherDashboard />} />
            <Route path="/teacher/timetable" element={<Teachertimetable />} />
            <Route path="/teacher/attendance" element={<TeacherAttendance />} />
            <Route path="/students" element={<TeacherStudents />} />


            <Route path="/admin/login" element={<AdminLogin />} />
            <Route path="/admin/dashboard" element={<AdminDashboard />} />
            

            </Routes>

        </Router>

        </>

    )

    }