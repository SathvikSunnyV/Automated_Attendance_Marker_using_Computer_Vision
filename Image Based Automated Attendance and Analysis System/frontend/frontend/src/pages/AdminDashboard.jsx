import { motion } from "framer-motion";
import {
  Shield,
  UserCheck,
  Clock,
  Users,
  LogOut,
  Check
} from "lucide-react";

import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import API from "../api/api";
import toast from "react-hot-toast";

export default function AdminDashboard() {

  const navigate = useNavigate();

  const [pending,setPending] = useState([])
  const [approved,setApproved] = useState([])
  const [total,setTotal] = useState([])

  // Protect dashboard
  useEffect(() => {

    const token = localStorage.getItem("admin_token");

    if (!token) {
      navigate("/admin/login");
    }

    loadTeachers()

  }, []);


  async function loadTeachers(){

    try{

      const res = await API.get("/admin/pending-teachers")
      setPending(res.data)

      const approvedRes = await API.get("/admin/approved-teachers")
      setApproved(approvedRes.data)

      const totalRes = await API.get("/admin/all-teachers")
      setTotal(totalRes.data)

    }catch(err){
      console.log(err)
    }

  }


  async function approveTeacher(id){

    try{

      await API.post(`/admin/approve/${id}`)

      toast.success("Teacher Approved")

      loadTeachers()

    }catch(err){

      toast.error("Approval failed")

    }

  }


  const handleLogout = () => {
    localStorage.removeItem("admin_token");
    navigate("/admin/login");
  };

async function rejectTeacher(id){

try{

await API.post(`/admin/reject/${id}`)

toast.success("Teacher Rejected")

loadTeachers()

}catch(err){

toast.error("Reject failed")

}

}
  return (

    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-800 text-white p-8 relative overflow-hidden">

      {/* Background Glow */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute w-[700px] h-[700px] bg-indigo-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]" />
        <div className="absolute w-[600px] h-[600px] bg-purple-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]" />
      </div>


      {/* Header */}
      <div className="flex justify-between items-center mb-10">

        <motion.div
          initial={{ opacity:0, y:20 }}
          animate={{ opacity:1, y:0 }}
          className="flex items-center gap-4"
        >

          <div className="p-4 rounded-xl bg-amber-500/10">
            <Shield size={35} className="text-amber-400"/>
          </div>

          <div>
            <h1 className="text-3xl font-bold">
              Admin Dashboard
            </h1>

            <p className="text-gray-400">
              Manage teachers and system approvals
            </p>
          </div>

        </motion.div>


        {/* Logout */}
        <motion.button
          whileHover={{ scale:1.05 }}
          whileTap={{ scale:0.95 }}
          onClick={handleLogout}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/10 border border-red-500/20 hover:bg-red-500/20"
        >
          <LogOut size={18}/>
          Logout
        </motion.button>

      </div>



      {/* Stats Cards */}
      <div className="grid md:grid-cols-3 gap-6 mb-8">

        {/* Pending */}
        <motion.div
          whileHover={{ scale:1.05 }}
          className="p-6 rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 shadow-xl"
        >
          <div className="flex items-center gap-4">

            <Clock className="text-yellow-400"/>

            <div>
              <h2 className="text-xl font-semibold">
                Pending Requests
              </h2>
              <p className="text-gray-400 text-sm">
                Teachers awaiting approval
              </p>
            </div>

          </div>

          <h1 className="text-3xl font-bold mt-4 text-yellow-400">
            {pending.length}
          </h1>

        </motion.div>


        {/* Approved */}
        <motion.div
          whileHover={{ scale:1.05 }}
          className="p-6 rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 shadow-xl"
        >
          <div className="flex items-center gap-4">

            <UserCheck className="text-green-400"/>

            <div>
              <h2 className="text-xl font-semibold">
                Approved Teachers
              </h2>
              <p className="text-gray-400 text-sm">
                Active teachers
              </p>
            </div>

          </div>

          <h1 className="text-3xl font-bold mt-4 text-green-400">
            {approved.length}
          </h1>

        </motion.div>


        {/* Total */}
        <motion.div
          whileHover={{ scale:1.05 }}
          className="p-6 rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 shadow-xl"
        >
          <div className="flex items-center gap-4">

            <Users className="text-indigo-400"/>

            <div>
              <h2 className="text-xl font-semibold">
                Total Teachers
              </h2>
              <p className="text-gray-400 text-sm">
                All registered teachers
              </p>
            </div>

          </div>

          <h1 className="text-3xl font-bold mt-4 text-indigo-400">
            {total.length}
          </h1>

        </motion.div>

      </div>



      {/* Pending Table */}
      <motion.div
        initial={{ opacity:0 }}
        animate={{ opacity:1 }}
        className="p-6 rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 shadow-xl"
      >

        <h2 className="text-xl font-semibold mb-6">
          Pending Teacher Requests
        </h2>


        {pending.length === 0 ? (

          <div className="text-gray-400">
            No pending requests available
          </div>

        ) : (

          <div className="space-y-3">

            {pending.map((teacher)=> (

              <div
                key={teacher.id}
                className="flex justify-between items-center p-4 rounded-xl bg-white/5"
              >

                <div>
                  <h3 className="font-semibold">
                    {teacher.name}
                  </h3>

                  <p className="text-gray-400 text-sm">
                    {teacher.email}
                  </p>

                </div>

                <motion.button
                  whileHover={{scale:1.05}}
                  whileTap={{scale:0.95}}
                  onClick={()=>approveTeacher(teacher.id)}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 rounded-lg"
                >
                  <Check size={16}/>
                  Approve
                </motion.button>
                <motion.button
onClick={()=>rejectTeacher(teacher.id)}
className="px-4 py-2 bg-red-600 rounded-lg"
>
Reject
</motion.button>

              </div>

            ))}

          </div>

        )}

      </motion.div>


    </div>

  );

}