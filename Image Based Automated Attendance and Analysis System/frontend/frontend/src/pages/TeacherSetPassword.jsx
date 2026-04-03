import { motion } from "framer-motion"
import { useState, useEffect } from "react"
import API from "../api/api"
import toast from "react-hot-toast"
import { useNavigate,useLocation } from "react-router-dom"
import { Lock } from "lucide-react"

export default function TeacherSetPassword(){

const [password,setPassword] = useState("")
const navigate = useNavigate()
const location = useLocation()

const email = location.state?.email

// 🔥 Protect direct access
useEffect(()=>{
if(!email){
navigate("/teacher/register")
}
},[email,navigate])


async function handleSetPassword(){

try{

await API.post("/teacher/set-password",{
email,
password
})

toast.success("Password set successfully")

navigate("/teacher/login")

}catch{

toast.error("Failed to set password")

}

}

return(

<div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-800 relative overflow-hidden text-white">

{/* Background Glow */}
<div className="absolute inset-0 -z-10">
<div className="absolute w-[700px] h-[700px] bg-indigo-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]" />
<div className="absolute w-[600px] h-[600px] bg-purple-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]" />
<div className="absolute w-[500px] h-[500px] bg-cyan-400 opacity-20 blur-[140px] top-[40%] left-[40%]" />
</div>

<motion.div
initial={{ opacity:0 , y:40 }}
animate={{ opacity:1 , y:0 }}
className="bg-white/5 backdrop-blur-2xl border border-white/10 p-10 rounded-3xl shadow-xl flex flex-col gap-5 w-96"
>

{/* Header */}
<div className="flex flex-col items-center gap-2">

<div className="p-3 rounded-full bg-indigo-500/10">
<Lock size={30} className="text-indigo-400"/>
</div>

<h2 className="text-2xl font-bold">
Set Password
</h2>

<p className="text-gray-400 text-sm">
Create password to activate account
</p>

</div>

{/* Password Input */}
<input
type="password"
placeholder="Enter Password"
value={password}
onChange={(e)=>setPassword(e.target.value)}
className="p-3 rounded-lg bg-white/10 border border-white/20 outline-none focus:ring-2 focus:ring-indigo-500 transition"
/>

{/* Button */}
<motion.button
whileHover={{scale:1.05}}
whileTap={{scale:0.95}}
onClick={handleSetPassword}
className="bg-indigo-600 p-3 rounded-lg shadow-lg"
>
Set Password
</motion.button>

{/* Back to login */}
<p className="text-xs text-gray-400 text-center mt-2">
Already set password?{" "}
<span
onClick={()=>navigate("/teacher/login")}
className="text-indigo-400 cursor-pointer"
>
Login
</span>
</p>

</motion.div>

</div>

)

}