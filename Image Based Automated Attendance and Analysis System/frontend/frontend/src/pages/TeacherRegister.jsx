import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import API from "../api/api";
import toast from "react-hot-toast";
import { School, MailCheck, Clock } from "lucide-react";

export default function TeacherRegister() {

  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [teacherId, setTeacherId] = useState("");
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");

  const [status, setStatus] = useState("register");
  const [loading, setLoading] = useState(false);

  // Check status when user revisits
 


  async function checkStatus(email){

try{

const res = await API.get("/teacher/status",{
params:{email}
})

if(res.data.status === "waiting"){
setStatus("waiting")
}

if(res.data.status === "approved"){
setStatus("approved")
}

if(res.data.status === "active"){
navigate("/teacher/login")
}

if(res.data.status === "rejected"){
setStatus("rejected")
}

}catch(err){
console.log(err)
}

}

  async function sendOTP(){

    try{

      setLoading(true)

      await API.post("/send-otp",null,{
        params:{
          email,
          role:"teacher"
        }
      })

      toast.success("OTP sent")
      setStatus("otp")

    }catch(err){

      toast.error("Failed to send OTP")

    }

    setLoading(false)

  }


async function verifyOTP(){

try{

await API.post("/verify-otp",null,{
params:{
email,
otp
}
})

toast.success("Email verified")

// AFTER OTP VERIFY → CHECK STATUS
const res = await API.get("/teacher/status",{
params:{email}
})

const status = res.data.status

if(status === "waiting"){
setStatus("waiting")
return
}

if(status === "approved"){
setStatus("approved")
return
}

if(status === "rejected"){
setStatus("rejected")
return
}

if(status === "active"){
navigate("/teacher/login")
return
}

// If new teacher
setStatus("request")

}catch(err){

toast.error("Invalid OTP")

}

}


  async function sendRequest(){

    try{

      await API.post("/teacher/request",{
        name,
        teacher_id:teacherId,
        email
      })

    

      toast.success("Request sent to admin")
      setStatus("waiting")

    }catch(err){

      toast.error("Failed to send request")

    }

  }



  return (

    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-800 relative overflow-hidden text-white">

      {/* Background Glow */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute w-[700px] h-[700px] bg-indigo-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]" />
        <div className="absolute w-[600px] h-[600px] bg-purple-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]" />
        <div className="absolute w-[500px] h-[500px] bg-cyan-400 opacity-20 blur-[140px] top-[40%] left-[40%]" />
      </div>


      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/5 backdrop-blur-2xl border border-white/10 p-10 rounded-3xl shadow-xl flex flex-col gap-5 w-96"
      >

        <div className="flex flex-col items-center gap-2">

          <div className="p-3 rounded-full bg-cyan-500/10">
            <School size={30} className="text-cyan-400"/>
          </div>

          <h2 className="text-2xl font-bold">
            Teacher Registration
          </h2>

        </div>


        {/* REGISTER FORM */}
        {status==="register" && (

        <>

        <input
          placeholder="Name"
          value={name}
          onChange={(e)=>setName(e.target.value)}
          className="p-3 rounded-lg bg-white/10 border border-white/20"
        />

        <input
          placeholder="Teacher ID"
          value={teacherId}
          onChange={(e)=>setTeacherId(e.target.value)}
          className="p-3 rounded-lg bg-white/10 border border-white/20"
        />

        <input
          placeholder="Institute Email"
          value={email}
          onChange={(e)=>setEmail(e.target.value)}
          className="p-3 rounded-lg bg-white/10 border border-white/20"
        />

        <motion.button
          whileHover={{scale:1.05}}
          whileTap={{scale:0.95}}
          onClick={sendOTP}
          disabled={loading}
          className="bg-cyan-600 p-3 rounded-lg shadow-lg"
        >
          Send OTP
        </motion.button>

        </>

        )}



        {/* OTP */}
        {status==="otp" && (

        <>

        <input
          placeholder="Enter OTP"
          value={otp}
          onChange={(e)=>setOtp(e.target.value)}
          className="p-3 rounded-lg bg-white/10 border border-white/20"
        />

        <motion.button
          whileHover={{scale:1.05}}
          whileTap={{scale:0.95}}
          onClick={verifyOTP}
          className="bg-indigo-600 p-3 rounded-lg"
        >
          Verify OTP
        </motion.button>

        </>

        )}



        {/* SEND REQUEST */}
        {status==="request" && (

        <motion.button
          whileHover={{scale:1.05}}
          whileTap={{scale:0.95}}
          onClick={sendRequest}
          className="bg-green-600 p-3 rounded-lg"
        >
          Send Request to Admin
        </motion.button>

        )}



        {/* WAITING */}
        {status==="waiting" && (

        <div className="text-center">

          <Clock size={35} className="mx-auto text-yellow-400 mb-2"/>

          <h3 className="text-lg font-semibold">
            Request Sent
          </h3>

          <p className="text-gray-400 text-sm">
            Waiting for admin approval
          </p>

        </div>

        )}

        {/* REJECTED */}
{status==="rejected" && (

<div className="text-center">

  <h3 className="text-lg font-semibold text-red-400">
    Request Rejected
  </h3>

  <p className="text-gray-400 text-sm">
    Your request was rejected by admin
  </p>

  <motion.button
    whileHover={{scale:1.05}}
    whileTap={{scale:0.95}}
    onClick={()=>{
      localStorage.removeItem("teacher_email")
      setStatus("register")
    }}
    className="bg-red-600 p-3 rounded-lg mt-4"
  >
    Register Again
  </motion.button>

</div>

)}

        {status==="approved" && (

<div className="text-center">

<MailCheck size={35} className="mx-auto text-green-400 mb-2"/>

<h3 className="text-lg font-semibold">
Approved
</h3>

<p className="text-gray-400 text-sm">
Create password to continue
</p>

<motion.button
whileHover={{scale:1.05}}
whileTap={{scale:0.95}}
onClick={()=>navigate("/teacher/set-password",{
state:{email}
})}
className="bg-green-600 p-3 rounded-lg mt-4"
>
Set Password
</motion.button>

</div>

)}


        <p className="text-xs text-gray-400 text-center mt-2">
          Already approved?{" "}
          <span
            onClick={()=>navigate("/teacher/login")}
            className="text-cyan-400 cursor-pointer"
          >
            Login
          </span>
        </p>

      </motion.div>

    </div>

  );

}