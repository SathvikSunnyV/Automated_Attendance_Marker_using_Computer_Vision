import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import Calendar from "react-calendar"
import "react-calendar/dist/Calendar.css"

import API from "../api/api"
import TeacherSidebar from "../components/TeacherSidebar"
import toast from "react-hot-toast"

export default function TeacherTimetable(){

const [date,setDate] = useState(new Date())
const [classes,setClasses] = useState([])
const [draftClasses,setDraftClasses] = useState([])

const [subject,setSubject] = useState("")
const [period,setPeriod] = useState("")

const day = date.toLocaleDateString("en-US",{weekday:"long"})

useEffect(()=>{
fetchClasses()
},[date])

// ✅ FETCH CLASSES
async function fetchClasses(){
try{
const formattedDate = date.toISOString().split("T")[0]

const res = await API.get(
  `/timetable/day?day=${day}&current_date=${formattedDate}`
)
setClasses(res.data)
}catch(err){
console.error("Fetch error:", err)
}
}

// ✅ ADD DRAFT CLASS
function addDraftClass(){

if(!subject || !period){
alert("Enter subject and period")
return
}

setDraftClasses(prev => [
...prev,
{
subject,
period: Number(period),
day_of_week: day
}
])

setSubject("")
setPeriod("")
}

// ✅ SUBMIT CLASSES (FULL SAFE VERSION)
async function submitClasses() {

try {

const teacherId = localStorage.getItem("teacher_id")

// 🔴 HARD VALIDATION
if (!teacherId || teacherId === "null" || teacherId === "undefined") {
alert("Teacher not logged in. Please login again.")
console.error("teacher_id missing in localStorage")
return
}

const parsedTeacherId = parseInt(teacherId)

if (isNaN(parsedTeacherId)) {
alert("Invalid teacher ID")
console.error("teacher_id is NaN:", teacherId)
return
}

// 🔥 LOOP API CALL
for (const c of draftClasses) {
  
const payload = {
  subject: c.subject,
  period: Number(c.period),
  day_of_week: c.day_of_week,
  teacher_id: parsedTeacherId,
  start_date: date.toISOString().split("T")[0] // ✅ ADD THIS
}

console.log("🚀 Sending:", payload)

await API.post("/teacher/add-class", payload)

}

toast.success("Classes added successfully ✅")

setDraftClasses([])
fetchClasses()

} catch (err) {

console.error("❌ ERROR:", err.response?.data || err.message)

alert(
err.response?.data?.detail
? JSON.stringify(err.response.data.detail)
: "Something went wrong"
)

}
}

// ✅ DELETE CLASS
async function deleteClass(id){
try{
await API.delete(`/teacher/delete-class/${id}`)
fetchClasses()
}catch(err){
console.error(err)
}
}

return(

<div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-800 text-white relative overflow-hidden">

{/* Background */}
<div className="absolute inset-0 -z-10">
  <div className="absolute w-[700px] h-[700px] bg-blue-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]"></div>
  <div className="absolute w-[600px] h-[600px] bg-cyan-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]"></div>
</div>

<TeacherSidebar/>

<div className="flex-1 p-10">

<motion.h1
initial={{opacity:0,y:-20}}
animate={{opacity:1,y:0}}
className="text-4xl font-bold mb-8"
>
Teacher Timetable
</motion.h1>

<div className="grid md:grid-cols-2 gap-10">

{/* CALENDAR */}
<motion.div
initial={{opacity:0,x:-20}}
animate={{opacity:1,x:0}}
className="bg-white/5 p-6 rounded-3xl"
>
<Calendar onChange={setDate} value={date}/>
</motion.div>

{/* PANEL */}
<motion.div
initial={{opacity:0,x:20}}
animate={{opacity:1,x:0}}
className="bg-white/5 p-8 rounded-3xl"
>

<h2 className="text-xl mb-6">
{day} Classes
</h2>

{/* ADD CLASS */}
<div className="flex gap-4 mb-6">

<input
placeholder="Subject"
value={subject}
onChange={(e)=>setSubject(e.target.value)}
className="bg-white/10 p-3 rounded-lg w-full"
/>

<select
value={period}
onChange={(e)=>setPeriod(e.target.value)}
className="bg-blue-900 p-3 rounded-lg"
>
<option value="">Period</option>
{[1,2,3,4,5,6,7,8].map(p=>(
<option key={p} value={p}>{p}</option>
))}
</select>

<motion.button
whileHover={{scale:1.05}}
whileTap={{scale:0.95}}
onClick={addDraftClass}
className="bg-blue-600 px-5 rounded-lg"
>
Add
</motion.button>

</div>

{/* PENDING */}
{draftClasses.length>0 &&(

<div className="bg-blue-900/30 p-4 rounded-xl mb-6">

<p className="mb-3 text-cyan-300">Pending Classes</p>

{draftClasses.map((c,i)=>(
<div key={i} className="flex justify-between mb-2 text-sm">
<span>Period {c.period} — {c.subject}</span>
</div>
))}

<motion.button
whileHover={{scale:1.05}}
whileTap={{scale:0.95}}
onClick={submitClasses}
className="mt-3 bg-green-600 px-4 py-2 rounded-lg"
>
Submit Timetable
</motion.button>

</div>

)}

{/* TABLE */}
<table className="w-full text-sm">

<thead>
<tr>
<th className="py-3 text-left">Period</th>
<th>Subject</th>
<th>Action</th>
</tr>
</thead>

<tbody>

{classes.length === 0 && (
<tr>
<td colSpan="3">No classes scheduled</td>
</tr>
)}

{classes.map((c) => (
<tr key={c.id}>
<td className="py-3">{c.period}</td>
<td>{c.subject}</td>
<td>
<button
onClick={() => deleteClass(c.id)}
className="bg-red-500 px-3 py-1 rounded"
>
Delete
</button>
</td>
</tr>
))}

</tbody>

</table>

</motion.div>

</div>

</div>

</div>

)
}