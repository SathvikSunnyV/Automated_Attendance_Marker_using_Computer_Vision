import { useState, useEffect } from "react"
import Calendar from "react-calendar"
import "react-calendar/dist/Calendar.css"

import API from "../api/api"
import TeacherSidebar from "../components/TeacherSidebar"
import toast from "react-hot-toast"

export default function TeacherAttendance(){

const [date,setDate] = useState(new Date())
const [classes,setClasses] = useState([])
const [selectedClass,setSelectedClass] = useState(null)

const [mode,setMode] = useState(null)
const [image,setImage] = useState(null)
const [students,setStudents] = useState([])

const [loading,setLoading] = useState(false)
const [submitting,setSubmitting] = useState(false)
const [attendanceStatus,setAttendanceStatus] = useState({})

const day = date.toLocaleDateString("en-US",{weekday:"long"})

useEffect(()=>{
fetchClasses()
fetchAttendanceStatus() 
},[date])

async function fetchClasses(){
  try{

    const formattedDate = date.toISOString().split("T")[0]  // ✅ IMPORTANT

    const res = await API.get(
      `/timetable/day?day=${day}&current_date=${formattedDate}`
    )

    setClasses(res.data)

  }catch(err){
    toast.error("Failed to load timetable")
    console.error(err)
  }
}

function resetAttendance(){
setStudents([])
setImage(null)
setMode(null)
setSelectedClass(null)
}

async function uploadImage(){

if(!selectedClass){
toast.error("Please select a class")
return
}

if(!image){
toast.error("Please upload an image")
return
}

try{

setLoading(true)

const formData = new FormData()
formData.append("image",image)
formData.append("class_id",selectedClass.id)

const res = await API.post("/detect-faces",formData)

const detected = res.data.recognized_students || []

const updated = detected.map(s=>({
student_id:s.student_id,
roll_no:s.roll_no || s.rollNo || "N/A",
name:s.name,
status:s.status || "Present"
}))

setStudents(updated)

}catch(err){
toast.error("Face detection failed")
}finally{
setLoading(false)
}

}

function toggleStatus(studentId,newStatus){
setStudents(prev =>
prev.map(student =>
student.student_id === studentId
? {...student,status:newStatus}
: student
)
)
}

async function fetchAttendanceStatus(){
try{
const res = await API.get(`/attendance/status?date=${date.toLocaleDateString("en-CA")}`)
setAttendanceStatus(res.data)
}catch(err){
console.log(err)
}
}

async function submitAttendance(){

if(students.length === 0){
toast.error("No students to submit")
return
}

try{

setSubmitting(true)

const payload = students.map(s=>({
student_id:s.student_id,
subject:selectedClass.subject,
date: date.toLocaleDateString("en-CA"),
period:selectedClass.period,
status:s.status
}))

await API.post("/submit-attendance",payload)

toast.success("Attendance submitted successfully")

resetAttendance()
fetchAttendanceStatus()

}catch(err){
toast.error("Failed to submit attendance")
}finally{
setSubmitting(false)
}

}

async function loadAttendanceForUpdate(cls){
try{
const res = await API.get(
`/attendance/update?date=${date.toLocaleDateString("en-CA")}&period=${cls.period}`
)

setStudents(res.data)
setSelectedClass(cls)
setMode("update")

}catch(err){
toast.error("Failed to load attendance")
}
}

return(

<div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-800 text-white relative overflow-hidden">

{/* 🔥 background glow */}
<div className="absolute inset-0 -z-10">
  <div className="absolute w-[700px] h-[700px] bg-blue-500 opacity-20 blur-[160px] top-[-200px] left-[-200px]"></div>
  <div className="absolute w-[600px] h-[600px] bg-cyan-500 opacity-20 blur-[160px] bottom-[-200px] right-[-200px]"></div>
</div>

<TeacherSidebar/>

<div className="flex-1 p-10">

<h1 className="text-4xl font-bold mb-8 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
Take Attendance
</h1>

<div className="grid md:grid-cols-2 gap-10 items-start">

<div className="bg-white/5 backdrop-blur-xl border border-white/10 p-4 rounded-2xl w-fit">

 <div className="flex justify-center">
  <div className="scale-[0.9] origin-top">
    <Calendar onChange={setDate} value={date}/>
  </div>
</div>

</div>

{/* CLASSES */}
<div className="bg-white/5 backdrop-blur-2xl border border-white/10 p-8 rounded-3xl shadow-lg">

<h2 className="text-xl mb-6 text-gray-300">
Classes for {day}
</h2>

{classes.length===0 &&(
<p className="text-white/50">No classes scheduled</p>
)}

{classes.map(c => {

const isDone = attendanceStatus[c.period]

return(

<div
key={c.id}
className={`p-4 mb-3 rounded-xl border transition
${selectedClass?.id===c.id
? "bg-blue-500/20 border-cyan-400"
: "bg-white/5 border-white/10 hover:bg-white/10"}`}
>

<div className="flex justify-between items-center">

<div className="flex flex-col">

<span>{c.subject} — Period {c.period}</span>

<span className={`text-xs mt-1 ${
isDone ? "text-red-400" : "text-green-400"
}`}>
{isDone ? "Attendance Already Taken" : "Not Taken"}
</span>

</div>

<div className="flex gap-3">

<button
disabled={isDone}
className={`px-3 py-1 rounded text-sm ${
isDone
? "bg-gray-500 cursor-not-allowed"
: "bg-blue-600 hover:bg-blue-500"
}`}
onClick={()=>{
setSelectedClass(c)
setMode("take")
}}
>
{isDone ? "Completed" : "Take"}
</button>

<button
className="bg-yellow-500 px-3 py-1 rounded text-sm"
onClick={()=>loadAttendanceForUpdate(c)}
>
Update
</button>

</div>

</div>

</div>

)

})}

</div>

</div>

{/* IMAGE UPLOAD */}
{mode==="take" && selectedClass &&(

<div className="mt-10 bg-white/5 backdrop-blur-2xl border border-white/10 p-8 rounded-3xl shadow-lg">

<h2 className="text-xl mb-4">Upload Classroom Photo</h2>

<label className={`flex flex-col items-center justify-center gap-2 p-5 rounded-xl 
bg-gradient-to-r from-blue-500/20 to-cyan-500/20 
border border-dashed cursor-pointer transition-all
${image ? "border-green-400 bg-green-500/10" : "border-white/30"}
hover:scale-[1.02] hover:border-blue-400`}>

  <span className="text-sm text-gray-300">
    {image ? " Image uploaded" : "Upload Classroom Photo"}
  </span>

  <input
    type="file"
    accept="image/*"
    className="hidden"
    onChange={(e)=>setImage(e.target.files[0])}
  />

</label>

<button
onClick={uploadImage}
disabled={loading}
className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg mt-4"
>
{loading ? "Detecting..." : "Detect Students"}
</button>

</div>

)}

{/* TABLE */}
{students.length>0 &&(

<div className="mt-10 bg-white/5 backdrop-blur-2xl border border-white/10 p-8 rounded-3xl shadow-lg">

<h2 className="text-xl mb-2">Attendance Preview</h2>

<p className="text-white/60 mb-6">
Total Students: {students.length}
</p>

<table className="w-full text-left">

<thead>
<tr className="text-white/60 border-b border-white/10">
<th className="py-3">Roll No</th>
<th>Name</th>
<th>Status</th>
</tr>
</thead>

<tbody>

{students.map((s)=>(

<tr key={s.student_id} className="border-b border-white/10 hover:bg-white/5">

<td className="py-4">{s.roll_no}</td>
<td>{s.name}</td>

<td>
<div className="flex bg-white/10 rounded-full p-1 w-fit">

<button
onClick={()=>toggleStatus(s.student_id,"Present")}
className={`px-4 py-1 rounded-full text-sm ${
s.status==="Present"
? "bg-green-500 text-white"
: "text-gray-300"
}`}
>
Present
</button>

<button
onClick={()=>toggleStatus(s.student_id,"Absent")}
className={`px-4 py-1 rounded-full text-sm ${
s.status==="Absent"
? "bg-red-500 text-white"
: "text-gray-300"
}`}
>
Absent
</button>

</div>
</td>

</tr>

))}

</tbody>

</table>

<button
onClick={submitAttendance}
disabled={submitting}
className="mt-8 bg-green-600 hover:bg-green-500 px-8 py-3 rounded-lg"
>
{submitting ? "Submitting..." : "Submit Attendance"}
</button>

</div>

)}

</div>

</div>

)

}