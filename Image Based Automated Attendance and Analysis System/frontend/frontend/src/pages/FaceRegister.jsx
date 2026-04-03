import { useRef, useState } from "react"
import API from "../api/api"
import toast from "react-hot-toast"

export default function FaceRegister() {

  const videoRef = useRef(null)

  const [scanning, setScanning] = useState(false)
  const [images, setImages] = useState([])
  const [message, setMessage] = useState("Click Start Scan")

  // ✅ ADDED (student details)
  const [name, setName] = useState("")
  const [roll, setRoll] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")

  const MAX_IMAGES = 5

  // =========================
  // START SCAN
  // =========================
  async function startScan() {

    try {

      setImages([])
      setScanning(true)
      setMessage("⚠️ Keep your face steady...")

      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      videoRef.current.srcObject = stream

      await new Promise((resolve) => {
        videoRef.current.onloadedmetadata = () => resolve()
      })

      let count = 0

      const interval = setInterval(() => {

        if (count >= MAX_IMAGES) {

          clearInterval(interval)
          stream.getTracks().forEach(t => t.stop())

          setScanning(false)
          setMessage("✅ Scan complete — you can move now")

          return
        }

        captureFrame()
        count++

      }, 500)

    } catch (err) {
      console.error(err)
      toast.error("Camera access denied")
      setScanning(false)
    }

  }

  // =========================
  // CAPTURE FRAME
  // =========================
  function captureFrame() {

    const video = videoRef.current

    if (!video || video.videoWidth === 0) return

    const canvas = document.createElement("canvas")
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const ctx = canvas.getContext("2d")
    ctx.drawImage(video, 0, 0)

    const img = canvas.toDataURL("image/jpeg")

    setImages(prev => [...prev, img])
  }

  // =========================
  // REGISTER (UPDATED)
  // =========================
  async function registerFace() {

    try {

      if (!name || !roll || !email || !password) {
        toast.error("Please fill all details")
        return
      }

      if (images.length === 0) {
        toast.error("No images captured")
        return
      }

      setMessage("⏳ Registering face...")

      const formData = new FormData()

      // ✅ student details
      formData.append("name", name)
      formData.append("roll", roll)
      formData.append("email", email)
      formData.append("password", password)

      // ✅ images
      for (let i = 0; i < images.length; i++) {
        const blob = await fetch(images[i]).then(r => r.blob())
        formData.append("images", blob, `webcam${i}.jpg`)
      }

      // 🔥 UPDATED API
      await API.post("/register-complete", formData, {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      })

      toast.success("Face registered successfully 🎉")
      setMessage("🎉 Face registered successfully")

    } catch (err) {

      console.error(err.response?.data || err.message)

      if (err.response?.data?.detail === "Face not detected properly") {
        toast.error("Face not clear. Try again with proper lighting")
        setMessage("❌ Face not detected — retry")
      } else {
        toast.error("Registration failed")
        setMessage("❌ Registration failed")
      }

    }

  }

  // =========================
  // UI
  // =========================
  return (

    <div className="flex flex-col items-center gap-4">

      {/* 🔥 ADDED INPUTS */}
      <input placeholder="Name" onChange={(e)=>setName(e.target.value)} className="p-2 rounded bg-white/10"/>
      <input placeholder="Roll" onChange={(e)=>setRoll(e.target.value)} className="p-2 rounded bg-white/10"/>
      <input placeholder="Email" onChange={(e)=>setEmail(e.target.value)} className="p-2 rounded bg-white/10"/>
      <input type="password" placeholder="Password" onChange={(e)=>setPassword(e.target.value)} className="p-2 rounded bg-white/10"/>

      {/* CAMERA */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="w-[320px] h-[240px] rounded-full border-4 border-green-500 object-cover shadow-lg"
      />

      {/* MESSAGE */}
      <p className="text-sm text-gray-300">{message}</p>

      {/* COUNTER */}
      <p className="text-xs text-gray-400">
        {images.length}/{MAX_IMAGES} captured
      </p>

      {/* START BUTTON */}
      {!scanning && images.length === 0 && (
        <button
          onClick={startScan}
          className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg shadow"
        >
          Start Scan
        </button>
      )}

      {/* REGISTER BUTTON */}
      {images.length === MAX_IMAGES && (
        <button
          onClick={registerFace}
          className="bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded-lg shadow"
        >
          Register Face
        </button>
      )}

    </div>

  )
}