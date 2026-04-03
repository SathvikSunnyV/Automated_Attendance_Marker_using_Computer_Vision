import { useEffect, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"

export default function ConfirmModal(){

  const [open,setOpen] = useState(false)
  const [message,setMessage] = useState("")
  const [resolvePromise,setResolvePromise] = useState(null)

  useEffect(()=>{

    window.confirm = (msg) => {
      setMessage(msg)
      setOpen(true)

      return new Promise((resolve)=>{
        setResolvePromise(() => resolve)
      })
    }

  },[])

  function handle(result){
    setOpen(false)
    if(resolvePromise) resolvePromise(result)
  }

  return(
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          initial={{opacity:0}}
          animate={{opacity:1}}
          exit={{opacity:0}}
        >

          <motion.div
            initial={{scale:0.8,opacity:0}}
            animate={{scale:1,opacity:1}}
            exit={{scale:0.8,opacity:0}}
            className="bg-slate-900 text-white p-8 rounded-2xl w-80 border border-white/10"
          >

            <h2 className="text-lg font-semibold mb-4">
              Confirm Action
            </h2>

            <p className="text-gray-300 mb-6">
              {message}
            </p>

            <div className="flex justify-end gap-3">

              <button
                onClick={()=>handle(false)}
                className="px-4 py-2 bg-gray-600 rounded-lg"
              >
                Cancel
              </button>

              <button
                onClick={()=>handle(true)}
                className="px-4 py-2 bg-red-500 rounded-lg"
              >
                Confirm
              </button>

            </div>

          </motion.div>

        </motion.div>
      )}
    </AnimatePresence>
  )
}