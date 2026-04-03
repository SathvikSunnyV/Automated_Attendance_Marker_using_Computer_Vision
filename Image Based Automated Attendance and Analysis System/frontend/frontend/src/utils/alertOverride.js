import toast from "react-hot-toast"

export function overrideAlerts(){

  // 🔥 REPLACE alert()
  window.alert = (message) => {
    toast.error(message, {
      duration: 3000
    })
  }

}