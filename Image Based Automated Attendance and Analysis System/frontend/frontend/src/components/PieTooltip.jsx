function PieTooltip({active,payload,data}){

if(!active || !payload?.length) return null

const item = payload[0]
const total = data.reduce((sum,d)=>sum + d.value,0)

const percentage = ((item.value / total) * 100).toFixed(1)

return(

<div className="bg-slate-900 border border-slate-700 p-3 rounded-xl shadow-lg">

<p className="text-gray-400 text-xs mb-1">
{item.name}
</p>

<p className="font-bold text-white">
{item.value} ({percentage}%)
</p>

</div>

)

}