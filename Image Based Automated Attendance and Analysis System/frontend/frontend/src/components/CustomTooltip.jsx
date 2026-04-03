function CustomTooltip({active,payload}){

if(active && payload && payload.length){

return(

<div className="bg-slate-900 border border-indigo-400 px-4 py-2 rounded-lg shadow-lg">

<p className="text-gray-300 text-sm">
{payload[0].name}
</p>

<p className="text-indigo-400 font-semibold">
{payload[0].value} Classes
</p>

</div>

)

}

return null
}