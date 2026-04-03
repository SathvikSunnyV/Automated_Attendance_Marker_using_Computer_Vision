import StatsCard from "../components/StatsCard";

export default function Dashboard() {

  return (

    <div className="flex-1 p-10">

      <h1 className="text-4xl font-bold mb-8 text-gray-700">
        Teacher Dashboard
      </h1>

      <div className="grid grid-cols-4 gap-8">

        <StatsCard
          title="Total Students"
          value="65"
          gradient="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500"
        />

        <StatsCard
          title="Present Today"
          value="52"
          gradient="bg-gradient-to-r from-green-400 to-emerald-600"
        />

        <StatsCard
          title="Absent"
          value="13"
          gradient="bg-gradient-to-r from-red-400 to-pink-500"
        />

        <StatsCard
          title="Attendance %"
          value="80%"
          gradient="bg-gradient-to-r from-cyan-400 to-blue-500"
        />

      </div>

    </div>

  );
}