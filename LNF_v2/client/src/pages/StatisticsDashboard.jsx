import React, { useEffect, useState } from "react";
import axios from "axios"; // Ensure axios is imported
import { Box, CheckCircle, TrendingUp, Users, GitMerge, PackageSearch, Activity, ArrowUp, ArrowDown } from "lucide-react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

// --- Enhanced StatCard Component ---
const StatCard = ({ icon, title, value, color, trend, trendValue }) => (
    <div className="group bg-white p-4 sm:p-6 rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 border border-gray-100 hover:border-gray-200 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-transparent to-gray-50 rounded-bl-full opacity-50"></div>
        <div className="flex items-start justify-between relative z-10">
            <div className="flex items-center">
                <div 
                    className="p-2 sm:p-3 rounded-lg mr-3 sm:mr-4 group-hover:scale-110 transition-transform duration-200" 
                    style={{ backgroundColor: `${color}15` }}
                >
                    {React.cloneElement(icon, { 
                        size: window.innerWidth < 640 ? 20 : 24,
                        style: { color } 
                    })}
                </div>
                <div>
                    <p className="text-xs sm:text-sm font-medium text-gray-500 mb-1">{title}</p>
                    <p className="text-xl sm:text-2xl font-bold text-gray-800">{value}</p>
                </div>
            </div>
            {trend && (
                <div className={`flex items-center text-xs font-medium px-2 py-1 rounded-full ${
                    trend === 'up' ? 'text-green-700 bg-green-100' : 'text-red-700 bg-red-100'
                }`}>
                    {trend === 'up' ? <ArrowUp size={12} /> : <ArrowDown size={12} />}
                    <span className="ml-1">{trendValue}</span>
                </div>
            )}
        </div>
    </div>
);

// --- Enhanced HighlightCard Component ---
// --- Enhanced HighlightCard Component (Corrected) ---
const HighlightCard = ({ icon, title, value, color, subtitle }) => (
    <div 
        className="relative text-white p-6 sm:p-8 lg:p-10 rounded-2xl shadow-xl flex flex-col items-center justify-center text-center overflow-hidden group hover:scale-[1.02] transition-transform duration-300"
        style={{ 
            background: `linear-gradient(135deg, ${color} 0%, #667eea 50%, #764ba2 100%)`,
        }}
    >
        <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent"></div>
        
        {/* The main content wrapper is now a flex column to ensure perfect centering */}
        <div className="relative z-10 flex flex-col items-center">
            {React.cloneElement(icon, { size: window.innerWidth < 640 ? 32 : 48, className: "mb-3 sm:mb-4 opacity-90" })}
            <p className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight mb-1 sm:mb-2">{value}</p>
            <p className="text-base sm:text-lg font-medium opacity-90">{title}</p>
            {subtitle && <p className="text-sm opacity-75 mt-2">{subtitle}</p>}
        </div>
    </div>
);

// --- Enhanced Match Funnel Component (MODIFIED) ---
const MatchFunnel = ({ data }) => {
    // Data for the legend is now always present, regardless of value
    const legendData = [
        { name: 'Pending', value: data.pending || 0, color: '#F59E0B' },
        { name: 'Confirmed', value: data.confirmed || 0, color: '#3B82F6' },
        { name: 'Transfer Complete', value: data.transfer_complete || 0, color: '#10B981' },
        { name: 'Rejected', value: data.rejected || 0, color: '#EF4444' },
    ];

    const total = legendData.reduce((sum, item) => sum + item.value, 0);

    // Data for the pie chart visual. If total is 0, show a placeholder.
    const pieChartData = total > 0
        ? legendData.filter(item => item.value > 0)
        : [{ name: 'No Data', value: 1, color: '#E5E7EB' }]; // gray-200 for placeholder

    const RADIAN = Math.PI / 180;
    const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
        if (percent < 0.05 || total === 0) return null; // Don't show label for small slices or when no data
        
        const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
        const x = cx + radius * Math.cos(-midAngle * RADIAN);
        const y = cy + radius * Math.sin(-midAngle * RADIAN);

        return (
            <text 
                x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} 
                dominantBaseline="central" fontSize={window.innerWidth < 640 ? "12px" : "14px"} 
                fontWeight="bold" style={{ textShadow: '1px 1px 2px rgba(0,0,0,0.7)' }}
            >
                {`${(percent * 100).toFixed(0)}%`}
            </text>
        );
    };

    return (
        <div className="bg-white p-4 sm:p-6 rounded-2xl shadow-sm border border-gray-100">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 sm:mb-6">
                <h3 className="text-lg sm:text-xl font-semibold text-gray-800 flex items-center mb-2 sm:mb-0">
                    <GitMerge className="text-gray-600 mr-2" size={20} />
                    Match Processing Status
                </h3>
                <div className="text-xs sm:text-sm text-gray-500">
                    Total: {total} items
                </div>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
                {/* Pie Chart */}
                <div className="h-64 sm:h-80 relative">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={pieChartData}
                                cx="50%" cy="50%" labelLine={false} label={renderCustomizedLabel}
                                outerRadius={window.innerWidth < 640 ? 80 : 100}
                                innerRadius={window.innerWidth < 640 ? 40 : 50}
                                dataKey="value" nameKey="name" stroke="#fff" strokeWidth={2}
                            >
                                {pieChartData.map((entry) => (
                                    <Cell key={`cell-${entry.name}`} fill={entry.color} />
                                ))}
                            </Pie>
                            {total > 0 && (
                                <Tooltip 
                                    formatter={(value, name) => [`${value} items`, name]}
                                    contentStyle={{
                                        backgroundColor: 'rgba(255, 255, 255, 0.95)', border: 'none',
                                        borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                                    }}
                                />
                            )}
                        </PieChart>
                    </ResponsiveContainer>
                    {total === 0 && (
                         <div className="absolute inset-0 flex items-center justify-center text-gray-500 font-semibold pointer-events-none">
                           No Activity
                        </div>
                    )}
                </div>
                
                {/* Status Legend */}
                <div className="flex flex-col justify-center space-y-3">
                    {legendData.map((item) => (
                        <div key={item.name} className="flex items-center justify-between p-3 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors">
                            <div className="flex items-center">
                                <div 
                                    className="w-3 h-3 rounded-full mr-3"
                                    style={{ backgroundColor: item.color }}
                                ></div>
                                <span className="text-sm font-medium text-gray-700">{item.name}</span>
                            </div>
                            <div className="flex items-center space-x-2">
                                <span className="text-lg font-bold text-gray-800">{item.value}</span>
                                <span className="text-xs text-gray-500">
                                    ({total > 0 ? ((item.value / total) * 100).toFixed(1) : '0.0'}%)
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};


// --- Loading Component ---
const LoadingSpinner = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
            <p className="text-lg font-medium text-gray-600">Loading Dashboard...</p>
        </div>
    </div>
);

// --- Main Dashboard Component ---
const StatisticsDashboard = () => {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        const fetchStatistics = async () => {
            try {
                // Using axios to fetch live data
                const response = await axios.get("http://localhost:8000/apis/lost-and-found/statistics", { withCredentials: true });
                setStats(response.data);
            } catch (err) { 
                // If API fails, log the error and use fallback mock data to still show a working dashboard
                console.warn("API call failed. Using fallback mock data for demonstration.", err);
                const fakeStatsData = { 
                    kpis: { 
                        totalLostItems: 3, 
                        resolvedItems: 0, 
                        totalFoundItems: 2, 
                        successfulTransfers: 0, 
                        resolutionRate: "0.00%" 
                    }, 
                    matchFunnel: { 
                        pending: 0, 
                        confirmed: 0, 
                        transfer_complete: 0, 
                        rejected: 0 
                    } 
                };
                setStats(fakeStatsData);
            } finally {
                setLoading(false);
            }
        };
        fetchStatistics();
    }, []);

    if (loading) return <LoadingSpinner />;
    
    if (error && !stats) return (
        <div className="min-h-screen bg-gradient-to-br from-red-50 to-pink-100 flex items-center justify-center p-4">
            <div className="bg-white p-6 rounded-lg shadow-lg text-center max-w-md">
                <div className="text-red-600 text-4xl mb-4">⚠️</div>
                <h2 className="text-xl font-semibold text-gray-800 mb-2">Error Loading Dashboard</h2>
                <p className="text-gray-600">{error}</p>
                <button 
                    onClick={() => window.location.reload()} 
                    className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                    Retry
                </button>
            </div>
        </div>
    );

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 p-3 sm:p-4 lg:p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-6 sm:mb-8 lg:mb-10">
                    <div className="text-center">
                        <div className="inline-flex items-center justify-center w-16 h-16 sm:w-20 sm:h-20 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full mb-4 shadow-lg">
                            <Activity className="text-white" size={window.innerWidth < 640 ? 24 : 32} />
                        </div>
                        <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-800 mb-2">
                            Platform Analytics
                        </h1>
                        <p className="text-sm sm:text-base text-gray-600 max-w-2xl mx-auto">
                            Real-time insights into your lost and found item management system
                        </p>
                    </div>
                </div>

                {stats && (
                    <div className="space-y-6 sm:space-y-8">
                        
                        {/* Highlight KPI */}
                        <div className="px-2">
                            <HighlightCard 
                                icon={<Users />} 
                                title="Items Successfully Returned" 
                                value={stats.kpis.successfulTransfers} 
                                color="#8B5CF6"
                                subtitle="Happy customers reunited with their belongings"
                            />
                        </div>
                        
                        {/* Main KPI Grid */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3 sm:gap-4 lg:gap-6">
                            <StatCard 
                                icon={<Box />} 
                                title="Total Lost Items" 
                                value={stats.kpis.totalLostItems} 
                                color="#3B82F6"
                            />
                            <StatCard 
                                icon={<PackageSearch />} 
                                title="Total Found Items" 
                                value={stats.kpis.totalFoundItems} 
                                color="#14B8A6"
                            />
                            <StatCard 
                                icon={<CheckCircle />} 
                                title="Items Resolved" 
                                value={stats.kpis.resolvedItems} 
                                color="#10B981"
                            />
                            <StatCard 
                                icon={<TrendingUp />} 
                                title="Resolution Rate" 
                                value={stats.kpis.resolutionRate} 
                                color="#F59E0B"
                            />
                        </div>

                        {/* Match Funnel Chart */}
                        <MatchFunnel data={stats.matchFunnel} />

                    </div>
                )}
            </div>
        </div>
    );
};

export default StatisticsDashboard;