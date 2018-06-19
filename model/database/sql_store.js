var store = {
    getPqmsBeforeData : "SELECT load, to_char(time, 'YYYY-MM-DD HH24:MI:SS') as time FROM public.data WHERE time <= $1 ORDER BY time desc LIMIT 12",
    getPqmsAfterData : "SELECT load, to_char(time, 'YYYY-MM-DD HH24:MI:SS') as time FROM public.data WHERE time >= $1 ORDER BY time desc LIMIT $2 "
};

module.exports = store;