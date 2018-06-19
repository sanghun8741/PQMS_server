var pythonShell = require('python-shell');
var path = require('path');
var Promise = require('promise');
var appDir = path.dirname(require.main.filename);


var pythonScriptRunner = {

    pqms_run : function(param){
        return new Promise(function(resolve, reject){
            console.log("algo promise start");
            const option = {
                'mode' : 'text',
                'pythonPath' : '',
                'pythonOptions' : ['-u'],
                'scriptPath' : '',
                'args' : param
                //'args' : ['192.168.1.67', 'postgres', '1234', 'pqms_data', 1, 10, '2018-02-01 15:00:00', 0]
                //db_host, db_userId, db_password, db_database, subs_id, dl_id, time, algoType
            };
            const scriptPath = __dirname + "/pqms/PQMS_LSTM_forecasting.py";
            pythonShell.run(scriptPath, option, function(err, results){
                console.log("----------python run-----------");
                if(err) {
                    console.log(err);
                    var returnDict = {};
                    returnDict.returnCode = 0;
                    returnDict.output = [];
                    resolve(returnDict);
                }
                //console.log(results);
                console.log(results);
                const r = JSON.parse(results);
                console.log("----------python end-----------");
                resolve(r);

            });
        });



    }
}

module.exports = pythonScriptRunner;