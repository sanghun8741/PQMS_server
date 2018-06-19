var express = require('express');
var router = express.Router();

var dbConfig = require('../config/database.js');
var sql_executer = require('../model/database/sql_executer');
var sql_store = require('../model/database/sql_store');
var algorithmRunner = require('../model/algorithm/accessor');

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});


router.post('/pqms', function(req, res, next) {
    //1. 파라미터 확인
    //2.
    const db_host = dbConfig.config.host;
    const db_userId = dbConfig.config.user;
    const db_password = dbConfig.config.password;
    const db_database = dbConfig.config.database;
    const db_port = dbConfig.config.port;

    //const subs_id = req.body.subs_id;
    //const dl_id = req.body.dl_id;
    //const time = req.body.time;
    //const algo_type = req.body.algo_type;

    const subs_id = 1;
    const dl_id = 10;
    const time = '2018-02-01 15:00:00';
    const algo_type = req.body.algo_type;

    //const param = [db_host, db_userId, db_password, db_database, db_port, subs_id, dl_id, time, algoType]; 실제 사용할 코드
    const param = [db_host, db_userId, db_password, db_database, db_port, subs_id, dl_id, time, algo_type]; //더미용 코드

    const pqmsAlgorithmPromise = algorithmRunner.pqms_run(param);
    pqmsAlgorithmPromise.then(function(resolve){
        //여기서 기준 시간가지고 이전으로 1시간~2시간 정도 쿼리 날려서 데이터 가져오도록 하자
        var pqmsBeforeDataPromise = sql_executer.query(dbConfig.db, sql_store.getPqmsBeforeData, [time]);
        pqmsBeforeDataPromise.then(function(resolve2){
            resolve.input = resolve2;
            for (var i in resolve.output) {
                resolve.output[i].load = parseFloat(resolve.output[i].load);
            }
            console.log(resolve);
            res.json(resolve);

            var limit;
            if (algo_type == 0) {limit = 1;}
            else if (algo_type == 1) { limit = 6;}
            else { limit = 18; }
            //추가적으로 기준시간 이전이 아닌 미래의 값을 algo_type에 따라서 가져와야함
            var pqmsAfterDataPromise = sql_executer.query(dbConfig.db, sql_store.getPqmsAfterData, [time, limit]);
            pqmsAfterDataPromise.then(function(reslove3){
                resolve.realOutput = reslove3;

                //가져온 값을 가지고 NRMSE 돌려서 정확도도 계산해야함
                //resolve.output 이랑 resolve.realOutput이랑 비교 load, time
                //var
                //for (var i in resolve.output) {
                //    resolve.output[i].load
                //}

                res.json(resolve);
            });




        });

    });
    /*
    'args' : ['192.168.1.67', 'postgres', '1234', 'pqms_data', 1, 10, '2018-02-01 15:00:00', 0]
    //db_host, db_userId, db_password, db_database, subs_id, dl_id, time, algoType
     */

});

module.exports = router;
