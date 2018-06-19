var request = require('request');
var express = require('express');

var accessor = {
    node_id : null,
    accessKey : "$2b$04$x6FK5mjUPbP9KM99yYF0x.gJLuQeElQb7r0kZP97eHFXDSou1TyVu",
    localPort : 6000,
    apiServerAddr : "http://localhost:3100",
    REPEAT_START_TIME : (30 * 1000),
    REPEAT_HEARTBEAT_TIME : (5 * 60 * 1000),

    repeatStart : function(delayTime){
        setTimeout(accessor.startNode , delayTime);
    },
    startNode : function(){
        console.log("service accessor start");
        var data = {node_access_key : accessor.accessKey, port : accessor.localPort};
        request.post({url: accessor.apiServerAddr + '/node/start', form: data}, function optionalCallback(error, response, body) {
            if (error) {
                //startNode 메서드 반복해야함
                accessor.repeatStart(accessor.REPEAT_START_TIME);
                return console.error('upload failed 1 :', error);
            } else {

                var rObject = JSON.parse(response.body);
                var rCode = rObject.errorCode;
                var node_id = rObject.data.node_id;

                if (rCode == 0) {
                    accessor.node_id = node_id;
                    accessor.sendHeartbeat();
                } else {
                    accessor.repeatStart(accessor.REPEAT_START_TIME);
                    return console.error('upload failed 2 :', error);
                }
            }
        });
    },
    intervalHeartBeat : null,

    sendHeartbeat : function(){
        var data = {node_access_key : accessor.accessKey, node_id : accessor.node_id, port : accessor.localPort};
        request.post({url: accessor.apiServerAddr + '/node/heartbeat', form: data}, function optionalCallback(error, response, body) {
                if (error) {
                //startNode 메서드 반복해야함 - interval 제거해야함
                if (accessor.intervalHeartBeat != null || accessor.intervalHeartBeat != undefined) {
                    clearInterval(accessor.intervalHeartBeat);
                    accessor.intervalHeartBeat = null;
                }
                //start 재시작
                accessor.repeatStart(accessor.REPEAT_START_TIME);
                return console.error('upload failed 1 :', error);
            } else {

                var rObject = JSON.parse(response.body);
                var rCode = rObject.errorCode;

                if (rCode == 0) {
                    //성공하면 그대로 계속
                    if (accessor.intervalHeartBeat == null || accessor.intervalHeartBeat == undefined) {
                        console.log("send heartbeat interval start");
                        accessor.intervalHeartBeat = setInterval(accessor.sendHeartbeat, accessor.REPEAT_HEARTBEAT_TIME);
                    } else {
                        console.log("heartbeat 진행중");
                    }

                } else {
                    //에러나면 재시작 - interval 제거해야함
                    if (accessor.intervalHeartBeat != null || accessor.intervalHeartBeat != undefined) {
                        clearInterval(accessor.intervalHeartBeat);
                        accessor.intervalHeartBeat = null;
                    }
                    accessor.repeatStart(accessor.REPEAT_START_TIME);
                    return console.error('upload failed 2 :', error);
                }
            }
        });

    },

};

module.exports = accessor;