var pg = require('pg');

var config = {

    db : {
        host: '192.168.1.67',
        user: 'postgres',
        password: '1234',
        database: 'pqms_data',
        port: '5432',
        max : 3
    }
};

var connectionTable = {
    db : null,
    config : config.db
};
var connection_init = function(){
    connectionTable.db = new pg.Pool(config.db);
    connectionTable.config = config.db
};
connection_init();

module.exports = connectionTable;
