# MongoDB Replication

## Creating a Node at port 27021
```bash
mkdir -p $HOME/mongo/data/db01
mongod --replSet dbrs --port 27021 --dbpath $HOME/mongo/data/db01
```

## Repeat the same for another two ports with other terminal sessions
```bash
mkdir -p $HOME/mongo/data/db02
mongod --replSet dbrs --port 27022 --dbpath $HOME/mongo/data/db02
```

```bash
mkdir -p $HOME/mongo/data/db03
mongod --replSet dbrs --port 27023 --dbpath $HOME/mongo/data/db03
```

## Connect to any MongoDB Daemon
```bash
mongo --port 27021
```

## Paste Config to Terminal
```bash
rsconf = {
    _id: "dbrs",
    members: [
        {
            _id: 0,
            host: "127.0.0.1:27021",
            priority: 3
        },
        {
            _id: 1,
            host: "127.0.0.1:27022",
            priority: 1,
        },
        {
            _id: 2,
            host: "127.0.0.1:27023",
            priority: 2
        }
    ]
};
db.getMongo().setReadPref('secondary')
```

```bash
rs.initiate(rsconf);

{ "ok" : 1 }
```