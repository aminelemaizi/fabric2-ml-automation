import json
from fabric2 import Connection, Config, task


SERVER_USER = 'lem'
SERVER_HOST = '192.168.56.100'
SERVER_KEY = 'assets/server_ssh_key'

with open("assets/secrets.json") as f:
    ENV_VALUES = json.load(f)
    f.close()


def connect_to_host(ctx):
    if isinstance(ctx, Connection):
        return ctx
    else:
        password = ENV_VALUES['password']
        config = Config(overrides={'sudo': {'password': password}})
        conn = Connection(ctx.host, ctx.user, connect_kwargs=ctx.connect_kwargs, config=config)
        return conn

@task
def remote(ctx):
    ctx.user = SERVER_USER
    ctx.host = SERVER_HOST
    ctx.connect_kwargs.key_filename = SERVER_KEY
    ctx.connect_kwargs.passphrase = ENV_VALUES['passphrase']

@task
def prepare_os(ctx):
    remote(ctx)
    conn = connect_to_host(ctx)
    conn.sudo("apt-get update")
    conn.sudo("apt-get install -y python3 python3-pip python3-dev python3-venv")        
    conn.sudo("apt-get install -y build-essential libssl-dev libffi-dev libpq-dev")
    conn.sudo("apt-get install -y zip unzip")
    conn.close()


@task
def prepare_menv(ctx):
    remote(ctx)
    conn = connect_to_host(ctx)
    conn.local("zip -r ml_assets.zip ml_assets")
    conn.put('ml_assets.zip', '/home/{}/'.format(SERVER_USER))
    conn.run("unzip -o -q ml_assets.zip")
    conn.run("rm ml_assets.zip")
    conn.local("rm ml_assets.zip")
    conn.run("python3 -m venv modelenv")
    conn.run("./modelenv/bin/pip3 install --upgrade pip")
    conn.run("./modelenv/bin/pip3 install -r ./ml_assets/Requirements.txt")
    conn.run("./modelenv/bin/pip3 freeze")
    conn.close()


@task
def train_model(ctx, dataset, epochs):
    remote(ctx)
    conn = connect_to_host(ctx)
    conn.run("./modelenv/bin/python ./ml_assets/model.py train {0} {1}".format(dataset, epochs))
    conn.get("model.h5", "./ml_assets/model.h5")
    conn.run("rm model.h5")
    conn.close()

@task
def predict_data(ctx, dataset):
    remote(ctx)
    conn = connect_to_host(ctx)

    conn.put(dataset, '/home/{}/'.format(SERVER_USER))
    conn.put('./ml_assets/model.h5', '/home/{}/'.format(SERVER_USER))
    conn.run("mkdir -p autoencoded")
    conn.run("./modelenv/bin/python ./ml_assets/model.py predict {}".format(dataset))
    conn.run("zip -r autoencoded.zip autoencoded")
    conn.get('autoencoded.zip', 'autoencoded.zip'.format(SERVER_USER))
    conn.run("rm -rf autoencoded")
    conn.run("rm autoencoded.zip")
    conn.run("rm model.h5")
    conn.run("rm {}".format(dataset))
    conn.local("unzip -o -q autoencoded.zip")
    conn.local("rm autoencoded.zip")

    conn.close()


