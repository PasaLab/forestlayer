import argparse

parser = argparse.ArgumentParser()

machines = ["192.168.100.19", "192.168.100.20",  # "192.168.100.29", "192.168.100.30",
            "192.168.100.22", "192.168.100.23", "192.168.100.24", "192.168.100.25",
            "192.168.100.28", "192.168.100.31",
            "192.168.100.32", "192.168.100.33", "192.168.100.34", "192.168.100.35",
            "192.168.100.36", "192.168.100.37", "192.168.100.38", "192.168.100.39",
            ]

worker_port = "3334"
worker_port2 = "3336"
merger_port = "3335"
merger_port2 = "3337"

num_Split = 2

ps_hosts = "192.168.100.35:3333"

workers = (["{}:{}".format(machine, worker_port) for machine in machines] )
           # ["{}:{}".format(machine, worker_port2) for machine in machines])
worker_hosts = ",".join(workers)
mergers = (["{}:{}".format(workers[i].split(":")[0], merger_port) for i in range(0, len(workers), num_Split)] )
           # ["{}:{}".format(workers[i].split(":")[0], merger_port2) for i in range(0, len(workers)//2, num_Split)])
merger_hosts = ",".join(mergers)

print("PS:", ps_hosts)
print("WORKER:", worker_hosts)
print("MERGER:", merger_hosts)

with_merger = False

if not with_merger:
    with open("ps16.sh", 'w') as f:
        f.write("python trainer.py --ps_hosts={} --worker_hosts={} "
                "--job_name=ps --task_index=0 --data $1 --numSplit 2".format(ps_hosts, worker_hosts))
        f.write('\n')

    with open('worker16.sh', 'w') as f:
        f.write("python trainer.py --ps_hosts={} --worker_hosts={} "
                "--job_name=worker --task_index=$1 --data $2 --numSplit 2".format(ps_hosts, worker_hosts))
        f.write('\n')
else:
    with open("ps16_imdb.sh", 'w') as f:
        f.write("python trainer_with_merger_tf.py --ps_hosts={} --worker_hosts={} --merger_hosts={} "
                "--job_name=ps --task_index=0 --data $1 --numSplit {}".format(ps_hosts, worker_hosts, merger_hosts, num_Split))
        f.write('\n')

    with open('worker16_imdb.sh', 'w') as f:
        f.write("python trainer_with_merger_tf.py --ps_hosts={} --worker_hosts={} --merger_hosts={} "
                "--job_name=worker --task_index=$1 --data $2 --numSplit {}".format(ps_hosts, worker_hosts, merger_hosts, num_Split))
        f.write('\n')

    with open('merger16_imdb.sh', 'w') as f:
        f.write("python trainer_with_merger_tf.py --ps_hosts={} --worker_hosts={} --merger_hosts={} "
                "--job_name=merger --task_index=$1 --data $2 --numSplit {}".format(ps_hosts, worker_hosts, merger_hosts, num_Split))
        f.write('\n')
