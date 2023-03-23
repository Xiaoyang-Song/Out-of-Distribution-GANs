gpu_usage() {
    if [ $# -eq 0 ]; then
        echo "The partition name needs to be supplied"
        return 1
    fi
    
    join \
    -a 1 \
    -o auto \
    <(sinfo \
        --partition $1 \
        --Node \
        --noheader \
        -O NodeList:7,CPUsState:12,AllocMem:7,Memory:7,Gres:11,StateCompact:6,GresUsed:.10
    ) \
    <(squeue \
        -t RUNNING \
        --partition $1 \
        --noheader \
        -o "%N %u %a" \
        | python -c "
import fileinput
import subprocess
from collections import defaultdict

# From https://github.com/NERSC/pytokio/blob/fdb8237/tokio/connectors/slurm.py#L81
def node_spec_to_list(node_spec):
  return subprocess.check_output(['scontrol', 'show', 'hostname', node_spec]).decode().strip().split()

users_by_node = defaultdict(list)

for line in fileinput.input():
  line = line.strip()

  node_spec, u, a = line.split()

  for node in node_spec_to_list(node_spec):
    users_by_node[node].append('{}({})'.format(u, a))

for node, node_info in sorted(users_by_node.items(), key=lambda t: t[0]):
  print('{} {}'.format(node, ','.join(node_info)))
    ") \
    | awk \
    'BEGIN{
      total_cpus_alloc = 0;
      total_cpus = 0;
      total_mem_alloc = 0;
      total_mem = 0;
      total_gpus_alloc = 0;
      total_gpus = 0;

      printf("%6s %5s %10s %9s %10s %s\n", "NODE", "STATE", "ALLOC_CPUS", "ALLOC_MEM", "ALLOC_GPUS", "USERS")
    };
    {
      split($2, cpu, "/");
      split($5, gres, ":");
      split($7, gres_used, ":");

      node = $1;

      state = $6;

      cpus_alloc = cpu[1];
      total_cpus_alloc += cpus_alloc;

      cpus = cpu[4];
      total_cpus += cpus;

      mem_alloc = $3 / 1024;
      total_mem_alloc += mem_alloc;

      mem = $4 / 1024;
      total_mem += mem;

      gpus_alloc = gres_used[3];
      total_gpus_alloc += gpus_alloc;

      gpus = gres[3];
      total_gpus += gpus;

      users = $8;

      printf("%6s %5s %6d/%3d %4d/%4d %7d/%2d %s\n", node, state, cpus_alloc, cpus, mem_alloc, mem, gpus_alloc, gpus, users)
    };
    END{
      printf("%6s %5s %6d/%3d %4d/%4d %7d/%2d\n", "TOTAL", "", total_cpus_alloc, total_cpus, total_mem_alloc, total_mem, total_gpus_alloc, total_gpus)
    }'
    pending_jobs=$(squeue --partition $1 -t PENDING --noheader)
    if [ ! -z "$pending_jobs" ]; then
        echo
        echo Pending jobs:
        echo "$pending_jobs"
    fi
}

gpu_usage()

# Usage: gpu_usage $PARTITION_NAME