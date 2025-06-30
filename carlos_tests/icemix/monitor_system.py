import psutil, time, json, os

# Get the output directory from environment variable, with fallback
output_dir_base = os.environ.get('OUTPUT_DIR_BASE', './monitoring')
slurm_nodeid = os.environ.get('SLURM_NODEID', 'unknown')
output_dir = f'{output_dir_base}/node_{slurm_nodeid}'

if not os.environ.get('OUTPUT_DIR_BASE') or not os.environ.get('SLURM_NODEID'):
    print(f"Warning: OUTPUT_DIR_BASE or SLURM_NODEID not set, using fallback: {output_dir}")

os.makedirs(output_dir, exist_ok=True)

print(f"System monitoring started for node {slurm_nodeid}")

try:
    with open(f'{output_dir}/psutil_host.jsonl', 'a', buffering=1) as f:
        while True:
            try:
                # Get disk I/O counters with null check
                disk_io = psutil.disk_io_counters()
                disk_read_mb = disk_io.read_bytes/2**20 if disk_io else 0.0
                disk_write_mb = disk_io.write_bytes/2**20 if disk_io else 0.0
                
                rec = {
                    "ts": time.time(),
                    "cpu_pct": psutil.cpu_percent(None),
                    "ram_used_mb": psutil.virtual_memory().used/2**20,
                    "ram_pct": psutil.virtual_memory().percent,
                    "disk_read_mb": disk_read_mb,
                    "disk_write_mb": disk_write_mb,
                }
                f.write(json.dumps(rec)+'\n')
                time.sleep(2)
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                time.sleep(5)  # Wait longer before retrying
except Exception as e:
    print(f"Failed to start system monitoring: {e}") 