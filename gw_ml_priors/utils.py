import time

from .logger import logger


def timing(function):
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        duration = (end_time - start_time) / 60.0
        duration_sec = (end_time - start_time) % 60.0
        f_name = function.__name__
        logger.info(
            f"{f_name} took "
            f"{int(duration)}min {int(duration_sec)}s ({duration * 60}s)"
        )

        return result

    return wrap


def print_cpu_info():
    import psutil

    logger.info("=" * 40, "CPU Info", "=" * 40)
    # number of cores
    logger.info("Physical cores:", psutil.cpu_count(logical=False))
    logger.info("Total cores:", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    logger.info(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    logger.info(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    logger.info(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    # CPU usage
    logger.info("CPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        logger.info(f"Core {i}: {percentage}%")
    logger.info(f"Total CPU Usage: {psutil.cpu_percent()}%")


def print_gpu_info():
    import GPUtil
    from tabulate import tabulate

    logger.info("=" * 40, "GPU Details", "=" * 40)
    gpus = GPUtil.getGPUs()
    list_gpus = []
    for gpu in gpus:
        # get the GPU id
        gpu_id = gpu.id
        # name of GPU
        gpu_name = gpu.name
        # get % percentage of GPU usage of that GPU
        gpu_load = f"{gpu.load * 100}%"
        # get free memory in MB format
        gpu_free_memory = f"{gpu.memoryFree}MB"
        # get used memory
        gpu_used_memory = f"{gpu.memoryUsed}MB"
        # get total memory
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        # get GPU temperature in Celsius
        gpu_temperature = f"{gpu.temperature} °C"
        gpu_uuid = gpu.uuid
        list_gpus.append(
            (
                gpu_id,
                gpu_name,
                gpu_load,
                gpu_free_memory,
                gpu_used_memory,
                gpu_total_memory,
                gpu_temperature,
                gpu_uuid,
            )
        )

    logger.info(
        tabulate(
            list_gpus,
            headers=(
                "id",
                "name",
                "load",
                "free memory",
                "used memory",
                "total memory",
                "temperature",
                "uuid",
            ),
        )
    )
