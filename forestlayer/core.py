# -*- coding:utf-8 -*-
"""
Forestlayer core.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import ray

REDIS_ADDRESS = "127.0.0.1:6379"


def get_redis_address():
    global REDIS_ADDRESS
    return REDIS_ADDRESS


def set_redis_address(redis_address):
    global REDIS_ADDRESS
    REDIS_ADDRESS = redis_address


# Ray 0.4.0
def init(redis_address=None, node_ip_address=None, object_id_seed=None,
         num_workers=None, driver_mode=ray.SCRIPT_MODE,
         redirect_worker_output=False, redirect_output=True,
         num_cpus=None, num_gpus=None, resources=None,
         num_custom_resource=None, num_redis_shards=None,
         redis_max_clients=None, plasma_directory=None,
         huge_pages=False, include_webui=True, object_store_memory=None):
    set_redis_address(redis_address)
    ray.init(redis_address=redis_address, node_ip_address=node_ip_address, object_id_seed=object_id_seed,
             num_workers=num_workers, driver_mode=driver_mode,
             redirect_worker_output=redirect_worker_output, redirect_output=redirect_output,
             num_cpus=num_cpus, num_gpus=num_gpus, resources=resources,
             num_custom_resource=num_custom_resource, num_redis_shards=num_redis_shards,
             redis_max_clients=redis_max_clients, plasma_directory=plasma_directory,
             huge_pages=huge_pages, include_webui=include_webui, object_store_memory=object_store_memory)
