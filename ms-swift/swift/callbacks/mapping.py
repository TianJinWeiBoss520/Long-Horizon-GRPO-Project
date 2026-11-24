# Copyright (c) ModelScope Contributors. All rights reserved.
try:
    from .activation_cpu_offload import ActivationCpuOffloadCallBack
except Exception:
    ActivationCpuOffloadCallBack = None
from .adalora import AdaloraCallback
from .deepspeed_elastic import DeepspeedElasticCallback, GracefulExitCallback
from .early_stop import EarlyStopCallback
from .lisa import LISACallback
from .perf_log import PerfMetricsLogCallback

callbacks_map = {
    'adalora': AdaloraCallback,
    'deepspeed_elastic': DeepspeedElasticCallback,
    'early_stop': EarlyStopCallback,
    'graceful_exit': GracefulExitCallback,
    'lisa': LISACallback,
    'perf_log': PerfMetricsLogCallback
}
if ActivationCpuOffloadCallBack is not None:
    callbacks_map['activation_cpu_offload'] = ActivationCpuOffloadCallBack
