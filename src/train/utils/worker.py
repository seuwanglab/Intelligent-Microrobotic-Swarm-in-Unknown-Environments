import sys
import multiprocessing
import tblib.pickling_support
from .plugin import create_env
tblib.pickling_support.install()


class WorkerException(Exception):
    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()
        super().__init__(str(ee))

    def re_raise(self):
        raise self.ee.with_traceback(self.tb)


def worker_process(remote, config):
    try:
        env = create_env(config)
    except KeyboardInterrupt:
        return

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(env.step(data))
            elif cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Command '{cmd}' is not implemented.")
    except Exception as e:
        remote.send(WorkerException(e))
        remote.close()


class Worker:
    def __init__(self, env_config):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, env_config))
        self.process.daemon = True
        self.process.start()

    def close(self):
        self.child.send('close')
        self.child.close()
        self.process.join()
