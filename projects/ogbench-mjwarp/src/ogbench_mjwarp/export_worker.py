"""One dataset writer consumes committed rounds while generation continues."""

import multiprocessing as mp
import queue
import traceback


def export_rounds(sources, results, output, settings):
    from .dataset import export_dataset

    try:
        result = export_dataset(
            iter(sources.get, None),
            output,
            repo_id="local/ogbench-manipulation-training",
            outcome="success",
            require_contact_valid=True,
            progress=results.put,
            **settings,
        )
        results.put({"event": "export_done", "result": result})
    except BaseException:
        results.put({"event": "export_error", "error": traceback.format_exc()})
        raise


class RoundExporter:
    def __init__(self, output, capacity=2, progress=None, **settings):
        if capacity < 1:
            raise ValueError("Export queue capacity must be positive")
        context = mp.get_context("spawn")
        self.sources = context.Queue(capacity)
        self.results = context.Queue()
        self.progress = progress
        self.result = None
        self.process = context.Process(
            target=export_rounds, args=(self.sources, self.results, output, settings)
        )
        self.process.start()

    def poll(self):
        while True:
            try:
                event = self.results.get_nowait()
            except queue.Empty:
                break
            if event["event"] == "export_error":
                raise RuntimeError(event["error"])
            if event["event"] == "export_done":
                self.result = event["result"]
            elif self.progress:
                self.progress(event)
        if self.process.exitcode is not None and self.result is None:
            raise RuntimeError(
                f"Export worker exited prematurely ({self.process.exitcode})"
            )

    def submit(self, root):
        while True:
            self.poll()
            try:
                self.sources.put(root, timeout=0.1)
                if self.progress:
                    self.progress(
                        {"event": "export_queue", "queued_rounds": self.sources.qsize()}
                    )
                return
            except queue.Full:
                pass

    def finish(self):
        self.submit(None)
        while self.result is None:
            self.process.join(timeout=0.1)
            self.poll()
        self.process.join()
        self.poll()
        return self.result

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join()
        self.sources.cancel_join_thread()
        self.sources.close()
        self.results.close()
