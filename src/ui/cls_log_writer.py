import sys

class StreamlitLogWriter:
    def __init__(self, ss, flush_callback=None):
        self.ss = ss
        self.flush_callback = flush_callback
        self._buffer = ""

    def write(self, text: str):
        if not text:
            return
        for ch in text:
            if ch == "\r":
                idx = self._buffer.rfind("\n")
                if idx == -1:
                    self._buffer = ""
                else:
                    self._buffer = self._buffer[: idx + 1]
            else:
                self._buffer += ch
        if "log_lines" not in self.ss:
            self.ss["log_lines"] = []
        self.ss["log_lines"] = [self._buffer]

        if self.flush_callback is not None:
            self.flush_callback(self._buffer)
        try:
            sys.__stdout__.write(text)
            sys.__stdout__.flush()
        except Exception:
            pass
        
    def flush(self):
        pass
